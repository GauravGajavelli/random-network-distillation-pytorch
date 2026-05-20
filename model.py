import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from torch.nn import init


def get_device(use_cuda: bool):
    """Pick the best accelerator: cuda > mps > cpu (when use_cuda=True)."""
    if not use_cuda:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class NoisyLinear(nn.Module):
    """Factorised Gaussian NoisyNet"""

    def __init__(self, in_features, out_features, sigma0=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.noisy_weight = nn.Parameter(
            torch.Tensor(out_features, in_features))
        self.noisy_bias = nn.Parameter(torch.Tensor(out_features))
        self.noise_std = sigma0 / math.sqrt(self.in_features)

        self.reset_parameters()
        self.register_noise()

    def register_noise(self):
        in_noise = torch.FloatTensor(self.in_features)
        out_noise = torch.FloatTensor(self.out_features)
        noise = torch.FloatTensor(self.out_features, self.in_features)
        self.register_buffer('in_noise', in_noise)
        self.register_buffer('out_noise', out_noise)
        self.register_buffer('noise', noise)

    def sample_noise(self):
        self.in_noise.normal_(0, self.noise_std)
        self.out_noise.normal_(0, self.noise_std)
        self.noise = torch.mm(
            self.out_noise.view(-1, 1), self.in_noise.view(1, -1))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.noisy_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            self.noisy_bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        normal_y = nn.functional.linear(x, self.weight, self.bias)
        if self.training:
            self.sample_noise()
        noisy_weight = self.noisy_weight * self.noise
        noisy_bias = self.noisy_bias * self.out_noise
        noisy_y = nn.functional.linear(x, noisy_weight, noisy_bias)
        return noisy_y + normal_y

    def __repr__(self):
        return (self.__class__.__name__
                + '(in_features=' + str(self.in_features)
                + ', out_features=' + str(self.out_features) + ')')


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class CnnActorCriticNetwork(nn.Module):
    """Policy + value networks.

    With ``num_ext_critics > 1``, the extrinsic critic becomes an ensemble
    of K linear heads on the shared trunk (Option B). Forward returns
    ``value_ext`` of shape [B, K] (so a single-head config is K=1, shape [B, 1]).

    With ``inventory_dim > 0``, an inventory one-hot vector is fused at the
    feature layer via a small linear projection.
    """

    def __init__(self, input_size, output_size, use_noisy_net=False,
                 num_ext_critics=1, inventory_dim=0):
        super().__init__()

        if use_noisy_net:
            print('use NoisyNet')
            linear = NoisyLinear
        else:
            linear = nn.Linear

        self.num_ext_critics = num_ext_critics
        self.inventory_dim = inventory_dim

        self.feature_cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            Flatten(),
        )
        cnn_out = 7 * 7 * 64

        # Optional inventory fusion path
        if inventory_dim > 0:
            self.inv_proj = nn.Linear(inventory_dim, 64)
            fused_in = cnn_out + 64
        else:
            self.inv_proj = None
            fused_in = cnn_out

        self.feature_head = nn.Sequential(
            linear(fused_in, 256),
            nn.ReLU(),
            linear(256, 448),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            linear(448, 448),
            nn.ReLU(),
            linear(448, output_size),
        )

        # Per-head MLPs for the extrinsic critic ensemble (Option B). Each head
        # has its own 2-layer hidden path so the ensemble has meaningful
        # parameter diversity beyond just final-layer initialization.
        def make_critic_head():
            return nn.Sequential(
                linear(448, 448),
                nn.ReLU(),
                linear(448, 1),
            )

        self.critic_ext_heads = nn.ModuleList(
            [make_critic_head() for _ in range(num_ext_critics)])

        # Intrinsic critic keeps the original shared-hidden + residual structure.
        self.extra_layer_int = nn.Sequential(
            linear(448, 448),
            nn.ReLU(),
        )
        self.critic_int = linear(448, 1)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        # Init each critic head's final linear with small gain (orthogonal 0.01)
        # so initial values are small; hidden layers stay at sqrt(2) for diversity.
        for head in self.critic_ext_heads:
            final = head[-1]
            init.orthogonal_(final.weight, 0.01)
            final.bias.data.zero_()
        init.orthogonal_(self.critic_int.weight, 0.01)
        self.critic_int.bias.data.zero_()

        for i in range(len(self.actor)):
            if isinstance(self.actor[i], nn.Linear):
                init.orthogonal_(self.actor[i].weight, 0.01)
                self.actor[i].bias.data.zero_()

        for i in range(len(self.extra_layer_int)):
            if isinstance(self.extra_layer_int[i], nn.Linear):
                init.orthogonal_(self.extra_layer_int[i].weight, 0.1)
                self.extra_layer_int[i].bias.data.zero_()

    def _features(self, state, inventory=None):
        x = self.feature_cnn(state)
        if self.inv_proj is not None:
            if inventory is None:
                inventory = torch.zeros(state.shape[0], self.inventory_dim,
                                         device=state.device)
            x = torch.cat([x, self.inv_proj(inventory)], dim=1)
        return self.feature_head(x)

    def forward(self, state, inventory=None):
        """Returns (policy, value_ext [B, K], value_int [B, 1])."""
        x = self._features(state, inventory)
        policy = self.actor(x)
        # Each extrinsic head has its own MLP over the shared features.
        value_ext = torch.cat([head(x) for head in self.critic_ext_heads], dim=1)
        # Intrinsic critic uses the original shared-hidden + residual structure.
        v_int_in = self.extra_layer_int(x) + x
        value_int = self.critic_int(v_int_in)
        return policy, value_ext, value_int


class RNDModel(nn.Module):
    """Frozen random target + trained predictor over single-channel observations.

    With ``inventory_dim > 0``, both the target and predictor fuse the inventory
    one-hot vector at the feature layer. Inventory is part of the state for
    novelty purposes so identical pixels with different inventory are distinct.
    """

    def __init__(self, input_size, output_size, inventory_dim=0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.inventory_dim = inventory_dim

        feature_output = 7 * 7 * 64
        in_dim = feature_output + (inventory_dim if inventory_dim > 0 else 0)

        self.predictor_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            Flatten(),
        )
        self.predictor_head = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

        self.target_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            Flatten(),
        )
        self.target_head = nn.Linear(in_dim, 512)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.target_cnn.parameters():
            param.requires_grad = False
        for param in self.target_head.parameters():
            param.requires_grad = False

    def _fuse(self, x, inventory):
        if self.inventory_dim > 0:
            if inventory is None:
                inventory = torch.zeros(x.shape[0], self.inventory_dim,
                                         device=x.device)
            x = torch.cat([x, inventory], dim=1)
        return x

    def target_forward(self, next_obs, inventory=None):
        x = self.target_cnn(next_obs)
        x = self._fuse(x, inventory)
        return self.target_head(x)

    def predictor_forward(self, next_obs, inventory=None):
        x = self.predictor_cnn(next_obs)
        x = self._fuse(x, inventory)
        return self.predictor_head(x)

    def forward(self, next_obs, inventory=None):
        target_feature = self.target_forward(next_obs, inventory)
        predict_feature = self.predictor_forward(next_obs, inventory)
        return predict_feature, target_feature


