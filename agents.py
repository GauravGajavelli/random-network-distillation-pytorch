"""RND agent extended with Option B (bootstrap-ensemble extrinsic critic +
intrinsic gating) and DSC (Discriminative Subgoal Curiosity).

The RND core is preserved: predictor and target networks stay; the intrinsic
reward is the MSE between them, augmented by Option B's variance gate (when
``use_option_b=True``) and DSC's discrimination bonus (when ``use_dsc=True``).
"""
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from model import CnnActorCriticNetwork, RNDModel, get_device
from utils import global_grad_norm_


class RNDAgent:
    def __init__(
            self,
            input_size,
            output_size,
            num_env,
            num_step,
            gamma,
            lam=0.95,
            learning_rate=1e-4,
            ent_coef=0.01,
            clip_grad_norm=0.5,
            epoch=3,
            batch_size=128,
            ppo_eps=0.1,
            update_proportion=0.25,
            use_gae=True,
            use_cuda=False,
            use_noisy_net=False,
            # Option B
            use_option_b=False,
            num_ext_critics=1,
            bootstrap_p=0.8,
            gate_alpha=0.5,
            gate_floor=0.2,
            # DSC
            use_dsc=False,
            num_anchors=64,
            dsc_lambda=0.5,
            # NovelD
            use_noveld=False,
            noveld_alpha=0.5,
            # inventory fusion
            inventory_dim=0,
    ):
        self.num_env = num_env
        self.output_size = output_size
        self.input_size = input_size
        self.num_step = num_step
        self.gamma = gamma
        self.lam = lam
        self.epoch = epoch
        self.batch_size = batch_size
        self.use_gae = use_gae
        self.ent_coef = ent_coef
        self.ppo_eps = ppo_eps
        self.clip_grad_norm = clip_grad_norm
        self.update_proportion = update_proportion
        self.device = get_device(use_cuda)

        self.use_option_b = use_option_b
        # K > 1 if any consumer of the ensemble is enabled (Option B's
        # pessimistic min OR posterior sampling per-rollout head selection).
        self.num_ext_critics = num_ext_critics if num_ext_critics > 0 else 1
        self.bootstrap_p = bootstrap_p
        self.gate_alpha = gate_alpha
        self.gate_floor = gate_floor
        # EMA running mean of ensemble variance — used to normalize the gate
        # so it can approach `gate_floor` in low-variance (well-known) regions
        # and ~1 in high-variance (novel) regions, regardless of absolute scale.
        self._var_ema = None

        self.use_dsc = use_dsc
        self.num_anchors = num_anchors
        self.dsc_lambda = dsc_lambda

        self.use_noveld = use_noveld
        self.noveld_alpha = noveld_alpha

        self.inventory_dim = inventory_dim

        self.model = CnnActorCriticNetwork(
            input_size, output_size, use_noisy_net,
            num_ext_critics=self.num_ext_critics,
            inventory_dim=inventory_dim,
        ).to(self.device)

        self.rnd = RNDModel(
            input_size, output_size, inventory_dim=inventory_dim,
        ).to(self.device)

        params = list(self.model.parameters()) + list(self.rnd.predictor_cnn.parameters()) \
            + list(self.rnd.predictor_head.parameters())

        self.optimizer = optim.Adam(params, lr=learning_rate)

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------
    def _to_t(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float().to(self.device)
        return x.to(self.device).float()

    def get_action(self, state, inventory=None):
        state_t = self._to_t(state)
        inv_t = self._to_t(inventory) if inventory is not None and self.inventory_dim > 0 else None
        with torch.no_grad():
            policy, value_ext, value_int = self.model(state_t, inv_t)
            action_prob = F.softmax(policy, dim=-1).data.cpu().numpy()
        action = self._random_choice_prob_index(action_prob)
        # value_ext shape [N, K]; value_int shape [N, 1]
        return (action,
                value_ext.detach().cpu().numpy(),
                value_int.detach().cpu().numpy().squeeze(-1),
                policy.detach())

    @staticmethod
    def _random_choice_prob_index(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)

    def gate_factor(self, value_ext_np):
        """Compute the intrinsic-reward gate factor from ensemble ext values.

        value_ext_np: [N, K] numpy array. Returns: [N] gate in [0, 1].
        With K==1 (no Option B), returns ones.

        Formula: gate = clip(alpha * var / (var + ema_var), 0, 1).
        At var=0 -> 0 (full suppression in value-certain states).
        At var=ema_var -> 0.5*alpha (typical case).
        At var>>ema_var -> alpha (saturating; clip at 1).

        The EMA-normalized form means the gate is scale-invariant and adapts
        to whatever variance scale appears during training.
        """
        if not self.use_option_b or value_ext_np.shape[1] <= 1:
            return np.ones(value_ext_np.shape[0], dtype=np.float32)
        var = value_ext_np.var(axis=1)
        batch_mean = float(var.mean())
        if self._var_ema is None:
            self._var_ema = max(batch_mean, 1e-6)
        else:
            self._var_ema = 0.99 * self._var_ema + 0.01 * batch_mean
        ref = max(self._var_ema, 1e-6)
        gate = var / (var + ref)
        # Floor at gate_floor so the intrinsic exploration signal is never
        # fully suppressed — important on sparse-reward tasks where critic
        # variance is uniformly small (no extrinsic signal to disagree about)
        # and the gate would otherwise close everywhere.
        gate = np.clip(self.gate_alpha * gate, self.gate_floor, 1.0)
        return gate.astype(np.float32)

    def _rnd_mse(self, obs, inventory):
        """Forward RND target + predictor on a single-channel obs batch.

        Returns the MSE intrinsic reward [N] and the target features [N, 512].
        Pure numpy outputs.
        """
        obs_t = self._to_t(obs)
        inv_t = self._to_t(inventory) if (inventory is not None and self.inventory_dim > 0) else None
        with torch.no_grad():
            target_feature = self.rnd.target_forward(obs_t, inv_t)
            predict_feature = self.rnd.predictor_forward(obs_t, inv_t)
            mse = (target_feature - predict_feature).pow(2).sum(1) / 2
        return mse.detach().cpu().numpy(), target_feature.detach().cpu().numpy()

    def compute_intrinsic_reward(self, next_obs, next_inventory=None, gate=None,
                                  prev_obs=None, prev_inventory=None):
        """Compute the per-step intrinsic reward.

        - Base signal: RND MSE on next_obs.
        - If use_noveld and prev_obs is provided: NovelD difference form
          `max(rnd_next - alpha * rnd_prev, 0)`. The visit-count multiplier
          and SimHash additive bonus are applied in train.py (per-env state).
        - If gate is provided (Option B): multiplied into the final signal.

        Returns (intrinsic_reward [N], target_features [N, 512]).
        """
        rnd_next, target_feature = self._rnd_mse(next_obs, next_inventory)

        if self.use_noveld and prev_obs is not None:
            rnd_prev, _ = self._rnd_mse(prev_obs, prev_inventory)
            r = np.maximum(rnd_next - self.noveld_alpha * rnd_prev, 0.0)
        else:
            r = rnd_next

        if gate is not None:
            r = r * gate

        return r, target_feature

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------
    def train_model(self, s_batch, target_ext_batch, target_int_batch, y_batch,
                    adv_batch, next_obs_batch, old_policy,
                    inventory_batch=None, next_inventory_batch=None):
        """Train one PPO epoch.

        Shapes:
          s_batch:           [B, 4, 84, 84]
          target_ext_batch:  [B, K] (K=num_ext_critics)
          target_int_batch:  [B]
          y_batch:           [B] (actions)
          adv_batch:         [B] (combined advantage)
          next_obs_batch:    [B, 1, 84, 84]
          inventory_batch:   [B, inventory_dim] or None
          next_inventory_batch: [B, inventory_dim] or None
        """
        s_batch = self._to_t(s_batch)
        target_ext_batch = self._to_t(target_ext_batch)
        target_int_batch = self._to_t(target_int_batch)
        y_batch = torch.from_numpy(y_batch).long().to(self.device)
        adv_batch = self._to_t(adv_batch)
        next_obs_batch = self._to_t(next_obs_batch)
        inv_t = self._to_t(inventory_batch) if (inventory_batch is not None and self.inventory_dim > 0) else None
        next_inv_t = self._to_t(next_inventory_batch) if (next_inventory_batch is not None and self.inventory_dim > 0) else None

        sample_range = np.arange(len(s_batch))
        forward_mse = nn.MSELoss(reduction='none')

        with torch.no_grad():
            policy_old_list = torch.stack(old_policy).permute(1, 0, 2).contiguous() \
                .view(-1, self.output_size).to(self.device)
            m_old = Categorical(F.softmax(policy_old_list, dim=-1))
            log_prob_old = m_old.log_prob(y_batch)

        K = self.num_ext_critics

        # Accumulators for diagnostic metrics returned to the caller for logging.
        diag = {'actor_loss': 0.0, 'critic_ext_loss': 0.0, 'critic_int_loss': 0.0,
                'forward_loss': 0.0, 'entropy': 0.0, 'approx_kl': 0.0}
        diag_steps = 0

        for _ in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(s_batch) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]
                idx_t = torch.from_numpy(sample_idx).long().to(self.device)

                # RND predictor loss
                target_next = self.rnd.target_forward(
                    next_obs_batch[sample_idx],
                    next_inv_t[sample_idx] if next_inv_t is not None else None)
                predict_next = self.rnd.predictor_forward(
                    next_obs_batch[sample_idx],
                    next_inv_t[sample_idx] if next_inv_t is not None else None)
                forward_loss = forward_mse(predict_next, target_next.detach()).mean(-1)
                mask = (torch.rand(len(forward_loss), device=self.device) < self.update_proportion).float()
                forward_loss = (forward_loss * mask).sum() / mask.sum().clamp(min=1.0)

                # Policy + critic
                inv_sample = inv_t[sample_idx] if inv_t is not None else None
                policy, value_ext, value_int = self.model(s_batch[sample_idx], inv_sample)
                # value_ext: [b, K]; value_int: [b, 1]

                m = Categorical(F.softmax(policy, dim=-1))
                log_prob = m.log_prob(y_batch[sample_idx])
                ratio = torch.exp(log_prob - log_prob_old[sample_idx])

                surr1 = ratio * adv_batch[sample_idx]
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_eps, 1.0 + self.ppo_eps) * adv_batch[sample_idx]
                actor_loss = -torch.min(surr1, surr2).mean()

                # Extrinsic critic loss: per-head MSE w/ per-sample bootstrap masks
                # target_ext_batch shape: [B, K]
                tgt = target_ext_batch[sample_idx]
                if K > 1:
                    # Bernoulli(p) per (sample, head) mask
                    bmask = (torch.rand_like(value_ext) < self.bootstrap_p).float()
                    per_head_loss = ((value_ext - tgt) ** 2) * bmask
                    critic_ext_loss = per_head_loss.sum() / bmask.sum().clamp(min=1.0)
                else:
                    critic_ext_loss = F.mse_loss(value_ext.squeeze(-1), tgt.squeeze(-1))

                critic_int_loss = F.mse_loss(value_int.squeeze(-1), target_int_batch[sample_idx])
                critic_loss = critic_ext_loss + critic_int_loss
                entropy = m.entropy().mean()

                loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy + forward_loss

                self.optimizer.zero_grad()
                loss.backward()
                global_grad_norm_(
                    list(self.model.parameters())
                    + list(self.rnd.predictor_cnn.parameters())
                    + list(self.rnd.predictor_head.parameters()))
                self.optimizer.step()

                # Diagnostics: mean per mini-batch step. approx_kl is the
                # standard "old vs new" KL approximation used in PPO papers.
                with torch.no_grad():
                    diag['actor_loss'] += float(actor_loss.item())
                    diag['critic_ext_loss'] += float(critic_ext_loss.item())
                    diag['critic_int_loss'] += float(critic_int_loss.item())
                    diag['forward_loss'] += float(forward_loss.item())
                    diag['entropy'] += float(entropy.item())
                    diag['approx_kl'] += float((log_prob_old[sample_idx] - log_prob).mean().item())
                diag_steps += 1

        if diag_steps > 0:
            for k in diag:
                diag[k] /= diag_steps
        return diag

