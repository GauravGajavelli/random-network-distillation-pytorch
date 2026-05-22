from config import *
import math
from collections import deque

import numpy as np
import torch

inf = math.inf

# if default_config['TrainMethod'] in ['PPO', 'ICM', 'RND']:
#     num_step = int(ppo_config['NumStep'])
# else:
#     num_step = int(default_config['NumStep'])

use_gae = default_config.getboolean('UseGAE')
lam = float(default_config['Lambda'])
train_method = default_config['TrainMethod']


def make_train_data(reward, done, value, gamma, num_step, num_worker):
    discounted_return = np.empty([num_worker, num_step])

    # Discounted Return
    if use_gae:
        gae = np.zeros_like([num_worker, ])
        for t in range(num_step - 1, -1, -1):
            delta = reward[:, t] + gamma * value[:, t + 1] * (1 - done[:, t]) - value[:, t]
            gae = delta + gamma * lam * (1 - done[:, t]) * gae

            discounted_return[:, t] = gae + value[:, t]

            # For Actor
        adv = discounted_return - value[:, :-1]

    else:
        running_add = value[:, -1]
        for t in range(num_step - 1, -1, -1):
            running_add = reward[:, t] + gamma * running_add * (1 - done[:, t])
            discounted_return[:, t] = running_add

        # For Actor
        adv = discounted_return - value[:, :-1]

    return discounted_return.reshape([-1]), adv.reshape([-1])


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div


class AnchorBuffer:
    """Reservoir-sampled buffer of (feature, discovery_step) pairs for DSC.

    Anchors are inserted when a state's intrinsic reward exceeds the running
    `percentile`-th percentile of recent intrinsic rewards. Reservoir sampling
    keeps a uniform sample of all candidates over time, so older anchors are
    not strictly forgotten but get more diluted as more candidates appear.

    Distances are tracked in feature space (L2). Lookup is O(N) brute-force
    which is fine for N<=64 on M1 Pro. A running mean of nearest-anchor
    distances is exposed for callers that want to normalize a distance-based
    bonus to be scale-invariant.
    """

    def __init__(self, capacity=64, feature_dim=512, percentile=0.9,
                 reward_window=2000, rank_beta=0.5,
                 num_clusters=0):
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.percentile = percentile
        self.rank_beta = rank_beta
        self.features = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.discovery_step = np.zeros(capacity, dtype=np.int64)
        self.filled = 0
        self._candidates_seen = 0
        self._reward_buffer = deque(maxlen=reward_window)
        # EMA of mean nearest-anchor distance; used by callers to normalize
        # distance-based DSC bonuses.
        self.dist_ema = None

        # Cluster-based "types" (DSC type-learning extension).
        # num_clusters=0 means clustering is disabled and nearest_cluster()
        # behaves like nearest(). When >0, recluster() periodically runs
        # K-means over the filled anchors; nearest_cluster() returns the
        # nearest cluster center, the distance to it, and a rank_weight
        # derived from the mean discovery_step of cluster members.
        self.num_clusters = num_clusters
        self.cluster_centers = (np.zeros((num_clusters, feature_dim), dtype=np.float32)
                                if num_clusters > 0 else None)
        self.cluster_mean_age = (np.zeros(num_clusters, dtype=np.float64)
                                 if num_clusters > 0 else None)
        self.cluster_member_count = (np.zeros(num_clusters, dtype=np.int64)
                                     if num_clusters > 0 else None)
        self.cluster_filled = 0  # how many cluster centers are valid

    def maybe_insert(self, feature, intrinsic_reward, global_step):
        """Consider a new feature for insertion based on intrinsic reward percentile.

        Returns True if inserted.
        """
        self._reward_buffer.append(float(intrinsic_reward))
        if len(self._reward_buffer) < 50:
            return False
        threshold = np.percentile(list(self._reward_buffer), 100 * self.percentile)
        if intrinsic_reward < threshold:
            return False
        self._candidates_seen += 1
        if self.filled < self.capacity:
            idx = self.filled
            self.features[idx] = feature
            self.discovery_step[idx] = global_step
            self.filled += 1
            return True
        # Reservoir sampling: replace with probability capacity / candidates_seen
        idx = np.random.randint(0, self._candidates_seen)
        if idx < self.capacity:
            self.features[idx] = feature
            self.discovery_step[idx] = global_step
            return True
        return False

    def nearest(self, features):
        """Vectorized nearest-anchor lookup.

        features: [B, D] numpy array.
        Returns (nearest_idx [B], dists [B], rank_weights [B]).
        If buffer is empty, all return zeros / index 0 / rank_weight 1.0.
        """
        B = features.shape[0]
        if self.filled == 0:
            return (np.zeros(B, dtype=np.int64),
                    np.zeros(B, dtype=np.float32),
                    np.ones(B, dtype=np.float32))
        diff = features[:, None, :] - self.features[None, :self.filled, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))  # [B, filled]
        nearest_idx = np.argmin(dists, axis=1)
        nearest_dists = dists[np.arange(B), nearest_idx]
        # Update running mean of distances (used for normalization elsewhere).
        batch_mean = float(nearest_dists.mean())
        if self.dist_ema is None:
            self.dist_ema = max(batch_mean, 1e-6)
        else:
            self.dist_ema = 0.99 * self.dist_ema + 0.01 * batch_mean
        # rank_weight uses *relative* anchor age within the buffer:
        # newest anchor -> largest weight; oldest -> smallest weight.
        if self.filled > 1:
            ages = self.discovery_step[:self.filled]
            max_age = ages.max()
            min_age = ages.min()
            span = max(1, max_age - min_age)
            # normalized age in [0, 1]; newer = larger
            norm_age = (ages[nearest_idx] - min_age) / span
            # rank_weight emphasises newer anchors; β controls steepness
            rank_weights = (norm_age + 0.1) ** self.rank_beta
        else:
            rank_weights = np.ones(B, dtype=np.float32)
        return nearest_idx, nearest_dists.astype(np.float32), rank_weights.astype(np.float32)


class EpisodicCountCounter:
    """Per-episode visit counter used by NovelD's multiplier.

    Keys can be any hashable (position tuples, cluster IDs, ...). The
    multiplier returns 1/sqrt(N) where N is the number of visits to the
    key in the current episode. Reset clears all counts.
    """

    def __init__(self):
        self._counts = {}

    def visit_and_multiplier(self, key):
        """Increment count for key, return 1/sqrt(post-increment count)."""
        self._counts[key] = self._counts.get(key, 0) + 1
        return 1.0 / math.sqrt(self._counts[key])

    def reset(self):
        self._counts.clear()

    def unique_count(self):
        return len(self._counts)


class FeatureClusterer:
    """Online K-means clusterer for the NovelD-with-cluster-types variant.

    Maintains a reservoir-sampled buffer of feature vectors (filled
    unconditionally, distinct from DSC's percentile-thresholded
    AnchorBuffer). Periodically reclusters to produce K centroid IDs that
    the EpisodicCountCounter keys off of.
    """

    def __init__(self, buffer_size=256, feature_dim=512, num_clusters=8, seed=0):
        self.buffer_size = buffer_size
        self.feature_dim = feature_dim
        self.num_clusters = num_clusters
        self.features = np.zeros((buffer_size, feature_dim), dtype=np.float32)
        self.filled = 0
        self._candidates_seen = 0
        self._rng = np.random.RandomState(seed)

        self.cluster_centers = np.zeros((num_clusters, feature_dim), dtype=np.float32)
        self.cluster_filled = 0

    def add(self, feature):
        """Reservoir-sample one feature into the buffer."""
        self._candidates_seen += 1
        if self.filled < self.buffer_size:
            self.features[self.filled] = feature
            self.filled += 1
            return
        idx = self._rng.randint(0, self._candidates_seen)
        if idx < self.buffer_size:
            self.features[idx] = feature

    def add_batch(self, features):
        for f in features:
            self.add(f)

    def recluster(self):
        if self.filled == 0:
            return
        K = min(self.num_clusters, self.filled)
        centers, _assign = _kmeans(self.features[:self.filled], K,
                                    seed=int(self._rng.randint(1 << 30)))
        self.cluster_centers[:K] = centers
        self.cluster_filled = K

    def cluster_id(self, feature):
        """Return nearest cluster ID for a query feature.

        Returns -1 if no clusters exist yet (caller should fall back to a
        default key like '__no_cluster__').
        """
        if self.cluster_filled == 0:
            return -1
        diff = self.cluster_centers[:self.cluster_filled] - feature[None, :]
        return int(np.argmin((diff * diff).sum(axis=1)))


class SimHashCounter:
    """Random-projection-based pseudo-count for SimHash exploration.

    Projects an observation through a fixed random matrix, takes the sign
    to get a binary hash, and counts occurrences. Bonus is 1/sqrt(count).
    Per-counter (not per-episode) so it provides a quasi-episodic novelty
    signal that persists across the run.
    """

    def __init__(self, obs_dim, hash_dim=64, seed=0):
        rng = np.random.RandomState(seed)
        # Random projection matrix; fixed for the lifetime of the counter.
        self.proj = rng.randn(obs_dim, hash_dim).astype(np.float32)
        self._counts = {}

    def _hash(self, obs_flat):
        """Convert a flat observation vector into a hashable bytes key."""
        proj = obs_flat @ self.proj
        bits = (proj > 0).astype(np.uint8)
        return bits.tobytes()

    def visit_and_bonus(self, obs_flat):
        key = self._hash(obs_flat)
        self._counts[key] = self._counts.get(key, 0) + 1
        return 1.0 / math.sqrt(self._counts[key])

    def unique_count(self):
        return len(self._counts)


def _kmeans(X, K, n_iters=20, seed=0):
    """Brute-force K-means for small data. Returns (centers, assignments).

    X: [N, D] feature matrix.
    K: number of clusters.
    n_iters: max Lloyd's algorithm iterations.
    """
    rng = np.random.RandomState(seed)
    N, D = X.shape
    K = min(K, N)
    if K == 0:
        return np.zeros((0, D), dtype=X.dtype), np.zeros(N, dtype=np.int64)
    centers = X[rng.choice(N, size=K, replace=False)].copy()
    assignments = np.zeros(N, dtype=np.int64)
    for _ in range(n_iters):
        # Assign each point to nearest center
        diff = X[:, None, :] - centers[None, :, :]
        dists = (diff * diff).sum(axis=2)
        new_assignments = np.argmin(dists, axis=1)
        # Update centers
        new_centers = centers.copy()
        for k in range(K):
            mask = (new_assignments == k)
            if mask.any():
                new_centers[k] = X[mask].mean(axis=0)
        if np.array_equal(new_assignments, assignments) and np.allclose(centers, new_centers, atol=1e-4):
            assignments = new_assignments
            centers = new_centers
            break
        assignments = new_assignments
        centers = new_centers
    return centers, assignments


def _patch_anchor_buffer_with_clusters():
    """Attach recluster() and nearest_cluster() methods to AnchorBuffer.

    Done as a post-hoc patch so the class definition above stays focused on
    the core buffer mechanics, and the clustering extension can be removed
    cleanly by deleting this patch block.
    """

    def recluster(self):
        """Recompute K-means cluster centers and per-cluster mean discovery age.

        No-op if clustering is disabled (num_clusters=0) or no anchors yet.
        """
        if self.num_clusters == 0 or self.filled == 0:
            return
        K = min(self.num_clusters, self.filled)
        centers, assignments = _kmeans(self.features[:self.filled], K)
        self.cluster_centers[:K] = centers
        self.cluster_filled = K
        for k in range(K):
            mask = (assignments == k)
            if mask.any():
                self.cluster_mean_age[k] = float(self.discovery_step[:self.filled][mask].mean())
                self.cluster_member_count[k] = int(mask.sum())
            else:
                self.cluster_mean_age[k] = 0.0
                self.cluster_member_count[k] = 0

    def nearest_cluster(self, features):
        """Cluster-based nearest lookup. Returns (cluster_idx, dists, rank_weights).

        Falls back to anchor-based nearest() if clustering is disabled or
        no cluster centers are populated yet.
        """
        if self.num_clusters == 0 or self.cluster_filled == 0:
            return self.nearest(features)
        B = features.shape[0]
        diff = features[:, None, :] - self.cluster_centers[None, :self.cluster_filled, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))  # [B, cluster_filled]
        nearest_idx = np.argmin(dists, axis=1)
        nearest_dists = dists[np.arange(B), nearest_idx]
        # Update running mean of distances (parallel to the anchor case).
        batch_mean = float(nearest_dists.mean())
        if self.dist_ema is None:
            self.dist_ema = max(batch_mean, 1e-6)
        else:
            self.dist_ema = 0.99 * self.dist_ema + 0.01 * batch_mean
        # rank_weight based on cluster mean age — newer clusters higher weight
        if self.cluster_filled > 1:
            ages = self.cluster_mean_age[:self.cluster_filled]
            max_age = ages.max()
            min_age = ages.min()
            span = max(1.0, max_age - min_age)
            norm_age = (ages[nearest_idx] - min_age) / span
            rank_weights = (norm_age + 0.1) ** self.rank_beta
        else:
            rank_weights = np.ones(B, dtype=np.float32)
        return nearest_idx, nearest_dists.astype(np.float32), rank_weights.astype(np.float32)

    AnchorBuffer.recluster = recluster
    AnchorBuffer.nearest_cluster = nearest_cluster


_patch_anchor_buffer_with_clusters()


def global_grad_norm_(parameters, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

    return total_norm
