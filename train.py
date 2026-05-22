"""Training entry point.

Run a single experiment by selecting a config section via the CONFIG_SECTION
environment variable:

    CONFIG_SECTION=EXP1_LAVA_OPTION_B python train.py [run_name]

If no run_name is given, the section name is used. TensorBoard logs go to
``runs/<run_name>/``. Models are saved under ``models/<run_name>.*``.

Supports MiniGrid (gymnasium) and Atari (gymnasium + ale-py) env types. Mario
is currently stubbed.
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Pipe
from torch.utils.tensorboard import SummaryWriter

from agents import RNDAgent
from config import default_config, default_section
from envs import (AtariEnvironment, MarioEnvironment, MiniGridEnvironment,
                  INVENTORY_DIM)
from utils import (AnchorBuffer, EpisodicCountCounter, FeatureClusterer,
                   RewardForwardFilter, RunningMeanStd, SimHashCounter,
                   make_train_data, softmax)


def main():
    print({k: v for k, v in default_config.items()})
    run_name = sys.argv[1] if len(sys.argv) > 1 else default_section
    print(f"run_name={run_name}")

    # Seed for reproducibility. SEED env var takes precedence over config.
    seed = int(os.environ.get('SEED', default_config.get('Seed', '0')))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    print(f"seed={seed}")

    env_id = default_config['EnvID']
    env_type = default_config['EnvType']

    use_cuda = default_config.getboolean('UseGPU')
    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet')
    load_model = default_config.getboolean('LoadModel', fallback=False)

    lam = float(default_config['Lambda'])
    num_worker = int(default_config['NumEnv'])
    num_step = int(default_config['NumStep'])

    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_worker / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    int_gamma = float(default_config['IntGamma'])
    clip_grad_norm = float(default_config['ClipGradNorm'])
    ext_coef = float(default_config['ExtCoef'])
    int_coef = float(default_config['IntCoef'])

    sticky_action = default_config.getboolean('StickyAction')
    action_prob = float(default_config['ActionProb'])
    life_done = default_config.getboolean('LifeDone')

    pre_obs_norm_step = int(default_config['ObsNormStep'])
    total_steps = int(default_config['TotalSteps'])

    # Option B
    use_option_b = default_config.getboolean('UseOptionB', fallback=False)
    num_ext_critics = int(default_config['NumExtCritics']) if use_option_b else 1
    bootstrap_p = float(default_config['BootstrapP'])
    gate_alpha = float(default_config['GateAlpha'])
    gate_floor = float(default_config.get('GateFloor', '0.2'))

    # DSC
    use_dsc = default_config.getboolean('UseDSC', fallback=False)
    num_anchors = int(default_config['NumAnchors'])
    anchor_percentile = float(default_config['AnchorPercentile'])
    dsc_lambda = float(default_config['DSCLambda'])
    rank_beta = float(default_config['RankBeta'])
    # DSC type-learning (K-means over anchors)
    use_clusters = default_config.getboolean('UseClusters', fallback=False)
    num_clusters = int(default_config['NumClusters']) if use_clusters else 0
    cluster_refresh_steps = int(default_config['ClusterRefreshSteps'])

    # NovelD (difference form + episodic visit-count multiplier)
    use_noveld = default_config.getboolean('UseNovelD', fallback=False)
    noveld_alpha = float(default_config.get('NovelDAlpha', '0.5'))
    use_noveld_clusters = default_config.getboolean('UseNovelDClusters', fallback=False)
    noveld_num_clusters = int(default_config.get('NovelDNumClusters', '8'))
    noveld_buffer_size = int(default_config.get('NovelDBufferSize', '256'))

    # SimHash additive bonus
    use_simhash = default_config.getboolean('UseSimHash', fallback=False)
    simhash_lambda = float(default_config.get('SimHashLambda', '0.5'))
    simhash_dim = int(default_config.get('SimHashDim', '64'))

    # Posterior sampling. Uses NumExtCritics critics; per-env active head
    # is resampled at the start of each rollout. Mutually exclusive with
    # Option B (both interpret the K critics differently).
    use_posterior_sampling = default_config.getboolean('UsePosteriorSampling', fallback=False)
    if use_posterior_sampling:
        num_ext_critics = int(default_config['NumExtCritics'])
        if use_option_b:
            raise ValueError("UsePosteriorSampling and UseOptionB are mutually exclusive")

    # MiniGrid wrapper options
    tv_on = default_config.getboolean('TVOn', fallback=False)
    tv_max_size = int(default_config['TVMaxSize'])
    p_move = float(default_config['PMove'])
    use_inventory = default_config.getboolean('UseInventory', fallback=False)
    tile_size = int(default_config['TileSize'])
    lava_penalty = float(default_config.get('LavaPenalty', '0.0'))
    inventory_dim = INVENTORY_DIM if use_inventory else 0

    # Logging / output paths
    log_dir = Path('runs') / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / f'{run_name}.model'
    predictor_path = models_dir / f'{run_name}.pred'
    target_path = models_dir / f'{run_name}.target'

    writer = SummaryWriter(log_dir=str(log_dir))

    # Dump full config + seed to log dir for rubric reproducibility.
    config_dump = {k: v for k, v in default_config.items()}
    config_dump['_run_name'] = run_name
    config_dump['_seed'] = seed
    config_dump['_config_section'] = default_section
    (log_dir / 'config.json').write_text(json.dumps(config_dump, indent=2))

    # Pick env class
    if env_type == 'atari':
        env_cls = AtariEnvironment
    elif env_type == 'mario':
        env_cls = MarioEnvironment
    elif env_type == 'minigrid':
        env_cls = MiniGridEnvironment
    else:
        raise NotImplementedError(f"EnvType={env_type}")

    # Action space size: open one env to probe
    import gymnasium as gym
    if env_type == 'minigrid':
        import minigrid  # noqa: F401
        probe = gym.make(env_id)
    elif env_type == 'atari':
        try:
            import ale_py
            gym.register_envs(ale_py)
        except ImportError:
            pass
        probe = gym.make(env_id)
    else:
        probe = gym.make(env_id)
    output_size = probe.action_space.n
    probe.close()
    input_size = (4, 84, 84)
    print(f"env={env_id}  output_size={output_size}  inventory_dim={inventory_dim}")
    print(f"use_option_b={use_option_b} K={num_ext_critics}  use_dsc={use_dsc}")

    # Agent
    agent = RNDAgent(
        input_size, output_size, num_worker, num_step, gamma,
        lam=lam, learning_rate=learning_rate, ent_coef=entropy_coef,
        clip_grad_norm=clip_grad_norm, epoch=epoch, batch_size=batch_size,
        ppo_eps=ppo_eps, use_cuda=use_cuda, use_gae=use_gae,
        use_noisy_net=use_noisy_net,
        use_option_b=use_option_b, num_ext_critics=num_ext_critics,
        bootstrap_p=bootstrap_p, gate_alpha=gate_alpha, gate_floor=gate_floor,
        use_dsc=use_dsc, num_anchors=num_anchors, dsc_lambda=dsc_lambda,
        use_noveld=use_noveld, noveld_alpha=noveld_alpha,
        inventory_dim=inventory_dim,
    )

    if load_model and model_path.exists():
        print(f'load model from {model_path}...')
        agent.model.load_state_dict(torch.load(model_path, map_location=agent.device))
        agent.rnd.load_state_dict(torch.load(predictor_path, map_location=agent.device))

    # Anchor buffer (DSC only). num_clusters=0 disables type-learning;
    # otherwise the buffer will recluster periodically and DSC will use
    # nearest-cluster lookup instead of nearest-anchor.
    anchor_buffer = AnchorBuffer(
        capacity=num_anchors, feature_dim=512,
        percentile=anchor_percentile, rank_beta=rank_beta,
        num_clusters=num_clusters) if use_dsc else None
    steps_since_recluster = 0

    # NovelD per-env episodic visit counters. Reset on real_done.
    noveld_counters = [EpisodicCountCounter() for _ in range(num_worker)] if use_noveld else None
    # Optional cluster mode: shared FeatureClusterer + per-env counter keys
    # off cluster IDs instead of position tuples.
    noveld_clusterer = (FeatureClusterer(
        buffer_size=noveld_buffer_size, feature_dim=512,
        num_clusters=noveld_num_clusters, seed=seed)
        if (use_noveld and use_noveld_clusters) else None)
    steps_since_noveld_recluster = 0

    # SimHash counter (shared across all envs; key is a deterministic hash so
    # this is per-run, not per-episode). Initialised lazily after we know the
    # obs shape.
    simhash_counter = None
    simhash_obs_dim = 84 * 84  # single-channel flattened

    # Running normalization
    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
    discounted_reward = RewardForwardFilter(int_gamma)

    # Spawn workers
    minigrid_kwargs = dict(tv_on=tv_on, tv_max_size=tv_max_size, p_move=p_move,
                            use_inventory=use_inventory, tile_size=tile_size,
                            seed=seed, lava_penalty=lava_penalty)
    works = []
    parent_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        kwargs = dict(sticky_action=sticky_action, p=action_prob, life_done=life_done)
        if env_type == 'minigrid':
            kwargs.update(minigrid_kwargs)
        work = env_cls(env_id, False, idx, child_conn, **kwargs)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)

    K = num_ext_critics
    states = np.zeros([num_worker, 4, 84, 84])
    inventories = np.zeros([num_worker, inventory_dim], dtype=np.float32) if inventory_dim > 0 else None
    # Previous-step normalized obs, used by NovelD's difference form. Initialized
    # to zeros so the first NovelD step gets max(rnd_next - alpha * rnd(zeros), 0)
    # which is close to rnd_next when rnd(zeros) is small.
    prev_norm_next_obs = np.zeros([num_worker, 1, 84, 84], dtype=np.float32)
    prev_next_inventories = (np.zeros([num_worker, inventory_dim], dtype=np.float32)
                              if (use_noveld and inventory_dim > 0) else None)

    if use_simhash:
        simhash_counter = SimHashCounter(obs_dim=simhash_obs_dim,
                                          hash_dim=simhash_dim, seed=seed)

    # Posterior-sampling per-env active head (one of NumExtCritics).
    # Resampled at the END of each rollout (which approximates per-episode
    # resampling, since episodes often span/finish within a rollout).
    active_heads = (np.random.randint(0, num_ext_critics, size=num_worker)
                    if use_posterior_sampling else None)

    # Global state-coverage tracking: union of (pos_x, pos_y, dir) keys
    # seen across all envs across all rollouts. Quantifies depth/breadth of
    # exploration over time — useful for measuring "deep exploration" gains
    # from posterior sampling.
    unique_positions_global = set()

    global_step = 0
    global_update = 0

    # ------------------------------------------------------------------
    # Observation-normalization warmup
    # ------------------------------------------------------------------
    print('Observation normalization warmup...')
    warmup_buf = []
    for step in range(num_step * pre_obs_norm_step):
        actions = np.random.randint(0, output_size, size=(num_worker,))
        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(int(action))
        for i, parent_conn in enumerate(parent_conns):
            s, r, d, rd, lr, extras = parent_conn.recv()
            warmup_buf.append(s[3, :, :].reshape([1, 84, 84]))
        if len(warmup_buf) % (num_step * num_worker) == 0:
            arr = np.stack(warmup_buf)
            obs_rms.update(arr)
            warmup_buf = []
    print('Warmup done.', flush=True)
    print(f'[main] states shape={states.shape} agent.device={agent.device}', flush=True)

    # Episode-level logging accumulators (per worker)
    episode_returns = np.zeros(num_worker)
    episode_ext_returns_recent = []  # extrinsic returns of completed episodes (rolling)
    deaths_recent = []
    goals_recent = []
    log_window = 100

    last_log_time = time.time()
    target_dtype_cast = lambda x: x.astype(np.float32)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    while global_step < total_steps:
        rollout_state = []
        rollout_next_state = []
        rollout_reward = []
        rollout_done = []
        rollout_action = []
        rollout_int_reward = []
        rollout_next_obs = []
        rollout_ext_values = []     # [num_step+1][num_worker, K]
        rollout_int_values = []
        rollout_policy = []
        rollout_inventory = []
        rollout_next_inventory = []
        rollout_gating = []

        for _ in range(num_step):
            # 1. Action
            actions, value_ext, value_int, policy = agent.get_action(
                np.float32(states) / 255., inventories)
            # value_ext: [N, K]; value_int: [N]
            # Compute gate factor at current state for Option B (used at this step's bonus)
            gate = agent.gate_factor(value_ext)

            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(int(action))

            next_states_l = []
            rewards_l = []
            dones_l = []
            real_dones_l = []
            log_rewards_l = []
            next_obs_l = []
            next_inv_l = []
            died_l = []
            goal_l = []
            pos_keys_l = []
            for i, parent_conn in enumerate(parent_conns):
                s, r, d, rd, lr, extras = parent_conn.recv()
                next_states_l.append(s)
                rewards_l.append(r)
                dones_l.append(d)
                real_dones_l.append(rd)
                log_rewards_l.append(lr)
                next_obs_l.append(s[3, :, :].reshape([1, 84, 84]))
                if inventory_dim > 0:
                    next_inv_l.append(extras.get(
                        'inventory_vec', np.zeros(inventory_dim, dtype=np.float32)))
                died_l.append(bool(extras.get('died', False)))
                goal_l.append(bool(extras.get('reached_goal', False)))
                pos_keys_l.append(extras.get('pos_key', (0, 0, 0)))

            next_states = np.stack(next_states_l)
            rewards = np.array(rewards_l, dtype=np.float32)
            dones = np.array(dones_l, dtype=np.bool_)
            real_dones = np.array(real_dones_l, dtype=np.bool_)
            log_rewards = np.array(log_rewards_l, dtype=np.float32)
            next_obs = np.stack(next_obs_l)
            next_inv = np.stack(next_inv_l) if inventory_dim > 0 else None
            died = np.array(died_l)
            goal = np.array(goal_l)

            # Global state-coverage update (cheap; ~8 set adds per step).
            for pk in pos_keys_l:
                unique_positions_global.add(pk)

            # 2. Intrinsic reward (RND base; optional NovelD difference form;
            # optional Option B gate). NovelD's visit-count multiplier and
            # SimHash additive bonus are applied below since they depend on
            # per-env state.
            norm_next_obs = ((next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)
            intrinsic_reward, target_features = agent.compute_intrinsic_reward(
                norm_next_obs, next_inv,
                prev_obs=prev_norm_next_obs if use_noveld else None,
                prev_inventory=prev_next_inventories if use_noveld else None,
                gate=gate if use_option_b else None)

            # NovelD episodic visit-count multiplier (per-env).
            if use_noveld and noveld_counters is not None:
                count_mults = np.empty(num_worker, dtype=np.float32)
                for i in range(num_worker):
                    if use_noveld_clusters and noveld_clusterer is not None:
                        cluster_id = noveld_clusterer.cluster_id(target_features[i])
                        key = ('cluster', cluster_id) if cluster_id >= 0 else ('cluster', '__none__')
                        noveld_clusterer.add(target_features[i])
                    else:
                        key = pos_keys_l[i]
                    count_mults[i] = noveld_counters[i].visit_and_multiplier(key)
                intrinsic_reward = intrinsic_reward * count_mults

            # SimHash additive bonus (per-run hash counter, shared across envs).
            if use_simhash and simhash_counter is not None:
                sh_bonuses = np.empty(num_worker, dtype=np.float32)
                for i in range(num_worker):
                    sh_bonuses[i] = simhash_counter.visit_and_bonus(
                        norm_next_obs[i].reshape(-1))
                intrinsic_reward = intrinsic_reward + simhash_lambda * sh_bonuses

            # 3. DSC bonus (distance-based, normalized by running mean).
            #    Multiplicative on top of the gated RND bonus. When type-
            #    learning is enabled (use_clusters), the nearest reference
            #    is a cluster center rather than an individual anchor.
            if use_dsc and anchor_buffer is not None:
                if anchor_buffer.filled > 0:
                    if use_clusters and anchor_buffer.cluster_filled > 0:
                        nearest_ids, dists, rank_w = anchor_buffer.nearest_cluster(target_features)
                    else:
                        nearest_ids, dists, rank_w = anchor_buffer.nearest(target_features)
                    ref_d = max(anchor_buffer.dist_ema or 1e-6, 1e-6)
                    norm_d = dists / ref_d
                    dsc_multiplier = 1.0 + dsc_lambda * norm_d * rank_w
                else:
                    dsc_multiplier = np.ones(num_worker, dtype=np.float32)
                intrinsic_reward = intrinsic_reward * dsc_multiplier
                # Maybe-insert into anchor buffer (per env). Use the
                # pre-DSC intrinsic_reward percentile threshold so the
                # selection signal isn't circular.
                for i in range(num_worker):
                    anchor_buffer.maybe_insert(
                        target_features[i], intrinsic_reward[i] / max(dsc_multiplier[i], 1e-6),
                        global_step + i)

            # 4. Track episode boundaries for logging
            episode_returns += log_rewards
            for i in range(num_worker):
                if real_dones[i]:
                    episode_ext_returns_recent.append(float(episode_returns[i]))
                    deaths_recent.append(int(died[i]))
                    goals_recent.append(int(goal[i]))
                    episode_returns[i] = 0.0
                    # NovelD episodic counter resets on real episode boundary
                    if noveld_counters is not None:
                        noveld_counters[i].reset()
            if len(episode_ext_returns_recent) > log_window:
                episode_ext_returns_recent = episode_ext_returns_recent[-log_window:]
                deaths_recent = deaths_recent[-log_window:]
                goals_recent = goals_recent[-log_window:]

            # 5. Bookkeeping
            rollout_state.append(states)
            rollout_next_state.append(next_states)
            rollout_reward.append(rewards)
            rollout_done.append(dones.astype(np.float32))
            rollout_action.append(actions)
            rollout_int_reward.append(intrinsic_reward)
            rollout_next_obs.append(next_obs)
            rollout_ext_values.append(value_ext)
            rollout_int_values.append(value_int)
            rollout_policy.append(policy)
            rollout_gating.append(gate)
            if inventory_dim > 0:
                rollout_inventory.append(inventories.copy())
                rollout_next_inventory.append(next_inv.copy())

            states = next_states
            if inventory_dim > 0:
                inventories = next_inv
            if use_noveld:
                prev_norm_next_obs = norm_next_obs
                if inventory_dim > 0:
                    prev_next_inventories = next_inv

        # Bootstrap value at the end
        _, value_ext, value_int, _ = agent.get_action(
            np.float32(states) / 255., inventories)
        rollout_ext_values.append(value_ext)
        rollout_int_values.append(value_int)

        global_step += num_worker * num_step
        global_update += 1

        # Periodic K-means recluster (DSC type-learning)
        if use_dsc and use_clusters and anchor_buffer is not None:
            steps_since_recluster += num_worker * num_step
            if steps_since_recluster >= cluster_refresh_steps and anchor_buffer.filled > 0:
                anchor_buffer.recluster()
                steps_since_recluster = 0

        # Periodic K-means recluster (NovelD cluster-types)
        if use_noveld and use_noveld_clusters and noveld_clusterer is not None:
            steps_since_noveld_recluster += num_worker * num_step
            if steps_since_noveld_recluster >= cluster_refresh_steps and noveld_clusterer.filled > 0:
                noveld_clusterer.recluster()
                steps_since_noveld_recluster = 0

        # Posterior sampling: resample active heads at the end of each
        # rollout. (Approximates per-episode resampling; could be made strictly
        # per-episode by tracking head-per-step but rollout granularity is
        # sufficient for trajectory-level diversification.)
        if use_posterior_sampling:
            active_heads = np.random.randint(0, num_ext_critics, size=num_worker)

        # ----- Reshape rollout -----
        total_state = np.stack(rollout_state).transpose([1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
        total_reward = np.stack(rollout_reward).T.clip(-1, 1)
        total_action = np.stack(rollout_action).T.reshape([-1])
        total_done = np.stack(rollout_done).T
        total_next_obs = np.stack(rollout_next_obs).transpose([1, 0, 2, 3, 4]).reshape([-1, 1, 84, 84])
        # ext values: [num_step+1, num_worker, K] -> [num_worker, num_step+1, K]
        total_ext_values = np.stack(rollout_ext_values).transpose(1, 0, 2)
        # int values: [num_step+1, num_worker]
        total_int_values = np.stack(rollout_int_values).T

        total_logging_policy = np.vstack([p.cpu().numpy() for p in rollout_policy])
        total_int_reward = np.stack(rollout_int_reward).T  # [num_worker, num_step]
        total_gating = np.stack(rollout_gating).T  # [num_worker, num_step]

        if inventory_dim > 0:
            total_inventory = np.stack(rollout_inventory).transpose(1, 0, 2).reshape([-1, inventory_dim])
            total_next_inventory = np.stack(rollout_next_inventory).transpose(1, 0, 2).reshape([-1, inventory_dim])
        else:
            total_inventory = None
            total_next_inventory = None

        # Normalize intrinsic reward per-env (running discount + std)
        rffs = np.array([discounted_reward.update(rew_step)
                         for rew_step in total_int_reward.T])
        mean, std, count = np.mean(rffs), np.std(rffs), len(rffs)
        reward_rms.update_from_moments(mean, std ** 2, count)
        total_int_reward = total_int_reward / np.sqrt(reward_rms.var)

        # --- Extrinsic targets per head (Option B) ---
        target_ext = np.zeros((num_worker, num_step, K), dtype=np.float32)
        for k in range(K):
            t_k, _ = make_train_data(total_reward, total_done,
                                      total_ext_values[:, :, k],
                                      gamma, num_step, num_worker)
            target_ext[..., k] = t_k.reshape(num_worker, num_step)
        target_ext_flat = target_ext.reshape(-1, K)

        # Extrinsic advantage. Three modes for combining the K critic heads:
        #   - Option B: pessimistic min(V_k) → conservative, suppresses positive
        #     signals on sparse-reward tasks (documented failure mode).
        #   - Posterior sampling: per-env active head's value → trajectory-level
        #     value-model diversity → "deep exploration" without suppression.
        #   - Default: mean across heads (effectively no different from single
        #     critic when K=1, but acts as a mild regularizer when K>1).
        if use_option_b:
            v_ext_combined = total_ext_values.min(axis=-1)
        elif use_posterior_sampling:
            v_ext_combined = np.zeros((num_worker, num_step + 1), dtype=np.float32)
            for i in range(num_worker):
                v_ext_combined[i] = total_ext_values[i, :, active_heads[i]]
        else:
            v_ext_combined = total_ext_values.mean(axis=-1)
        _, ext_adv = make_train_data(total_reward, total_done, v_ext_combined,
                                      gamma, num_step, num_worker)

        # Intrinsic targets (non-episodic per RND paper)
        int_target, int_adv = make_train_data(total_int_reward,
                                                np.zeros_like(total_int_reward),
                                                total_int_values, int_gamma,
                                                num_step, num_worker)

        total_adv = int_adv * int_coef + ext_adv * ext_coef

        # Update obs normalization
        obs_rms.update(total_next_obs)
        norm_next_obs_batch = ((total_next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)

        # Training
        train_diag = agent.train_model(
            np.float32(total_state) / 255., target_ext_flat, int_target,
            total_action, total_adv, norm_next_obs_batch, rollout_policy,
            inventory_batch=total_inventory,
            next_inventory_batch=total_next_inventory,
        )

        # --- Logging ---
        if episode_ext_returns_recent:
            extr = float(np.mean(episode_ext_returns_recent))
            goal_rate = float(np.mean(goals_recent))
            death_rate = float(np.mean(deaths_recent))
        else:
            extr = 0.0
            goal_rate = 0.0
            death_rate = 0.0

        writer.add_scalar('data/extrinsic_return', extr, global_step)
        writer.add_scalar('data/goal_reach_rate', goal_rate, global_step)
        writer.add_scalar('data/death_rate', death_rate, global_step)
        writer.add_scalar('data/int_reward_per_rollout',
                          float(total_int_reward.sum() / num_worker), global_step)
        writer.add_scalar('data/max_prob',
                          float(softmax(total_logging_policy).max(1).mean()), global_step)
        if use_option_b:
            writer.add_scalar('data/gating_factor',
                              float(total_gating.mean()), global_step)
            writer.add_scalar('data/ensemble_extrinsic_variance',
                              float(total_ext_values.var(axis=-1).mean()), global_step)
        if use_dsc and anchor_buffer is not None:
            writer.add_scalar('data/anchor_coverage', float(anchor_buffer.filled), global_step)
            if use_clusters:
                writer.add_scalar('data/cluster_count',
                                  float(anchor_buffer.cluster_filled), global_step)
        if use_noveld and noveld_counters is not None:
            avg_unique = float(np.mean([c.unique_count() for c in noveld_counters]))
            writer.add_scalar('data/noveld_unique_keys_per_env', avg_unique, global_step)
            if use_noveld_clusters and noveld_clusterer is not None:
                writer.add_scalar('data/noveld_cluster_count',
                                  float(noveld_clusterer.cluster_filled), global_step)
        if use_simhash and simhash_counter is not None:
            writer.add_scalar('data/simhash_unique_hashes',
                              float(simhash_counter.unique_count()), global_step)
        if use_posterior_sampling:
            # Ensemble V_ext disagreement at the active states this rollout.
            writer.add_scalar('data/posterior_v_ext_spread',
                              float(total_ext_values.std(axis=-1).mean()), global_step)
        # Deep-exploration breadth metric: total unique (x, y, dir) keys
        # visited across all envs and all rollouts so far. Higher = broader
        # coverage of the state space; useful for measuring whether
        # posterior sampling produces deeper exploration than per-step
        # curiosity alone.
        writer.add_scalar('data/unique_positions_seen',
                          float(len(unique_positions_global)), global_step)

        # PPO diagnostics (Part 2 rubric: loss curves, entropy, KL)
        if train_diag is not None:
            for k, v in train_diag.items():
                writer.add_scalar(f'train/{k}', float(v), global_step)

        if global_update % 10 == 0:
            now = time.time()
            sps = (num_worker * num_step * 10) / max(1e-6, now - last_log_time)
            print(f"[update {global_update} | step {global_step}] "
                  f"extr={extr:.3f} goal_rate={goal_rate:.3f} death_rate={death_rate:.3f} "
                  f"int_r={total_int_reward.sum()/num_worker:.3f} sps={sps:.1f}")
            last_log_time = now

        if global_update % 100 == 0:
            torch.save(agent.model.state_dict(), model_path)
            torch.save(agent.rnd.state_dict(), predictor_path)

    # Final save
    torch.save(agent.model.state_dict(), model_path)
    torch.save(agent.rnd.predictor_cnn.state_dict(), predictor_path)
    torch.save(agent.rnd.target_cnn.state_dict(), target_path)

    # Cleanup workers
    for parent_conn in parent_conns:
        parent_conn.close()
    for work in works:
        work.terminate()
        work.join(timeout=1)

    writer.close()
    print(f"Done. {global_step} steps. Logs at {log_dir}")


if __name__ == '__main__':
    # MPS requires 'spawn'; safe on CUDA/CPU too.
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
