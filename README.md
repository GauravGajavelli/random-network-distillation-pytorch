# RND Exploration Enhancement Study

**Fork of [jcwleo/random-network-distillation-pytorch](https://github.com/jcwleo/random-network-distillation-pytorch)** — extends the original RND+PPO implementation with MiniGrid environments and a systematic study of exploration enhancements for sparse-reward tasks.

## Main Contribution

Four exploration enhancements to Random Network Distillation (Burda et al. 2018) are implemented and evaluated on MiniGrid proxies for the sparse-reward Atari failures (Pitfall, Gravitar) documented in the original paper:

| Enhancement | Description | Outcome |
|---|---|---|
| **Option B** | K=5 ensemble extrinsic critics + variance-gated intrinsic reward | Wins on Pitfall proxy; fails on Gravitar proxy — mechanism validated, not universally better |
| **NovelD + cluster types** | Episodic count bonus on K-means cluster IDs over RND target features | Best on small envs (Lava, DoorKey); adds noise on KeyCorridor where vanilla RND already works |
| **SimHash + RND** | Additive random-projection pseudo-count on top of RND intrinsic signal | **16.5% faster convergence** on KeyCorridor (n=3 seeds, consistent across seeds) |
| **Posterior Sampling** | Per-rollout head sampling on K-critic ensemble for trajectory-level exploration | Useful alone; catastrophic when stacked with NovelD |

**Key finding**: SimHash+RND converges to goal in 267.6k ± 12.7k steps vs vanilla RND's 320.5k ± 3.6k (n=3 matched seeds, `MiniGrid-KeyCorridorS3R1-v0`). A three-seed ablation (SimHash-only, `UseRNDBonus=False`) shows the hash bonus alone completely fails — 0.000 extrinsic return at 1.5M steps despite 1.2M unique hash observations — establishing that RND provides the essential directional signal and SimHash provides complementary coverage.

A structural constraint analysis deriving five properties (C1–C5) any RND enhancement must satisfy is in [`reports/rnd_enhancement_constraints.md`](reports/rnd_enhancement_constraints.md).

## Environments

MiniGrid proxies for the known Atari failure modes:

| Exp | Environment | Proxy for | Budget |
|---|---|---|---|
| 1 | `MiniGrid-LavaCrossingS9N2-v0` | Pitfall — curiosity attracts agent to deadly states | 1M steps |
| 2 | `MiniGrid-DoorKey-5x5-v0` | Gravitar — intrinsic distracts from mature extrinsic reward | 1M steps |
| 3 | `MiniGrid-KeyCorridorS3R1-v0` | Multi-room navigation — primary benchmark | 1.5M steps |
| 4 | `MiniGrid-KeyCorridorS3R1-v0` | Posterior sampling deep-exploration test | 1.5M steps |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.8+, PyTorch 2.x, `minigrid`, `tensorboard`. For CUDA machines, see `cuda/setup.sh`.

## Running Experiments

Select a section from `config.conf` and pass it as an environment variable:

```bash
CONFIG_SECTION=EXP3_KEYCORRIDOR_BASELINE python train.py exp3_keycorridor_baseline
```

Run scripts for multi-seed comparisons are in `scripts/`:

```bash
# 3-way SimHash comparison — vanilla RND vs SimHash+RND vs SimHash-only (3 seeds each)
bash scripts/run_simhash_vs_vanilla.sh a   # vanilla RND seeds 1+2
bash scripts/run_simhash_vs_vanilla.sh b   # SimHash+RND seeds 1+2
bash scripts/run_simhash_vs_vanilla.sh c   # SimHash-only seeds 0+1+2

# NovelD ablation — position keys vs cluster keys
bash scripts/run_noveld_ablation.sh a      # LavaCrossing seeds
bash scripts/run_noveld_ablation.sh b      # KeyCorridor seeds
```

All experiment sections are defined in `config.conf` under `[EXP1_*]` through `[EXP4_*]`.

## Evaluation

```bash
python scripts/eval_simhash.py     # 3-way SimHash comparison (vanilla / +RND / alone)
python scripts/eval_summary.py     # Cross-method summary across all experiments
tensorboard --logdir runs/         # TensorBoard — all active experiment logs
```

## Directory Structure

```
runs/                        # TensorBoard event logs for all experiments
  exp1_lava_*/               # Exp 1: LavaCrossing (Pitfall proxy)
  exp2_doorkey_*/            # Exp 2: DoorKey (Gravitar proxy)
  exp3_keycorridor_*/        # Exp 3: KeyCorridor — primary benchmark
    exp3_keycorridor_baseline*          # Vanilla RND (3 seeds)
    exp3_keycorridor_simhash*           # SimHash+RND (3 seeds)  ← key result
    exp3_keycorridor_simhash_only_seed* # SimHash-only ablation (3 seeds)
    exp3_keycorridor_noveld*            # NovelD variants
    exp3_keycorridor_simhash_tv*        # SimHash + noisy TV diagnostic
  exp4_keycorridor_*/        # Exp 4: Posterior sampling
  runs_1/                    # Archive: DSC experiment logs (earlier sweep)
  runs_2/                    # Archive: DSC experiment logs (second sweep)
  models_1/                  # Archive: DSC experiment checkpoints (earlier sweep)
  models_2/                  # Archive: DSC experiment checkpoints (second sweep)

models/                      # Model checkpoints for active experiments
  exp3_keycorridor_simhash.{model,pred,target}          # SimHash+RND seed 0
  exp3_keycorridor_simhash_seed{1,2}.{model,pred,target}
  exp3_keycorridor_simhash_only_seed{0,1,2}.{model,...} # SimHash-only ablation
  exp3_keycorridor_baseline*.{model,pred,target}         # Vanilla RND (3 seeds)
  ...                        # All active checkpoints follow exp<N>_<env>_<method> naming

scripts/                     # Run and evaluation scripts
  eval_simhash.py            # 3-way SimHash evaluation (primary result)
  eval_summary.py            # Cross-method summary
  run_simhash_vs_vanilla.sh  # Multi-seed SimHash run script
  run_noveld_ablation.sh     # NovelD position-key vs cluster-key comparison

reports/                     # Analysis documents
  rnd_enhancement_constraints.md   # C1–C5 structural constraints (main analysis)
```

## References

[1] Burda et al. (2018) [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)  
[2] Tang et al. (2017) [#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning](https://arxiv.org/abs/1611.04717) — SimHash basis  
[3] Zhang et al. (2021) [NovelD: A Simple yet Effective Exploration Criterion](https://arxiv.org/abs/2110.09527)  
[4] Osband et al. (2016) [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621)  
[5] Schulman et al. (2017) [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)  
[6] Chevalier-Boisvert et al. [MiniGrid](https://github.com/Farama-Foundation/Minigrid)  

---

*Upstream: [jcwleo/random-network-distillation-pytorch](https://github.com/jcwleo/random-network-distillation-pytorch)*
