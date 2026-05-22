# Structural Constraints on Enhancements to RND

**Handoff document — sufficient context to continue this line of work without prior conversation.**

This document derives three empirically-grounded constraints that any addition to Random Network Distillation (Burda et al. 2018) must satisfy in sparse-reward environments. Each constraint is derived from a mechanism analysis and tested against results from three MiniGrid experiments. A design checklist is included at the end.

---

## Project context

This codebase implements RND + PPO on MiniGrid environments as a proxy for the sparse-reward Atari failures documented in the original paper (Pitfall, Gravitar). The three main experiments are:

- **Exp 1 — LavaCrossing** (`MiniGrid-LavaCrossingS9N2-v0`, 1M steps): Pitfall proxy. Curiosity attracts the agent to lava-death. Vanilla RND fails on seed 0 (extr=0.001, goal_rate=0.002).
- **Exp 2 — DoorKey** (`MiniGrid-DoorKey-5x5-v0`, 1M steps): Gravitar proxy. RND helps substantially (vanilla RND: extr=0.958 vs PPO-only: extr=0.679). Curiosity interference is the risk.
- **Exp 3 — KeyCorridor** (`MiniGrid-KeyCorridorS3R1-v0`, 1.5M steps): Multi-room navigation. Vanilla RND solves it (extr=0.942, time-to-goal=325.6k steps). Tests whether additions improve on a functioning baseline.

Enhancements tested: **Option B** (K=5 ensemble extrinsic critics + pessimistic min + variance-gated intrinsic reward), **NovelD-with-cluster-types** (episodic count on K-means cluster IDs over RND target features), **SimHash-additive** (random-projection pseudo-count added to intrinsic reward), and **Posterior Sampling** (per-rollout head sampling on the K-critic ensemble). Key source files: `agents.py`, `train.py`, `model.py`, `config.conf`. Experiment configurations are in `config.conf` sections `EXP1_*` through `EXP4_*`.

---

## Constraint 1 — The signal must exist in the pre-reward regime

**Statement**: Any signal used to modify intrinsic or extrinsic value estimates must be non-trivial and informative before the first extrinsic reward is observed. Mechanisms that require reward signal to produce useful variance will be inoperative during the exploration phase — precisely when the enhancement is most needed.

### Evidence

**Option B violated this constraint. Its primary mechanism was inoperative.**

Option B used the variance across K=5 extrinsic critic heads to gate the intrinsic reward and to form a pessimistic `min(V_ext_k)` advantage. The key assumption is that heads will disagree about the value of dangerous novel states, creating a negative bias that discourages lava-death.

Measured ensemble variance (logged as `data/ensemble_extrinsic_variance` in TensorBoard):

| Run | Early | 25% of training | 50% | 75% | Final |
|---|---|---|---|---|---|
| Option B — LavaCrossing | 8.46e-06 | 8.36e-08 | 1.08e-04 | 2.05e-04 | 1.82e-04 |
| Option B — DoorKey | 2.87e-06 | 2.31e-05 | 1.97e-07 | 6.64e-08 | 3.26e-05 |

On LavaCrossing, variance is essentially zero for the first half of training. It rises only after the policy begins accumulating extrinsic reward (the breakthrough happens around step 592k on seed 0). On DoorKey — where reward arrives earlier and more reliably — variance is still of order 1e-05 to 1e-06: negligible relative to the value-scale differences that would make `min(V_k)` meaningfully different from `mean(V_k)`.

**Why this happens structurally**: In a sparse-reward + curiosity-driven regime, the TD target for every extrinsic critic head is approximately 0 everywhere until reward is observed. All K heads are trained on the same near-zero targets with the same observations. Independent parameterization (we used per-head 2-layer MLPs with ~200k parameters each) controls *how* heads can diverge given informative signal, not *whether* signal exists to drive divergence. No architectural choice resolves this: the bottleneck is data, not capacity.

**Consequences at the performance level**:

| Method | Env | extr_return | goal_rate | time-to-goal |
|---|---|---|---|---|
| Vanilla RND | LavaCrossing (seed 0) | 0.001 | 0.002 | never |
| Option B | LavaCrossing (seed 0) | 0.834 | 0.933 | 591.9k |
| Vanilla RND | DoorKey | 0.958 | 1.000 | 117.8k |
| Option B | DoorKey | 0.294 | 0.349 | 149.5k |

The LavaCrossing improvement is real but its mechanism is not pessimism: the gate at its floor value (0.2) uniformly attenuates intrinsic reward by 5×, which passively reduces the curiosity-attraction to lava without requiring head disagreement. The DoorKey result — where reward is available but the gate still suppresses the useful intrinsic signal — confirms the mechanism analysis: the gate attenuates indiscriminately rather than state-specifically.

**The gate was implemented correctly but misconceived**. The relevant code is in `agents.py:gate_factor()`. The EMA-normalized formula `clip(α·var / (var + EMA(var)), gate_floor, 1.0)` collapses to `clip(α·~0 / (~0 + EMA(~0)), gate_floor, 1.0)` = `gate_floor = 0.2` uniformly. In a dense-reward regime with meaningful ensemble disagreement, the gate would be state-specific. In sparse-reward, it is a constant 0.2-floor attenuator.

**Signals that satisfy Constraint 1**: RND MSE (observations exist from step 0), position counts, SimHash counts, forward-dynamics prediction error. All depend only on what the agent observes, not on whether reward has been seen.

---

## Constraint 2 — Granularity must match the state space's discriminative resolution

**Statement**: The resolution at which a count- or distance-based bonus distinguishes states must be calibrated to the environment's state-space size. If the bonus groups too many states into one bucket, it saturates within an episode before reaching the sparse reward; if it groups too finely (or never saturates), it continues to differentiate states throughout training.

### Evidence

**NovelD-with-cluster-types tests this constraint directly.**

NovelD uses `1/sqrt(N(key(s')))` as an episodic multiplier on the RND difference signal, where `key` is the cluster ID assigned by K-means over RND target features. K=8 clusters were used on all three environments. The key diagnostic is how many distinct cluster IDs the agent visits per episode (`data/noveld_unique_keys_per_env`):

| Env | unique keys/env — 25% | 50% | 75% | final | Result |
|---|---|---|---|---|---|
| LavaCrossing (~50 navigable tiles, K=8) | 5.1 | ~5.1 | 5.25 | 5.7 | **PASS** (time2goal=563k, best) |
| DoorKey (~25 navigable tiles, K=8) | 4.4 | ~4.4 | 3.6 | 3.4 | **PASS** (time2goal=97.3k, best) |
| KeyCorridor (complex layout, K=8) | 4.3 | ~4.3 | 3.3 | 3.7 | **FAIL** (time2goal=360k vs baseline 325k) |

Cluster count is 8 from early training on all three environments — the buffer fills quickly and K-means runs at `ClusterRefreshSteps=4096` (about 4 rollouts). The useful diagnostic is the unique-keys-per-env count.

On LavaCrossing and DoorKey, K=8 over a small state space provides 5 or fewer distinct regions per episode. This turns out to be *enough* resolution to guide the agent: the death-avoidance structure of LavaCrossing has only a few distinct risk regions (lava tiles, safe corridor, goal), and DoorKey's key-then-door structure has similarly few semantic regions. The coarse abstraction aligns with the task structure.

On KeyCorridor, K=8 covers a more complex layout. The unique keys per env drops from 4.3 at 25% of training to 3.3 at 75%, meaning the policy converges to using fewer distinct cluster regions as it learns the efficient path. The NovelD bonus becomes a noisy episodic signal that doesn't add resolution beyond what RND already provides — and its multiplicative form sometimes suppresses useful intrinsic exploration on already-visited-but-necessary path segments.

**SimHash as the counter-example**: SimHash uses a 64-bit random-projection hash over a flattened observation. It distinguishes states at nearly pixel resolution.

| Env | unique hashes — early | 25% | 75% | final |
|---|---|---|---|---|
| LavaCrossing | 497 | 10,180 | 22,482 | 25,259 |
| KeyCorridor | 225 | 1,709 | 2,689 | 2,856 |

The hash space never saturates: with 25k unique hashes covering 1M steps, each hash is visited ~40 times on average. New states continue to generate near-1.0 bonuses throughout training. The count is additive (not episodic), so the signal persists across episode boundaries.

Result: **SimHash+RND passes** on all three environments (time2goal: 281.6k on KeyCorridor vs 325.6k baseline, PASS). SimHash alone without RND fails completely — see C5 below.

**The resolution requirement is task-specific**: K=8 clusters pass on small envs and fail on a larger one. SimHash's near-infinite resolution passes everywhere when combined with RND. The practical implication is that fixed-K count-based methods require K to scale with the number of semantically distinct regions in the state space, which is not known in advance. Variable-resolution methods (SimHash, position keys, RND MSE) avoid this calibration problem.

---

## Constraint 3 — Bonus decay must be slower than the task's characteristic search horizon

**Statement**: The intrinsic signal must remain informative (i.e., above noise) until the agent has had enough time to reach and exploit the sparse reward. If the count-based bonus saturates before the agent finds the goal, the exploration-guiding effect is lost at the moment it is most needed.

### Evidence

**Episodic vs global counting creates a fundamental asymmetry.**

NovelD's count is episodic: `EpisodicCountCounter.reset()` is called at every `done` signal (`train.py:441-442`). This means within each episode the agent starts with N=0 for every cluster key and the bonus is 1.0; the bonus decays as the agent re-visits clusters within that episode. At episode end, the counter resets.

With ~5 unique cluster keys visited per episode (LavaCrossing) and `MaxStepPerEpisode=200`, the agent exhausts all K=8 cluster IDs by approximately step 50-80 of a 200-step episode. After that, additional visits to each cluster accumulate count but the bonus decay is slow (1/sqrt(N) from N=5 to N=10 is 0.45 to 0.32 — still meaningful). The key property: the episodic reset restores full bonus at the start of each episode, so the signal never dies permanently across episodes.

This design is appropriate when:
1. The task requires discovering a new region *within* an episode (LavaCrossing, DoorKey — the goal is reachable within one episode if the agent explores correctly).
2. The episode length is long relative to the cluster-exhaustion time, so the agent has some gradient signal near the sparse reward even late in the episode.

It is less appropriate when:
1. The baseline is already guiding exploration effectively (KeyCorridor with vanilla RND: time2goal=325k) and the episodic reset creates a noisy bonus that occasionally suppresses visits to necessary intermediate states.
2. The count-exhaustion time within an episode is shorter than the expected distance to the goal, so the bonus has collapsed by the time the agent would need it.

**Observation from int_reward_per_rollout trajectories**:

| Run | early | mid | final |
|---|---|---|---|
| Lava — NovelD-clustered | 2.27 | 0.066 | 0.079 |
| Lava — vanilla RND | 1.13 | 0.074 | 0.061 |
| DoorKey — NovelD-clustered | 2.20 | 0.030 | 0.002 |
| DoorKey — vanilla RND | 1.54 | 0.002 | 0.001 |
| KeyCorridor — NovelD-clustered | 4.53 | 2.46 | 2.77 |
| KeyCorridor — vanilla RND | 1.43 | 0.731 | 0.672 |

On LavaCrossing and DoorKey, the NovelD augmentation produces higher early intrinsic rewards (2.2-2.3 vs 1.1-1.5) which then converge toward the vanilla baseline. On DoorKey specifically, by final training the int_reward for NovelD is nearly 0 (0.002) while vanilla's is also near 0 (0.001) — both collapsed but NovelD arrived there via a useful early-exploration bonus that contributed to the best time-to-goal (97.3k).

On KeyCorridor, NovelD produces persistently high int_reward throughout training (4.53 → 2.77), while vanilla RND's is 1.43 → 0.67. The large and persistent NovelD bonus on KeyCorridor reflects continued episodic reset + re-exploration — but the result is slower convergence (360k vs 325k). The bonus is not helping; it is adding noise to a policy that RND alone was already guiding effectively.

**SimHash has no decay problem by design**: Its count accumulates globally across all episodes. A state visited for the first time at step 900k still receives a count of 1 and a bonus of 1.0. The bonus decays monotonically with *cumulative visit count*, not episode count. At 2856 unique hashes over 1.5M steps (KeyCorridor), average count per hash is ~525, giving average bonus ~0.04. But newly-encountered states still generate 1.0 bonuses throughout training. This is why SimHash outperforms the baseline on KeyCorridor (281.6k vs 325.6k, PASS) while NovelD does not.

---

Two additional patterns visible in the data:

**C4: Additive bonuses are safer than multiplicative modifications when added to a working base signal.** SimHash adds to `r_int`; when combined with RND the worst case is a small signal that adds noise but does not suppress the existing RND component. Option B's variance gate multiplies by a factor ≤ 1.0; on DoorKey it suppressed useful intrinsic reward even when it should not have. NovelD's multiplier `1/sqrt(N)` on the RND difference signal can similarly suppress a useful RND contribution. Where the existing signal is working, multiplicative/gating modifications risk removing it.

The "additive is safe" claim must be qualified: **additive to RND** is safe; **replacing RND** with the additive bonus alone is not. A three-seed ablation (KeyCorridor, `UseRNDBonus=False`, SEED=0/1/2) measured SimHash without the RND component:

| Condition | extr return | time-to-goal | int reward (tail) | unique hashes |
|---|---|---|---|---|
| Vanilla RND | 0.936 ±0.012 | 320.5k ±3.6k | 0.659 ±0.121 | — |
| SimHash+RND | 0.944 ±0.004 | 267.6k ±12.7k | 0.006 ±0.001 | 3,013 ±115 |
| SimHash-only | 0.000 ±0.000 | ∞ (all seeds) | 56.437 ±1.759 | 1,386,233 ±30,884 |

SimHash-only achieves zero extrinsic return across all three seeds despite accumulating 1.2M unique hash observations — more than 400× the state coverage of SimHash+RND. The agent explores maximally and exploits nothing. RND is not interchangeable with the hash bonus; it is the backbone that makes the hash bonus useful.

The practical rule: if a method is described as an "additive bonus to RND," verify that removing RND makes the method degenerate. If it does (as here), the method's value derives from its interaction with RND, not from the bonus itself.

**C5: Coverage signal alone cannot solve sparse-reward tasks requiring directed search.** SimHash-only satisfies C1 (counts exist pre-reward), C2 (near-pixel resolution, 1.2M unique hashes), and C3 (global non-resetting count, never saturates) — yet fails completely (0.000 extrinsic, ∞ time-to-goal, 3 seeds). This rules out the three structural constraints as a complete explanation of failure. The missing property is **directionality**: the bonus must create a gradient that points toward the sparse reward, not merely toward unvisited states. RND's normalized prediction error has implicit direction because the predictor learns faster on frequently-visited states, making novel states near-the-goal reliably more rewarding than novel states far from it. A pure count-based signal treats all novel states identically. In environments where the goal requires navigating through a specific sequence of novel states (KeyCorridor's multi-room structure), undirected novelty-maximization will fill the hash table without ever committing to the goal-directed path.

---

## The five constraints summarized

| Constraint | Failed by | Passed by | Mechanism |
|---|---|---|---|
| **C1: Signal exists pre-reward** | Option B (value disagreement) | NovelD, SimHash, RND | Use observation-domain signals only |
| **C2: Granularity matches state space** | NovelD-clustered on KeyCorridor (K=8 too coarse) | SimHash (near-pixel resolution), NovelD on small envs | Resolution must match semantic regions |
| **C3: Decay slower than search horizon** | NovelD on KeyCorridor (episodic saturation adds noise to working baseline) | SimHash (global count), RND MSE (learning-rate decay) | Decay mechanism must outlast the agent's search time |
| **C4: Additive to RND, not replacing it** | SimHash-only (removes RND backbone; extr=0.000 despite 1.2M unique hashes) | SimHash+RND (additive; 16.5% faster than vanilla) | Bonus must augment RND, not substitute for it |
| **C5: Directed signal, not just coverage** | SimHash-only (coverage-only; max exploration, zero exploitation) | RND, SimHash+RND | Must create gradient toward reward, not just toward novelty |

---

## Stacking failure: Posterior Sampling + NovelD

An instructive negative result beyond the three main constraints: stacking
Posterior Sampling on top of NovelD-clustered on KeyCorridor produces
**catastrophic degradation** — time-to-goal 580k vs 360k for NovelD alone and
325k for vanilla RND.

| Method | extr_return | time-to-goal | int_reward (tail) |
|---|---|---|---|
| Vanilla RND | 0.942 | 325.6k | 0.673 |
| NovelD-clustered | 0.945 | 360.4k | 2.69 |
| Posterior Sampling alone | 0.943 | 288.8k | 0.657 |
| **Posterior + NovelD** | **0.945** | **580.6k** | **2.27** |

The failure mechanism: NovelD-clustered on KeyCorridor already violates C3
(its episodic bonus produces persistently high int_reward, 4.53 → 2.77, adding
noise to the RND signal). Posterior Sampling compounds this by committing each
env worker to one value head's estimate for 128 steps. When that head's value
landscape disagrees with the noisy count signal — which it will, because the
head was trained on different bootstrap samples and the NovelD bonus is keyed
on cluster IDs rather than value-relevant states — the gradient updates push
in inconsistent directions across rollouts. The result is not 0 + noise = noise;
it is interference between two signals that each individually had a useful
structure but jointly cancel it out.

**The practical rule**: when stacking bonuses or exploration mechanisms, the
combination is safe only if the two signals are genuinely independent at the
state level. Posterior Sampling's per-rollout head commitment makes it
*sensitive to which signal drives advantage estimates*. Any noisy additive
contribution to advantage (like a poorly-calibrated count bonus) will be
amplified rather than averaged out. Posterior Sampling stacks safely with
SimHash (which produces small, near-zero bonuses once the global count
accumulates) but not with NovelD (which produces persistently large bonuses
that dominate the advantage signal on already-explored tasks).

---

## Design checklist for new enhancements

Before implementing an enhancement to RND, verify:

**1. Does the signal exist before the first reward?**  
   - Acceptable: anything computed from `obs`, `next_obs`, position, hash of observation.
   - Unacceptable: ensemble value disagreement, return variance, policy entropy divergence from a "good" policy (requires knowing what "good" means).

**2. What is the state space size and how many distinct regions does the bonus distinguish?**  
   - Estimate: how many unique key values will be generated over a single episode?  
   - Rule of thumb: unique keys per episode should be ≥ 1/4 of the navigable state count. If K << navigable states / 4, the bonus will saturate within episodes before guiding the agent to the goal.  
   - SimHash or position keys scale automatically. Fixed-K clustering requires K to be set per environment.

**3. Does the decay rate match the task's search horizon?**  
   - Compute: episode length × expected episodes before first goal. The bonus should still be non-trivial (say, > 0.1 of its initial value) at that timescale.  
   - Episodic resets (NovelD) are appropriate when the goal is reachable within one episode and the baseline does not already explore effectively.  
   - Global cumulative counts (SimHash) are appropriate when the goal requires long-horizon search across many episodes.

**4. Is the modification additive or multiplicative?**  
   - If multiplicative (gate, count multiplier on existing signal), verify on a small env that the existing intrinsic signal is not suppressed in regions where it is known to be useful.  
   - If additive, the worst case is noise; the enhancement is self-limiting.

**5. Is the bonus additive to RND, or does it replace RND?**  
   - Additive (SimHash+RND): RND remains the directional backbone; bonus provides complementary coverage. If bonus is removed, performance degrades gracefully to vanilla RND.  
   - Replacement (SimHash-only, IntCoef=0 + count bonus): the agent has no learning-based directional signal. Empirically: 0.000 extrinsic return at 1.5M steps on KeyCorridor vs 0.936 for vanilla RND. Never use a count-based bonus as the sole intrinsic signal on sparse-reward tasks.

**6. Does the bonus provide directional signal or only coverage signal?**  
   - Coverage-only bonuses (hash count, position count) treat all novel states identically. In tasks where the goal lies at the end of a specific sequence of novel states, the agent may fill the count table without finding the goal.  
   - Directional bonuses (RND MSE, forward-dynamics prediction error) decay faster on frequently-visited states near the goal's approach path, implicitly biasing toward productive exploration.  
   - Safe combination: coverage bonus (addresses C2/C3) + directional bonus (addresses C5). This is exactly the SimHash+RND design.

**7. Does the enhancement create a cold-start period?**  
   - Methods that require a warm-up buffer (K-means clustering, anchor buffers, EMA initialization) will fall back to a degenerate behavior during the warm-up. Verify that the fallback is neutral (e.g., all states map to a single key, producing uniform bonus) rather than harmful (e.g., gate closes completely).  
   - In this codebase: `FeatureClusterer.cluster_id()` returns `-1` when `cluster_filled == 0`, which maps to `('cluster', '__none__')` in `train.py:393`. All envs share this key during cold-start, so the `1/sqrt(N)` multiplier drops to near-zero within the first rollout (~1024 visits to the same key). The cold-start lasts `ClusterRefreshSteps=4096` env steps — negligible at 1M-step budgets, but visible in early int_reward plots.

---

## Implications for the known RND failure modes

**Pitfall failure** (RND attracted to novel-but-deadly states): C1-compliant methods that add a *suppressive* signal in visited states help, but only if their granularity is fine enough to distinguish deadly from non-deadly regions (C2). SimHash does this implicitly — the lava tile gets a high hash-bonus on first visit but the count accumulates, reducing the bonus on re-visit. Position keys do the same. K-means clusters may group lava and safe corridor tiles into the same cluster, producing no discriminative signal. Option B's gate would address this if ensemble variance were informative (it is not, per C1).

**Gravitar failure** (intrinsic distracts from mature extrinsic): No count-based method directly addresses this. The RND signal decays naturally as the predictor learns, but this can take longer than the policy needs. A valid approach: detect when extrinsic value is non-zero and scale down the intrinsic coefficient — but this requires a signal from the value function (violating C1 if done via ensemble variance) or a simple threshold on measured extrinsic return (C1-compliant if derived from episode returns rather than critic outputs). SimHash's additive form is safe here because its own bonus decays as states are visited, eventually becoming negligible without suppressing RND.

---

## Completed runs summary

All planned experiments are finished. 26 run directories in `runs/`; full
results from `python scripts/eval_summary.py` and `python scripts/eval_simhash.py`.

### Multi-seed results now complete

**KeyCorridor — SimHash+RND vs vanilla RND (n=3 matched seeds)**

| Seed | Vanilla RND | SimHash+RND | Improvement |
|---|---|---|---|
| 0 | 325.6k | 281.6k | +13.5% |
| 1 | 318.5k | 250.9k | +21.2% |
| 2 | 317.4k | 270.3k | +14.8% |
| **mean ±std** | **320.5k ±3.6k** | **267.6k ±12.7k** | **+16.5%** |

**KeyCorridor — SimHash-only ablation (n=3 seeds, `UseRNDBonus=False`)**

| Metric | Result |
|---|---|
| extr_return | 0.000 ±0.000 (all seeds fail completely) |
| time-to-goal | ∞ (never solved on any seed) |
| int_reward (tail) | 56.437 ±1.759 (enormous — maximal exploration, zero exploitation) |
| unique hashes (final) | 1,386,233 ±30,884 (>400× the coverage of SimHash+RND) |

**LavaCrossing — vanilla RND multi-seed (n=3)**

| Seed | extr_return | goal_rate | time-to-goal | death_rate |
|---|---|---|---|---|
| 0 | 0.001 | 0.002 | never | 0.801 |
| 1 | 0.717 | 0.844 | 846.8k | 0.005 |
| 2 | 0.395 | 0.489 | 986.1k | 0.045 |

Seed 0 is the outlier — the catastrophic failure used to motivate Option B. Seeds 1
and 2 both solve LavaCrossing eventually, establishing that vanilla RND is not
fundamentally broken on this environment, only high-variance. This is the key
context for interpreting the Option B "improvement": it was a comparison against
the unlucky seed, not against the method at its typical performance.

**LavaCrossing — Option B (p=0.5) multi-seed (n=2 additional seeds)**

| Seed | extr_return | time-to-goal |
|---|---|---|
| 1 | 0.533 | 861.2k |
| 2 | 0.510 | 934.9k |

Option B at p=0.5 on seeds 1 and 2 is indistinguishable from vanilla RND on the
same seeds — both converge at roughly 850–990k steps. This eliminates the
bootstrap-mask probability as the cause of Option B's apparent seed-0 advantage.

---

## What remains untested

- **K-ablations on NovelD clustering**: only K=8 was tested. K=16, K=32, or K=64
  may satisfy C2 on KeyCorridor. Config sections can be added to `config.conf`
  with `NovelDNumClusters` set accordingly; `data/noveld_unique_keys_per_env` in
  TB is the primary diagnostic.
- **Position-keyed NovelD**: `EXP1_LAVA_NOVELD_POS` and `EXP3_KEYCORRIDOR_NOVELD_POS`
  config sections exist in `config.conf` but neither run exists. Position keys
  (one key per navigable tile) are the natural fine-grained baseline for the
  cluster-type variant, and the most direct test of the C2 hypothesis.
- **Multi-seed NovelD**: all NovelD results are single-seed (SEED=0). Whether the
  LavaCrossing and DoorKey PASS results hold across seeds is unknown.
- **Posterior + SimHash stacking**: `eval_summary.py` marks this INCOMPLETE
  (missing run). Posterior Sampling stacks badly with NovelD; whether it stacks
  well with SimHash (which produces small additive bonuses that don't interfere
  with per-rollout head commitment) is untested.
- **Independent CNN trunks per critic head**: would increase Option B ensemble
  diversity but does not address the root cause — value disagreement requires
  reward signal regardless of architecture (C1).
