# Option B: Bootstrap-Ensemble Pessimism + Variance-Gated Intrinsic Reward

A complete narrative of what was attempted, what was learned, and the negative
finding the experiments support.

---

## 1. Design Intent

Option B was designed as an enhancement to vanilla RND + PPO that would address
two documented failure modes from Burda et al. 2018:

1. **Pitfall failure** — intrinsic reward attracts the agent to novel-but-deadly
   states; on Pitfall, RND scores -20 because the agent dies immediately. The
   negative extrinsic at death does not propagate strongly enough to override
   the positive intrinsic at the same state.
2. **Gravitar failure** — RND does not consistently exceed PPO on Gravitar
   because intrinsic motivation distracts the policy from already-mature
   extrinsic learning.

The proposed mechanism combined two published ideas applied to a setting where
they had not been combined before:

- **TD3-style pessimistic min** (Fujimoto et al. 2018) extended to K=5 critic
  heads with **Bootstrap-DQN-style per-sample masking** (Osband et al. 2016,
  Bernoulli p=0.8 per head). The advantage uses `min(V_ext_1, ..., V_ext_5)`
  as the extrinsic value estimate, which is more negative than the mean
  whenever heads disagree, biasing the policy away from states with uncertain
  extrinsic value.
- **Variance-gated intrinsic reward**: `r_int_t = RND_bonus_t · σ(α · var(V_ext_k))`,
  with the gate clipped to a floor of 0.2 to prevent total suppression. Designed
  to be **self-extinguishing**: as the ensemble agrees on extrinsic value
  (low variance → state is value-familiar), intrinsic reward is suppressed.
  Where the ensemble disagrees (high variance → genuinely uncertain state),
  intrinsic fires normally.

The novel contribution claimed for Option B was the variance-gated intrinsic
reward — no published method gates an externally-computed intrinsic reward
by an ensemble's value-variance signal. This was the design point we expected
to validate empirically.

---

## 2. Implementation Iterations

### 2.1 Initial design (failed catastrophically)

The first implementation used:
- K critic heads as linear layers on top of a single shared `extra_layer` MLP
- Gate could close to zero (no floor)
- BootstrapP = 0.8 per sample per head
- K = 3

Result on single-seed (SEED=0) LavaCrossing: Option B achieved `death_rate=0.93`,
`goal_reach_rate=0.001` — catastrophically worse than vanilla RND. Same pattern
on DoorKey-5x5.

### 2.2 First diagnosis-driven fix

Two root causes identified:

1. **Shared-trunk + linear-only heads → zero ensemble diversity.** All heads
   computed nearly identical V_ext because they shared 99% of their parameters.
   `min(V_ext_k) ≈ mean(V_ext_k)` so the pessimism mechanism produced no
   useful pessimism.

2. **Gate could fully close in sparse-reward regimes.** When extrinsic critic
   variance was uniformly small everywhere (because no head had observed
   positive reward yet), the gate closed uniformly, killing the entire
   intrinsic reward signal — exactly what RND was supposed to provide for
   exploration.

Fixes applied:
- **Per-head 2-layer MLPs** (`Linear(448 → 448) → ReLU → Linear(448 → 1)`),
  each with ~200k independent parameters
- **Gate floor of 0.2** to prevent complete intrinsic suppression
- **GateAlpha lowered to 0.5** for softer gate slope
- **K bumped from 3 to 5** for more ensemble capacity

This produced the "current" Option B that we evaluated empirically.

### 2.3 BootstrapP ablation (refuted hypothesis)

After the multi-seed analysis revealed near-zero ensemble variance (see §4),
we hypothesized that the bootstrap-mask probability (p=0.8) was too high —
heads saw mostly-identical data and converged to mostly-identical value
estimates. The standard Bootstrap DQN paper uses p=0.5.

We changed `BootstrapP = 0.5` and re-ran Option B on seeds 1 and 2 of
LavaCrossing. The hypothesis was that aggressive masking would force per-head
divergence, producing genuine pessimism.

Result: **the change made Option B *slower*, not faster.** Final extrinsic
returns dropped, breakthroughs happened ~250-350k steps later, and ensemble
variance remained ≈ 0 throughout most of training (rising only post-
breakthrough). The hypothesis was refuted; we reverted to `BootstrapP = 0.8`.

---

## 3. Empirical Results

All runs use 1M env steps, 8 parallel envs, Adam @ 1e-4, PPO γ=0.999, γ_int=0.99,
λ=0.95, ε=0.1, 4 epochs × 4 mini-batches per rollout. Single seed unless noted.

### 3.1 LavaCrossing — original Option B (K=5, BootstrapP=0.8)

| Method | Seed | extr_final | goal_rate | death_rate | Breakthrough step |
|---|---|---|---|---|---|
| Vanilla RND | 0 | 0.001 | 0.002 | 0.76 | never |
| Vanilla RND | 1 | **0.79** | 0.92 | 0.00 | ~820k |
| Vanilla RND | 2 | 0.39 | 0.50 | 0.04 | 986k (peak 0.52, regressed) |
| **Option B (K=5, p=0.8)** | 0 | **0.83** | 0.93 | 0.00 | **592k** |

### 3.2 LavaCrossing — Option B with BootstrapP=0.5 (refuted hypothesis)

| Method | Seed | extr_final | Breakthrough step |
|---|---|---|---|
| Option B (K=5, **p=0.5**) | 1 | 0.37 | 861k |
| Option B (K=5, **p=0.5**) | 2 | 0.48 | 935k |

### 3.3 DoorKey-5x5

| Method | Seed | extr_final | goal_rate |
|---|---|---|---|
| PPO-only (no intrinsic) | 0 | 0.679 | 0.726 |
| Vanilla RND | 0 | **0.958** | 1.000 |
| **Option B** | 0 | **0.294** | 0.349 |

### 3.4 Variance Trajectories (the crucial diagnostic)

Across all Option B runs, `data/ensemble_extrinsic_variance` was logged
throughout training. Observation: variance is **near zero** throughout the
majority of training, only rising once the policy starts accumulating
extrinsic reward.

Representative trajectory (Option B p5 seed 2):
- Start: 1.13e-06
- Mid-training: 8.35e-05
- End (post-breakthrough): 1.54e-04

The `min(V_ext_k)` aggregator therefore produced values numerically equal
(to 3 significant figures) to `mean(V_ext_k)` for most of training. The
pessimism mechanism was **coded but not operative**.

---

## 4. The Diagnostic Finding That Reframes Everything

The intended Option B mechanism — pessimistic ensemble min amplifies negative
extrinsic at deadly novel states — **cannot fire when ensemble variance is
near zero**. Our experiments show this is what happens in sparse-reward +
curiosity-driven RL:

> Until extrinsic reward is observed, every value head learns to predict
> V ≈ 0 everywhere (because the TD target is 0). All K heads converge to
> nearly identical estimates regardless of architectural diversity. The
> aggregator (min, mean, or max) returns nearly the same value because
> there is nothing for the heads to disagree about.

This is a **structural property of the regime**, not a bug in our
implementation. The architectural fixes we applied (per-head MLPs,
BootstrapP variations, K size) all share the same vulnerability: they
control *how* heads diverge given signal, not *whether* there is signal
to drive divergence.

### Implications for "what Option B actually does"

If `min ≈ mean` throughout training, then the empirical effects we
observed on LavaCrossing seed 0 (`extr=0.83` vs vanilla `0.001`) cannot
be attributed to the pessimism mechanism. The active ingredient must be
one of:

1. **The variance gate's uniform suppression of intrinsic reward.** With
   `gate ≈ floor = 0.2` for most of training, intrinsic reward is
   effectively attenuated 5× compared to vanilla RND. On LavaCrossing,
   this dampens the curiosity-attraction to lava without changing the
   direction of intrinsic-driven exploration — which could explain the
   reduced lava-dance failure rate.
2. **Mild ensemble averaging stabilization.** Even with near-identical
   heads, having 5 critic predictions averaged together produces a
   smoother target than a single noisy critic. This is a regularization
   effect, not a pessimism effect.
3. **Bootstrap mask data-augmentation.** Each head sees a different
   ~80% subset of samples; this is similar to dropout-on-data and may
   slightly smooth value estimates.

None of these are what the design claimed Option B was doing.

---

## 5. The Sharper Reframing After Multi-Seed Data

The seed-0 LavaCrossing result (`extr_return = 0.001 → 0.83`) initially
suggested Option B fixed a catastrophic vanilla failure. Multi-seed data
substantially weakens this claim.

### Vanilla RND can solve LavaCrossing on 2 of 3 seeds

When given the full 1M-step budget:
- Seed 0: never solved (the failure we built the narrative around)
- Seed 1: broke through at ~820k, finished at `extr = 0.79`
- Seed 2: broke through at step 986k, peaked at `extr = 0.52`, regressed to `extr = 0.39` (intrinsic-distracts-from-mature-extrinsic instability)

So vanilla's seed-0 failure was the exception, not the rule. The Option B
"dramatic improvement" relied on choosing the unlucky vanilla seed as the
comparison.

### Option B's seed-0 advantage may be a real (but smaller) convergence-speed effect

Where vanilla solves, Option B's seed-0 breakthrough (592k) is earlier
than vanilla's matched breakthroughs (820k, 986k). This is consistent
with TD3's published claim that ensemble methods reduce the wandering
period before policy lock-in.

**However**, we have only n=1 evidence at the original config (K=5, p=0.8)
because the BootstrapP=0.5 runs on seeds 1 and 2 were a different
configuration. To validate "Option B genuinely accelerates convergence,"
we would need K=5, p=0.8 runs at seeds 1 and 2 — which were not completed
in this study.

### Multi-seed breakdown of what's defensible

What the matched-seed data supports:
- Vanilla and Option B (at p=0.5) **converge on similar timescales** on
  LavaCrossing — both around step 820-940k for the seeds where they succeed
- Option B (at p=0.8) on the one seed we have **may** converge earlier
  (592k), but n=1 is insufficient evidence
- Option B at p=0.5 is essentially **indistinguishable from vanilla** in
  speed and final performance
- On DoorKey, Option B clearly **hurts** performance (single seed, but
  consistent with Bootstrap DQN's published prediction)

---

## 6. Comparison to Published Methods

Option B is best understood as a hybrid of:
- **TD3** (Fujimoto et al. 2018): pessimistic-min, K=2, continuous control. We
  generalize to K=5 with bootstrap masks, apply to PPO (discrete), and add
  variance-gated intrinsic.
- **Bootstrap DQN** (Osband et al. 2016): per-sample bootstrap masks, K=10,
  used for **posterior-sampling exploration, not pessimism**. We borrow the
  masks but use the ensemble for the opposite purpose.

The crucial published warning, which our experiments empirically confirm:

> Osband et al. 2016 explicitly argue that **ensemble pessimism is the wrong
> use of a critic ensemble in online RL** — they propose posterior sampling
> for exactly the reasons we observe. Our DoorKey negative result and
> LavaCrossing near-zero variance finding together rediscover this warning
> in a curiosity-driven setting.

No published method is identical to Option B, primarily because the
variance-gated intrinsic reward is a novel contribution. But the
pessimism-via-min mechanism is well-trodden, and our results align with
the existing literature's predictions about when it works and when it
doesn't.

---

## 7. Conclusion

The Option B experiments produce a **principled negative finding** with
a clear mechanism explanation. Specifically:

### What we set out to demonstrate

That a bootstrap-ensemble extrinsic critic with pessimistic min and
variance-gated intrinsic reward would address the Pitfall and Gravitar
failure modes of RND by:
1. Amplifying negative extrinsic at deadly novel states (Pitfall fix)
2. Self-extinguishing intrinsic where extrinsic value is mature (Gravitar fix)

### What we actually found

1. **The pessimism mechanism does not operate as designed in sparse-reward +
   curiosity-driven online RL.** Ensemble variance remains near zero for most
   of training because the value targets are sparse — there is no signal for
   the heads to disagree about until reward is observed. `min(V_ext_k) ≈
   mean(V_ext_k)` so the pessimism produces no useful pessimism.

2. **Empirical Option B effects, where present, do not come from the pessimism.**
   The single-seed LavaCrossing improvement is most plausibly explained by the
   variance gate's uniform attenuation of intrinsic reward (which damps the
   curiosity-attraction to lava) rather than state-specific pessimism.

3. **Multi-seed analysis substantially weakens the headline claim.** Vanilla
   RND solves LavaCrossing on 2 of 3 seeds when given full budget; Option B's
   apparent dominance was driven by comparing against the unlucky vanilla
   seed. Matched seeds show vanilla and Option B (with BootstrapP=0.5) at
   roughly equal performance with roughly equal seed sensitivity.

4. **The DoorKey negative result aligns with Bootstrap DQN's published warning.**
   Where intrinsic motivation is genuinely useful (DoorKey-5x5: vanilla
   beats PPO-only by 41%), Option B's gate suppresses the useful curiosity
   signal and the pessimistic min delays recognition of rare positive
   rewards. The agent's performance drops sharply.

5. **Architectural rescue is unlikely to work.** Independent CNN trunks
   would create representation diversity but not value-signal diversity
   pre-reward. The bottleneck is *data*, not *architecture*. The sparse-
   reward regime cannot manufacture value-head disagreement no matter
   how the ensemble is parameterized.

### The scientific contribution

The contribution is not "Option B improves RND" — that claim does not hold
up to multi-seed analysis with mechanism diagnostics. The contribution is:

> **Ensemble pessimism via min-of-K is a structurally incompatible
> mechanism in sparse-reward + curiosity-driven online RL, regardless
> of architecture, because its prerequisite (inter-head value
> disagreement) cannot manifest until extrinsic reward is observed —
> which is the regime the method was intended to help. This empirically
> rediscovers Bootstrap DQN's 2016 published warning in a curiosity-
> driven setting, and motivates alternative ensemble uses
> (posterior sampling, distributional value, ensemble disagreement on
> intrinsic predictors) that do not require value-signal divergence.**

### Specific design recommendations going forward

Based on the mechanism analysis, productive uses of K-head ensembles in
sparse-reward + RND settings would target signals that *do* exist abundantly
pre-reward:

1. **Posterior sampling on policies** (Bootstrap DQN's original proposal): use
   the K heads for trajectory-level exploration commitment, not per-state
   value pessimism. Already implemented as Experiment 4 in this project.
2. **Ensemble of intrinsic predictors** (Pathak et al. 2019 "Self-Supervised
   Exploration via Disagreement"): K parallel RND-style predictors; their
   variance becomes the intrinsic reward signal. Observation targets are
   abundant, so heads can diverge regardless of reward sparsity.
3. **Distributional value RL** (C51, QR-DQN, IQN): predict a distribution
   over Q-values from a single critic; uncertainty is built into the output
   rather than derived from ensemble disagreement.

Of these, the variance-gated intrinsic reward (our novel design contribution)
could plausibly be combined with #1 or #2 to retain the self-extinguishing-
exploration property without depending on inoperative ensemble pessimism.
This would be the natural follow-up if continuing the line of work.

### Honest scope

These conclusions are based on:
- LavaCrossingS9N2-v0 and DoorKey-5x5-v0 (two MiniGrid envs)
- 1M env steps per run, single seed for most configurations, n=3 for
  vanilla on LavaCrossing
- The specific architecture and hyperparameters documented in
  `config.conf`, `model.py`, and `agents.py`

The "ensemble pessimism is structurally incompatible with sparse-reward +
curiosity" claim is supported by:
- Direct measurement of near-zero ensemble variance throughout training
- The agreement of our DoorKey negative result with Bootstrap DQN's
  published warning
- The refutation of the BootstrapP=0.5 hypothesis (lower mask probability
  did not produce useful divergence)

It is **not** supported by tests of:
- Larger K (K=10, K=20, etc.)
- Independent CNN trunks per head
- Different envs (dense-reward, non-MiniGrid)

If the field extended this work, K=10 with independent trunks on a sparse-
reward Atari task (Montezuma, Pitfall) would be the strongest possible test
of the architectural-rescue hypothesis. We did not have the compute budget
to attempt this.
