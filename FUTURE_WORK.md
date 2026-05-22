# Future Work

This document catalogs open directions from the current state of the project:
RND + PPO baseline, with NovelD-clustered, SimHash+RND, and Posterior Sampling
as the evaluated enhancements. It is organized by the structural constraints
(C1–C5) in `reports/rnd_enhancement_constraints.md` that explain the pattern
of what worked and what did not.

---

## 1. Validate the granularity finding for NovelD (C2)

**What we know**: NovelD-clustered with K=8 passes on LavaCrossing and DoorKey
(small state spaces, 3–5 unique cluster keys visited per episode) but fails on
KeyCorridor (time2goal 360k vs baseline 325k). The hypothesis is K=8 is too
coarse for KeyCorridor's layout.

**What to test**: K-ablations — K=16, K=32, K=64 — on KeyCorridor with 3 seeds
each. Config sections `EXP3_KEYCORRIDOR_NOVELD` already exists; add
`EXP3_KEYCORRIDOR_NOVELD_K16`, `_K32`, `_K64` to `config.conf` with
`NovelDNumClusters` set accordingly. The `data/noveld_unique_keys_per_env` TB
metric is the primary diagnostic: if unique keys per episode rises to ≥10 at
higher K, and time-to-goal improves, the C2 hypothesis is confirmed.

**Position-keyed NovelD baseline**: `EXP1_LAVA_NOVELD_POS` and
`EXP3_KEYCORRIDOR_NOVELD_POS` config sections exist but were never run. These
are the natural comparison — fine-grained (one key per navigable tile) vs
clustered. Expected: position keys solve C2 on both envs; cluster keys are only
competitive when K ≈ number of semantic regions.

---

## 2. Multi-seed validation for NovelD (C3)

**What we know**: All NovelD results are single-seed (SEED=0). SimHash+RND and
vanilla RND on KeyCorridor have 3 seeds each, confirming the 16.5% improvement
is consistent (+13.5%, +21.2%, +14.8%). NovelD's PASS on LavaCrossing (563k,
seed 0) and DoorKey (97.3k, seed 0) is a single data point each.

**What to test**: Run `EXP1_LAVA_NOVELD` and `EXP2_DOORKEY_NOVELD` at seeds 1
and 2. Adds 4 runs. If NovelD's LavaCrossing improvement holds across seeds, it
has the same evidential weight as the SimHash result. If it regresses, it raises
the same multi-seed concern that weakened the Option B headline claim.

---

## 3. A C1-compliant gate: intrinsic-ensemble disagreement

**What we know**: Option B's variance gate failed because ensemble disagreement
on *extrinsic* critics requires reward before heads can diverge (C1 violation).
The gate idea itself — suppress intrinsic in familiar states, let it fire in
novel ones — is sound.

**What to test**: Replace the K=5 extrinsic critic ensemble with K=5 parallel
RND *predictor* networks (each trained on the same observations with different
random seeds). Their prediction variance is non-trivial from step 0 (different
initializations → different prediction errors). Gate the RND intrinsic reward
by the variance across these K predictor outputs instead of across critic values.

This is structurally equivalent to Pathak et al. 2019's Disagreement Curiosity
applied as a *regulator* rather than a *generator* — the novel composition that
Option B was trying to achieve, but using a C1-compliant signal.

**Implementation cost**: ~100 lines in `agents.py` and `model.py`. Reuse the
existing gate formula in `agents.py:gate_factor()` with a different variance
source.

---

## 4. Stacking SimHash + NovelD (C3 + C2 interaction)

**What we know**: Posterior + NovelD stacking was catastrophic (580k vs 360k).
But that failure is attributable to Posterior Sampling's per-rollout head
commitment amplifying NovelD's noise. SimHash + NovelD haven't been stacked.

**What to test**: `EXP3_KEYCORRIDOR_SIMHASH` + `UseNovelD=True`. SimHash is
additive (C4-safe) and never saturates (C3-safe). NovelD's episodic component
provides within-episode freshness that SimHash's global count doesn't capture.
If C2 is satisfied (use K=32 or position keys to avoid coarse-graining), the
two methods address different failure modes and should be complementary. If it
degrades, the episodic/global count interaction has an interference mechanism
worth understanding.

---

## 5. Adaptive granularity (resolves C2 without K-tuning)

**What we know**: Fixed-K clustering requires K to be calibrated per environment.
SimHash avoids this with near-infinite resolution. A method that starts fine and
coarsens only when needed would get the best of both.

**What to test**: Online K-means that grows K over training (start K=4, double
every `GrowthSteps` whenever the average cluster size exceeds a threshold).
Alternatively: hierarchical counting — use both position keys (fine) and cluster
keys (coarse) and combine their `1/sqrt(N)` multipliers additively. The
hierarchical form requires no new training infrastructure; it's a one-line
change to the key construction in `train.py:393`.

---

## 6. Longer-horizon directions

- **Atari validation**: All results are on MiniGrid. The C1–C5 constraints are
  derived from MiniGrid behavior; whether they hold on Montezuma's Revenge or
  Pitfall (the original RND paper failures) is unknown. SimHash+RND on
  Montezuma would be the most direct test of the C2/C3 claims at scale.

- **Intrinsic-ensemble disagreement as the *source* of intrinsic reward**
  (Disagreement Curiosity, Pathak et al. 2019): rather than gating RND with
  predictor variance, replace RND MSE with predictor-ensemble variance
  entirely. This satisfies C1 and C5 (predictor disagreement is higher near
  novel states) and provides a natural decay as predictors converge. The
  trade-off is losing RND's directional bias from normalized MSE.

- **Global + episodic count combination**: A state's intrinsic bonus could use
  `λ_global/sqrt(N_global(h)) + λ_episodic/sqrt(N_ep(h))` — global for
  long-horizon pressure, episodic for within-episode freshness. This addresses
  C3 without choosing between the two decay regimes.
