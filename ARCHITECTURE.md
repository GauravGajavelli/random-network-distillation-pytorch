# Architecture: Augmenting RND with Option B + DSC

> **Historical document.** This file covers the Option B + DSC design, which
> was the plan for Experiments 1–3 before multi-seed analysis revealed
> Option B's mechanism was inoperative in sparse-reward settings and DSC was
> subsumed by simpler methods. The **final shipped enhancements** are
> **NovelD-with-cluster-types**, **SimHash+RND**, and **Posterior Sampling**,
> documented in § 6 at the bottom and in `reports/rnd_enhancement_constraints.md`.
>
> Sections 1–5 remain accurate as a record of what was built and why some
> choices were made; they are part of the "challenges" and "design evolution"
> narrative for the presentation. Do not read them as the current architecture.

This document describes the architectural changes made on top of the original
RND implementation and the specific failure mode each one addresses. It assumes
familiarity with the RND paper (Burda et al., 2018) and the failure-mode
catalogue documented in the plan file.

The two interventions covered in §§ 1–3, both of which **augment** RND (the
frozen target and trained predictor are unchanged), are:

- **Option B**: bootstrap-ensemble extrinsic critics + variance-gated intrinsic
  reward. Evaluated as the primary enhancement; empirically found to be
  inoperative in the sparse-reward + curiosity regime (see § 6 and
  `reports/option_b/option_b_writeup.md`).
- **DSC** (Discriminative Subgoal Curiosity): a distance-based, anchor-ranked
  bonus. Implemented and tested; ultimately subsumed by NovelD+SimHash on the
  measured metrics.

---

## 1. Bootstrap-ensemble extrinsic critics with per-head MLPs

### What it is

The original `critic_ext` single linear head is replaced by `K=5` independent
2-layer MLPs (`448 → 448 → 1`), all sitting on the shared CNN trunk. They
train with per-sample bootstrap masks (Bernoulli p=0.8 per head per sample).
At policy-update time, the extrinsic advantage uses the **pessimistic value**
`min(V_ext_k)` over the K heads; each head trains against its own TD target.

```
                                          ┌── head_1 (448→448→1) ──┐
   CNN trunk → 448-d features ─────────── ┤    ⋮                  ├── min → V_ext_pess
                                          └── head_K (448→448→1) ──┘
```

### What problem it solves

**Pitfall-style "novel-but-deadly" attractor (#9 partial, user's #3).** Vanilla
RND has no mechanism for asymmetric pessimism on bad outcomes. Intrinsic
curiosity pulls toward novelty, and a small extrinsic negative gets averaged
in symmetrically against the intrinsic positive in the advantage. On
LavaCrossing, this manifests as the agent walking into lava ~80% of episodes
because each lava-adjacent step gives a positive intrinsic bonus that
outweighs the negative extrinsic signal.

### Mechanism by which it solves it

When the ensemble heads diverge in their value estimates at a state, the `min`
across them is significantly lower than the mean. On states near a death tile
(where extrinsic value is genuinely uncertain because the agent dies if it
steps wrong), variance is high → `min(V_ext_k)` is strongly negative → the
advantage at the *parent* state of that lava-adjacent action becomes
negative enough to overcome the intrinsic positive. The policy learns to
avoid that action despite its curiosity-bait nature.

### Why per-head MLPs and not the simpler design

The first iteration shared the `extra_layer` MLP across all heads and only
varied the final linear layer. With ~99% of the parameters shared, the
heads' outputs were near-identical — variance across heads was small enough
that `min(V_ext_k) ≈ mean(V_ext_k)`, and the pessimism effect disappeared.
The current per-head design gives each critic ~200k independent parameters,
so different bootstrap-mask histories produce genuinely different head
outputs. Variance becomes meaningful, and `min` becomes meaningfully more
pessimistic than `mean`.

### Why K=5 specifically

K trades off compute against variance-estimate quality:
- K=3 (initial choice) gives noisy variance estimates from too few samples.
- K=10+ (textbook Bootstrap DQN) is more accurate but adds ~6× the per-head
  parameter count over the K=3 baseline.
- K=5 is a compromise: reasonable variance estimate, modest compute hit
  (~15% slower training in smoke tests).

### What this does NOT do

- **No risk-aware planning.** It's a per-state value pessimism, not a
  trajectory-level risk measure. Multi-step coordinated risk (Pitfall's
  scorpion timing, Gravitar's gravity dynamics) needs more than min over
  ensemble estimates.
- **No principled epistemic uncertainty.** Ensemble variance is a proxy for
  uncertainty, not a calibrated one. The heads are correlated through the
  shared trunk, so the variance underestimates true epistemic uncertainty.
  Distributional RL (C51, IQN) or Bayesian critics would be more principled
  but substantially more complex.
- **No protection against truly unobservable risk.** If a state's danger
  isn't reflected in the extrinsic reward signal during training, no amount
  of ensemble pessimism will catch it. The mechanism amplifies signals that
  exist, not signals that don't.

### Files
- `model.py`: `CnnActorCriticNetwork.critic_ext_heads` (`nn.ModuleList` of K MLPs).
- `agents.py`: `RNDAgent.train_model` (per-head TD loss with Bernoulli masks).
- `train.py`: per-head TD targets + pessimistic-min advantage.

---

## 2. Variance-gated intrinsic reward

### What it is

A gate factor multiplies the raw RND bonus before it enters the advantage
stream:

```
gate = clip(α · var / (var + EMA(var)), 0, 1)
r_int_t = RND_bonus_t · gate
```

`var` is the variance of the K extrinsic critic outputs at the current state.
`EMA(var)` is a running mean of that variance (decay 0.99) maintained as
`self._var_ema` on the agent. The gate is **scale-invariant** because the
denominator self-normalizes against whatever variance regime training is in.

Behaviour:
- `var = 0` (heads perfectly agree, state's value is fully known): gate → 0,
  intrinsic suppressed.
- `var = EMA(var)` (typical state): gate ≈ 0.5α.
- `var ≫ EMA(var)` (heads strongly disagree, genuinely uncertain state):
  gate → 1, intrinsic fires normally.

### What problem it solves

**Gravitar-style "intrinsic-as-distractor" (#9 partial).** RND keeps producing
non-trivial intrinsic reward even in regions where the extrinsic critic has
already converged. The policy gets confused: the extrinsic signal says
"this is the right behaviour, keep doing it," while the intrinsic signal
says "go look for something new." On dense-extrinsic tasks where vanilla
RND underperforms PPO, this is the canonical mechanism — the curiosity
budget gets spent fighting already-good extrinsic learning rather than
augmenting it.

### Mechanism by which it solves it

Ensemble convergence is the signal. As the K extrinsic critics train on
mature regions of the state space, their outputs converge, variance drops,
and the gate closes — intrinsic reward gets multiplied by something small,
effectively suppressing it. Conversely, when the agent reaches a genuinely
new region of state space (heads disagree because they've seen different
bootstrap samples and haven't converged), variance is high, gate opens, and
curiosity drives exploration again. The whole mechanism is **self-extinguishing
without a manual schedule** — no annealing of `IntCoef`, no human-set
"exploration phase length."

### Why this formula, not sigmoid

The first iteration used `gate = sigmoid(α · var)`. Because `sigmoid(0) = 0.5`,
the gate had a floor of 0.5 — it could only attenuate intrinsic reward by 2×
at most, even when ensemble agreement was perfect. That's not enough
suppression to override mature extrinsic learning. The current formula
`var / (var + EMA(var))` can reach 0 when variance is small relative to its
running mean, which is exactly what's needed for the Gravitar fix.

Smoke tests of the current formula show the gate dropping from ~0.55 to
~0.10 over 25k env steps as the ensemble converges. That's the
order-of-magnitude suppression the original design needed but couldn't
achieve with sigmoid.

### What this does NOT do

- **The "uncertainty" signal is correlation-contaminated.** Ensemble heads
  share the CNN trunk, so their disagreement reflects only the variance
  of the per-head MLPs, not the variance of the underlying feature
  representation. True epistemic uncertainty (e.g., from a fully Bayesian
  posterior or independent networks) would be larger. We get a *useful*
  signal, not a calibrated one.
- **Variance and novelty are not the same thing.** A state can have high
  ensemble variance because heads haven't seen many bootstrap samples there
  (true novelty) OR because the value function is genuinely multi-modal
  (e.g., stochastic dynamics). The gate opens for both, which is fine —
  but it means the gate isn't strictly an "is this novel?" detector.
- **The gate is computed at the current state, not the next state.** The
  bonus is technically a reward for *transitioning into* the next state,
  so the variance at the *next* state would be more theoretically correct.
  In practice the one-step lag is tiny relative to other approximations,
  but it's a known imprecision.
- **No fine-grained control.** `α` (gate_alpha) is the only tuning knob.
  Sub-region calibration (e.g., "be more aggressive in low-reward regions")
  isn't possible without a more complex design.

### Files
- `agents.py`: `RNDAgent.gate_factor`.
- `train.py`: gate computed per rollout step and passed into
  `compute_intrinsic_reward`.

---

## 3. DSC: distance-based subgoal curiosity with anchor ranking

### What it is

A reservoir-sampled `AnchorBuffer` of up to 64 anchor feature vectors
(see [§ What are anchors](#what-are-anchors) below). At every step, the DSC
bonus multiplies the already-gated RND signal:

```
nearest_dist, rank_weight = anchor_buffer.nearest(target_features)
norm_dist = nearest_dist / EMA(nearest_dist)
r_int_t = RND_bonus_gated_t · (1 + λ · norm_dist · rank_weight)
```

where:
- `nearest_dist` is the L2 distance from the current state's RND target
  features to the nearest anchor in the buffer.
- `rank_weight` = `((age_normalized + 0.1) ** β)` with `age_normalized = 1`
  for the most recently discovered anchor and `0` for the oldest. Newer
  anchors get larger weight.
- `λ` (default 0.5) controls maximum multiplicative bonus magnitude.

Anchors are inserted when a state's intrinsic reward (pre-DSC, post-gate)
exceeds the running 90th percentile of recent intrinsic rewards. Reservoir
sampling keeps older anchors fairly represented.

### What problem it solves

**Short-term bias (#1), local-only exploration (#4), predictor saturation
(#12).** Vanilla RND's signal collapses to whatever the predictor has
memorized, gives no preference among novel states, and saturates globally
over long runs. DSC provides:

- A *distance* term: high when the current state is far from any anchor →
  rewards genuine feature-space novelty, not just "predictor disagrees."
- A *rank weight* term: high when the nearest anchor was discovered
  recently → pushes the agent toward the frontier instead of loitering
  near old discoveries.
- *Independence from the RND predictor*: the distance bonus doesn't decay
  as the predictor learns. It only shrinks when the agent *actually*
  visits anchor regions (pushing them into the buffer reduces
  nearest-distance for similar future states), which is a real novelty
  signal, not a self-defeating one.

### Mechanism by which it solves it

The DSC bonus multiplies the gated RND signal, so its contribution rides on
top of whatever RND/Option B is already producing. Two distinct levers
operate at every step:

1. **Distance lever.** When the agent ventures into a region where no anchor
   has been placed, `nearest_dist` is large, `norm_dist` is large (relative
   to the running mean of distances), and the multiplier is large. The
   policy learns that this region was rewarding.
2. **Rank lever.** Among multiple anchors the agent might be near, the one
   discovered most recently has the largest rank weight. So if anchors 1
   (early) and 4 (latest) are both nearby, the agent gets more bonus for
   states near anchor 4 than states near anchor 1.

Together, these biases produce an emergent *shifting frontier*: as new
anchors get added, older ones become rank-inert, and the bonus
center-of-mass moves forward. The agent's exploration follows.

### Why distance, not discriminator

The first iteration trained a small discriminator network `q_θ(z|s)` to
predict the nearest-anchor ID and used `−log q(z*|s)` as the bonus. As the
discriminator trains, it gets better at predicting `z*`, so `−log q(z*|s)`
shrinks to zero — regardless of whether the state is actually novel. The
bonus *self-defeats through training*. The current distance-based bonus
doesn't have this pathology because the L2 distance between feature vectors
isn't something the network can "learn away."

### What DSC does NOT do — the sequencing/typing limits

This is the part most likely to be over-claimed elsewhere in the project,
so it deserves explicit clarity.

#### DSC does not learn types of subgoals

A "type" of subgoal would be a category that generalizes across related
states — e.g., "the type *key pickup* applies whether the key is red, blue,
in room 1, or in room 2." Type-learning would require some aggregation
mechanism: clustering anchors, training a categorical discriminator,
hierarchical anchor structure, successor features, or DIAYN-style skill
discrimination.

DSC has none of these. Each anchor is a single point in 512-d feature
space. The buffer holds 64 independent points; the nearest-anchor lookup
returns one-of-64 and never aggregates. Whatever "types" exist are
emergent from the embedding geometry of the frozen RND target network —
behaviourally similar states tend to land in nearby feature regions, so
anchors capturing similar events end up clustered. But DSC doesn't exploit
this clustering. The nearest-anchor lookup collapses any latent type
structure into a one-of-64 label that's no more informative than the
underlying point.

For Experiment 3 (KeyCorridor), the task has roughly 3-4 categorical
subgoals (find key, pick up key, find door, unlock door, find ball).
DSC will likely create *multiple anchors per subgoal* — maybe 3-5
anchors for "states near the key" alone, depending on the luck of
reservoir sampling. The rank weight treats these as independent
landmarks. The agent gets denser reward (good), but it doesn't learn
an abstract "pickup phase → unlock phase → goal phase" structure. It
learns to be near *whichever specific 512-d feature vectors happened
to be sampled into the buffer*.

#### DSC does not learn explicit sequences

A real sequence-learning method (Go-Explore, hierarchical RL with options,
goal-conditioned policies with prerequisites) would have explicit
machinery like:

- "Anchor B is reachable from states near anchor A — encode this dependency."
- "If you've visited anchor 1 this episode, then visiting anchor 2 is
  more valuable."
- A discrete plan or option-graph over anchors.

DSC has none of that either. The "sequence" that emerges is the
combination of three forces:

1. **Prerequisites are physically enforced by the environment.** To reach
   the "after-pickup" anchor, the agent must first pick up the key — not
   because DSC requires it, but because the environment transitions do.
   So if DSC successfully rewards being-near the after-pickup anchor, the
   policy implicitly learns the prerequisite.
2. **Credit assignment is easier with denser reward.** PPO's advantage
   estimate is much better-behaved when reward isn't just `+1` at the end
   of a 270-step episode. Each anchor visit provides a localized reward
   signal, so the policy can credit the action that *got it near anchor 3*
   even before it figures out how to chain to anchor 4.
3. **The frontier moves.** Once the agent reliably reaches an anchor
   region, more candidate states get into the buffer near and past it.
   Newer anchors become the bonus attractor, and the implicit curriculum
   advances.

So DSC turns sparse-reward exploration into **shaped point-landmark
reward with a moving frontier**. It does not model, enforce, or reason
about sequences. Calling that "sequence learning" would be overclaiming.

#### Where this fails

- **Re-collecting bonuses near a recent anchor without progressing.**
  Nothing in DSC stops the agent from hovering near a recently-discovered
  anchor and re-collecting bonuses there. Rank weight only decays when
  *newer* anchors get added. If discovery stalls, the bonus gradient
  flattens and there's nothing pushing past the plateau. An episodic
  visit-counter would fix this — we don't have one.
- **Non-physical prerequisites.** If anchor B is physically reachable
  without doing anchor A first but only *valuable* after A, DSC has
  nothing to enforce ordering. The user's "learn an order/sequence that
  must happen" framing implies explicit prerequisite reasoning, which
  DSC doesn't provide.

### What it would take to actually learn types or sequences

The plumbing exists for several principled upgrades, ordered by complexity:

| Upgrade | Adds | Cost |
|---|---|---|
| K-means clustering over anchors | Periodic clustering into K=8 cluster centers; nearest-cluster lookup; cluster IDs become discrete "types" | **implemented — see § 3a** |
| Episodic anchor counter | Reward only the *first* visit to each anchor per episode → forces breadth | ~30 lines |
| Conditional anchor bonus | Bonus for anchor N gated on visit-to-anchor-N-1 this episode | ~80 lines, needs anchor graph |
| Hierarchical anchors | Two-level buffer: coarse "type" anchors + fine "instance" anchors | ~200 lines |
| Successor anchor graph | Track typical anchor-to-anchor transitions; reward observed-sequence completions | substantial new module |
| Goal-conditioned hierarchical RL | Each anchor is a sub-goal; train sub-policies to reach each anchor | major rewrite |

The first row (K-means clustering) **is now implemented** as the type-learning
extension for Experiment 3 — see § 3a below. The episodic counter is the next
cheapest meaningful upgrade if even the clustered version plateaus.

### Files
- `utils.py`: `AnchorBuffer` (reservoir sampling, discovery timestamps,
  brute-force nearest-neighbour, EMA of nearest distances).
- `train.py`: per-step bonus computation and anchor insertion.

---

## 3a. DSC type-learning extension: K-means clustering over anchors

### What it is

When `UseClusters=True`, the `AnchorBuffer` periodically runs K-means
(default K=8) over its 64 stored anchors. The K cluster centers form a
coarser representation: each cluster aggregates anchors that landed in
similar regions of the 512-d feature space. At every step, the DSC bonus
uses **nearest-cluster** lookup instead of nearest-anchor:

```
cluster_idx, dist_to_center, rank_weight = anchor_buffer.nearest_cluster(features)
norm_dist = dist_to_center / EMA(dist)
r_int_t = RND_bonus_gated_t · (1 + λ · norm_dist · rank_weight)
```

Cluster rank weight uses the *mean* discovery step of the cluster's member
anchors (rather than a single anchor's discovery step), so a cluster's
"recency" is its members' average age.

Reclustering happens every `ClusterRefreshSteps` (default 4096) env steps.
With 64 anchors × 8 clusters × 512 dims × ~20 Lloyd's iterations, a single
recluster takes a few ms on CPU — negligible compared to a rollout.

### What problem it solves

**DSC's "no types" gap, addressed properly.** The plain DSC bonus treats
every anchor as an independent point landmark. As noted in § 3, this means
the buffer might have 5 anchors all capturing "near-key" states and 5 more
capturing "after-pickup" states, but DSC will treat each as a separate
discovery — no notion that they're "the same type of subgoal." The agent
gets denser reward but doesn't see categorical structure.

Clustering aggregates the 64 anchors into K=8 cluster centers. Each
cluster is effectively "a kind of state-feature region" — close to a
type. The nearest-cluster lookup means the DSC bonus now operates on
8 discrete reference points that summarize the buffer's geometry,
rather than 64 individual points.

### Mechanism by which it solves it

Three concrete improvements over point-DSC:

1. **Stable typological reference.** A cluster center is the mean of
   ~8 anchors, so it's less noisy than any single anchor. As anchors
   churn through reservoir sampling, the cluster centers move smoothly
   rather than jumping. The agent's bonus landscape is less jittery.

2. **Per-cluster rank weight reflects category recency.** If a cluster
   contains anchors discovered between steps 50k and 100k, its mean age
   is ~75k. As new categories emerge (a new cluster forms around
   recently-added anchors at step 200k), the old cluster's rank weight
   drops *as a category*, not just as individual anchors. The frontier
   pressure is more coherent.

3. **Coverage signal becomes interpretable.** With clusters, "the agent
   has visited cluster 3 this run" is a meaningful statement — cluster 3
   represents a region of state-feature space, not a single arbitrary
   point. The `data/cluster_count` log + nearest-cluster-id traces
   approximate the "type discovery curve" the user asked for.

### Why K=8 specifically

K=8 is chosen relative to NumAnchors=64 (about 8 anchors per cluster on
average). KeyCorridor has 3-4 natural categorical subgoals (explore room,
pick up key, unlock door, reach ball). K=8 gives roughly 2 clusters per
subgoal type — enough for the clusters to represent type variants
(e.g., "door from north" vs "door from south") without over-fragmenting.

The trade-off:
- **K too small** (e.g., K=2): clusters become too coarse, lose
  discriminative power; the bonus landscape flattens.
- **K too large** (e.g., K=32): clusters become individual anchors again,
  losing the aggregation benefit.

### What this does NOT do

Even with clustering, DSC is *still not* doing explicit type-learning in the
strict sense:

- **Cluster IDs are not interpretable categories.** "Cluster 3" is a region
  in 512-d random embedding space; we can call it "the key-pickup type"
  in our intuition, but the model has no symbolic representation of that.
- **No type-conditional reasoning.** The bonus uses cluster distance and
  cluster age, but the policy doesn't know "cluster 3 means key-pickup
  and should precede cluster 5 unlock." That kind of reasoning needs the
  conditional-bonus or hierarchical-RL upgrades from the table above.
- **Cluster discovery is greedy and unsupervised.** K-means finds the
  geometry of whichever anchors were sampled. If the reservoir sample is
  unbalanced (e.g., 50 anchors near the agent's spawn room, 5 elsewhere),
  the clustering reflects that imbalance — not the "true" subgoal
  structure of the task.
- **The clustering is unstable in early training.** Until enough anchors
  accumulate (typically ~30k env steps), K-means runs on a small,
  noisy sample. We mitigate by falling back to anchor-based nearest()
  when `cluster_filled == 0`, but the typing signal is genuinely weak
  early on.

### Files
- `utils.py`: `_kmeans`, `AnchorBuffer.recluster`, `AnchorBuffer.nearest_cluster`.
- `train.py`: periodic recluster call, `nearest_cluster` lookup when
  `UseClusters=True`, `data/cluster_count` TB logging.
- `config.conf`: `UseClusters`, `NumClusters`, `ClusterRefreshSteps`;
  new section `EXP3_KEYCORRIDOR_DSC_TYPED` for the comparison.

### Experiment 3 design with the new variant

The Experiment 3 sweep now has three intervention variants on top of the
RND+OptionB baseline, enabling a three-way comparison:

| Run | Bonus mechanism | What it isolates |
|---|---|---|
| `exp3_keycorridor_baseline` | gated RND only | Option B floor |
| `exp3_keycorridor_dsc` | + point-DSC | gain from per-step landmark bonus |
| `exp3_keycorridor_dsc_typed` | + clustered-DSC | gain from type-aggregated bonus over points |

If typed-DSC outperforms point-DSC, the type-aggregation hypothesis is
supported. If it matches point-DSC, the aggregation didn't carry useful
signal (point landmarks were already enough). If it underperforms, the
clustering noise (early-training instability, unbalanced cluster sizes)
dominates the aggregation benefit.

---

## 4. Inventory observation fusion

### What it is

A wrapper reads `env.unwrapped.carrying` from MiniGrid, encodes the carried
object as a 17-d one-hot (11 object types + 6 colors), and projects it
through a small linear layer (`17 → 64`) before concatenating with the
CNN trunk features. Applied in both `CnnActorCriticNetwork` (policy and
value heads) and `RNDModel` (target and predictor).

```
   image → CNN trunk → 3136-d
                       └──┐
   inventory → linear → 64-d
                         └─── concat ─── 3200-d → feature_head
```

### What problem it solves

**Pixel-identical states with different progress get conflated.** In
KeyCorridor, the agent at the same `(room, x, y)` with or without a
picked-up key produces identical partial-pixel observations — the key
vanishes from the floor when carried, and MiniGrid doesn't draw the
carried object in the partial-observation view. Without inventory fusion,
the RND target features can't distinguish "before pickup" from "after
pickup," so:

- DSC anchors fail to capture key-pickup as a distinct novel event (both
  before-pickup and after-pickup feature vectors map to roughly the same
  region; whichever gets sampled first claims the anchor slot).
- The policy can't condition its action choice on what it's carrying,
  which is exactly what's needed for the unlock-door action.

### Mechanism by which it solves it

The fused 64-d projection of the inventory encoding adds a non-trivial
component to the feature space that *directly distinguishes* carrying
states from non-carrying states. After this addition:

- Two states identical in pixels but differing in inventory have features
  that differ by at least the magnitude of the inventory projection
  (typically ~5-15 units in L2). This is large enough that the nearest-anchor
  lookup distinguishes them.
- DSC can now create separate anchors for "near-key-before-pickup" and
  "carrying-key" — the key-pickup event becomes capturable as a discrete
  novelty.
- The policy network has a direct input signal telling it what it's
  carrying, so it can learn `if carrying[key]: try unlock action` from
  fewer samples than if it had to infer carry-state from pixel residuals.

### Why fuse at the feature layer (not as a separate stream)

The alternative would be to concatenate inventory at the *input* (as an
extra image channel or pre-CNN feature). We fuse at the feature layer
instead because:
- Inventory is symbolic, not spatial — there's no meaningful local-pattern
  structure to extract via convolution.
- Letting it bypass the CNN avoids burning CNN capacity on a trivially
  one-hot signal.
- The 64-d projection is small enough that it doesn't dominate the
  3136-d CNN features, but large enough to be detectable in distance
  computations.

### What this does NOT do

- **Doesn't handle multi-slot inventory.** The 17-d encoding represents
  exactly one carried object. MiniGrid only ever carries one thing at a
  time, so this is fine here — but the design wouldn't generalize to
  envs with arbitrary inventories (Crafter, MineRL).
- **Doesn't capture inventory *history*.** The agent knows what it's
  carrying *now*, not what it picked up and dropped earlier. This is
  fine for MiniGrid which has no drop mechanic but would matter for
  more complex envs.
- **Adds a methodological caveat to the writeup.** We're no longer doing
  pure pixel-only RL on the inventory-bearing experiments. This is fine
  for research on exploration mechanisms — and consistent with what
  BabyAI, Crafter, and other modern benchmarks do — but should be
  declared explicitly in any comparison to pixel-only baselines.

### Methodological note

Inventory reading is via the standard `env.unwrapped.carrying` attribute,
applied uniformly to baseline and intervention. The contribution being
measured is the *exploration algorithm*, not the input pipeline.

### Files
- `envs.py`: `encode_carrying`, applied in `MiniGridEnvironment`.
- `model.py`: `CnnActorCriticNetwork._features` and `RNDModel._fuse`.

---

## 5. Gymnasium migration + MiniGrid harness

### What it is

Full migration of all env wrappers to the gymnasium API (5-tuple `step()`
returning `(obs, reward, terminated, truncated, info)`; `reset()` returning
`(obs, info)`), a new `MiniGridEnvironment` worker process matching the
existing Pipe-based subprocess protocol, MPS device support throughout the
model/agent stack, and `multiprocessing.set_start_method('spawn')` to allow
MPS to work inside subprocess workers.

### What problem it solves

**Montezuma's compute cost makes the demonstration infeasible on M1 Pro.**
The original repo only supported Atari (which needed days of GPU time for
Burda et al.'s published results) and Mario (which required a now-unmaintained
gym wrapper). The migration lets all three target experiments run on M1 Pro
in ~4–6 hours total. Without this layer, none of the rest of the work
matters because the experiments can't be run.

### Mechanism by which it solves it

Two practical wins:
- **MiniGrid is ~100× cheaper than Atari.** A KeyCorridor episode takes
  ~50 ms wall-clock; an Atari episode takes ~5 seconds. The full
  experiment sweep that would be days of GPU time on Atari fits in a
  laptop afternoon.
- **MPS on Apple Silicon gives a free 5-10× speedup** over CPU-only PyTorch
  for the model forward/backward, even with the M1 Pro's modest GPU.
  Without the spawn start method override, multiprocessing workers
  silently lose MPS access (fork is unsafe with MPS); the explicit spawn
  preserves it.

### Why we kept the Pipe-based subprocess protocol

The original repo's training loop talks to env workers through pipes. We
preserved that interface so the rest of `train.py` could remain almost
unchanged — only the subprocess class is new. The trade-off is some message
overhead per step (~0.1 ms per env), which is invisible on Atari and
mildly noticeable on MiniGrid but still produces 350-450 SPS throughput
in smoke tests. A modern alternative (vectorized envs, async API) would
be cleaner but require restructuring the rollout loop, which wasn't
worth the complexity for a research demo.

### What this does NOT do

- **Does not preserve Mario support.** `gym-super-mario-bros` hasn't migrated
  cleanly to gymnasium; `MarioEnvironment` is stubbed to raise
  `NotImplementedError`. Restoring Mario would require either pinning to
  an older gym + shimmy bridge, or rewriting against an unmaintained
  upstream.
- **Does not provide vectorized envs.** Each parallel env runs in its own
  subprocess. This is fine for MiniGrid where the per-env cost is low
  but would not scale efficiently to 128+ workers like the original
  Atari config.
- **Does not abstract over env types.** The training loop still has
  `if env_type == 'atari' else if env_type == 'minigrid'` branches. A
  cleaner factory pattern would be nicer; we kept the explicit branches
  for ease of debugging.

### Files
- `envs.py`: full rewrite.
- `train.py`: env-type branching, spawn start method, MPS device guard.
- `config.py`: `CONFIG_SECTION` env var selects which section becomes
  `default_config`.
- `config.conf`: per-experiment sections under a common `DEFAULT`.

---

## What are anchors?

### Definition

An **anchor** is a single 512-d feature vector — specifically, the output of
`RND.target(s)` evaluated at a state `s` that the agent encountered when its
intrinsic reward was unusually high (above the running 90th percentile of
recent intrinsic rewards). Each anchor is tagged with the global env-step
at which it was first inserted into the buffer.

The buffer (`AnchorBuffer`, capacity 64 by default, configurable via
`NumAnchors`) holds these points and decides whether to admit new candidates
via reservoir sampling — keeping a roughly uniform sample of qualifying
states across the entire training history rather than just the most recent.

### How they're used at runtime

At each rollout step:

1. Compute the current state's target features `f(s) = RND.target(s)`.
2. Find the nearest anchor in L2 distance → `nearest_dist`.
3. Get the discovery timestep of that nearest anchor → translate into
   `rank_weight` (higher for recently-discovered anchors).
4. Compute the DSC multiplier:
   `1 + λ · (nearest_dist / EMA(nearest_dist)) · rank_weight`.
5. Multiply the gated RND signal by this multiplier.

### KeyCorridor example (with explicit caveats)

The anchors aren't human-interpretable, but for intuition imagine the
buffer ends up populated something like:

| Anchor index | Approximate "kind of state" captured | Discovery time |
|---|---|---|
| 0 | First wall/corner pattern | early |
| 1 | First time entering an unexplored room | medium |
| 2 | First time adjacent to the key | medium |
| 3 | First time *carrying* the key (requires inventory fusion to distinguish) | late |
| 4 | First time after door unlock | latest |

Re-visiting near anchor 0 gives little bonus (small rank weight, short
distance). Approaching anchor 4 gives a large bonus (high rank weight,
typically large distance to other anchors). That's how the agent gets
pulled through `explore → find key → carry key → unlock door` in
approximate discovery order rather than getting stuck.

### What anchors are NOT

The "anchor" terminology is suggestive but can mislead about three properties
that anchors **do not have**:

- **Anchors are not types or categories.** Each anchor is one specific
  point in a 512-d embedding, not a class or cluster center. The
  nearest-anchor lookup returns one-of-64, but the buffer doesn't aggregate
  related anchors into "kinds of states." Behaviourally similar states do
  tend to land near similar anchors (because the RND target network is a
  fixed nonlinear projection that preserves some local structure), but
  DSC doesn't actively exploit this — it just returns the nearest single
  point. See § 3 above for what type-learning would actually require.
- **Anchors are not interpretable individually.** They live in a random
  512-d embedding defined by the frozen RND target net's initialization.
  We can only say "this state's features are 12.4 units away from
  anchor 4" — we cannot say "anchor 4 means 'door unlocked'." The table
  above is an *intuition* about what *might* be captured given typical
  exploration trajectories, not a guarantee about what *is* in the
  buffer for any given run.
- **Anchors are not goals in any planning sense.** There's no policy that
  navigates to a specific anchor. The bonus is a per-step reward
  modification only. Goal-conditioned policies (which DO let you say
  "navigate to anchor 4") would require additional architecture — a
  goal-conditioned value head, hindsight-experience-replay, or
  hierarchical options.

### Concrete state stored in `AnchorBuffer`

- `features` — `(64, 512)` numpy array of target features.
- `discovery_step` — `(64,)` int64 array of global env-steps at insertion.
- `dist_ema` — running mean of nearest-anchor distances, used to normalize
  the distance bonus to be scale-invariant.
- `filled` — current number of valid anchors (grows from 0 to 64 as the
  buffer fills, then stays at 64).
- `_reward_buffer` — deque of recent intrinsic rewards (capacity 2000) used
  to compute the 90th-percentile insertion threshold.
- `_candidates_seen` — total number of qualifying states ever considered
  for insertion (used for reservoir-sampling probability).

### What we log to TensorBoard

- `data/anchor_coverage` — `AnchorBuffer.filled` value, logged per rollout.
  Grows from 0 to `NumAnchors` and then stays. The shape of the growth
  curve is informative: a slow growth + early plateau suggests insertion
  thresholds are too strict; a fast fill suggests the threshold is too
  permissive. A healthy run fills to capacity within the first ~100k env
  steps then stays full.

This metric is the closest we get to verifying the sequence-of-novelty
claim empirically. It says nothing about whether the agent is *visiting*
the anchors in a useful order — only that the buffer is being populated.
For visit-order analysis, you'd need additional logging (per-anchor visit
counts, anchor-visit-by-episode heatmaps) that isn't currently wired up.

---

## Sparseness preservation: do these augmentations keep RND's sparse-reward benefits?

RND's value proposition on sparse-reward tasks rests on four properties:

1. **Continuous per-step intrinsic signal** — every step gets a non-zero
   bonus, so PPO has gradient information even when extrinsic reward is
   zero for thousands of steps.
2. **Novelty-correlated** — the bonus is larger in unfamiliar states,
   biasing exploration toward unknown territory.
3. **No dependence on extrinsic reward signal** — the intrinsic stream
   works even when the agent has never seen a single positive reward.
4. **Predictor convergence over time** — eventually the predictor catches
   up to well-visited regions and the intrinsic signal there fades,
   leaving the agent to optimize extrinsic in those regions.

How each augmentation interacts with these:

### Option B (bootstrap ensemble + variance gate)

| Property | Preserved? | Details |
|---|---|---|
| Continuous per-step signal | **Mostly** | Gate is multiplicative and nonzero in any region with even small ensemble variance |
| Novelty-correlated | Yes | Higher variance ≈ less-trained region ≈ novel; gate opens there |
| No extrinsic dependence | **Partial — important caveat** | See below |
| Convergence over time | Yes, accelerated | Gate provides a second convergence mechanism alongside RND predictor |

**The honest caveat.** Option B's gate uses *extrinsic critic ensemble
variance* as its uncertainty signal. On tasks with no extrinsic reward
*ever* (pure-exploration regimes), all critic heads converge to V=0
everywhere — variance is low and uniform, gate hovers around 0.5α (the
typical-state baseline), attenuating intrinsic reward by ~2× without
discriminative benefit. So Option B will *slightly degrade* pure-exploration
performance.

For sparse-but-eventually-positive reward (Montezuma, KeyCorridor), the
gate has signal to work with once any reward has been observed once: states
near reward-bearing regions have non-trivial variance, states far from them
do not. The discriminative benefit kicks in. Pre-first-reward, the
attenuation is ~2× uniformly — undesirable but tolerable.

**Bottom line for sparseness:** Option B is **slightly worse than vanilla
RND on pure exploration**, **roughly neutral on extreme sparse-reward**,
and **strictly better on dense or shaped rewards** (which is where the
Gravitar failure happens). For the experiments we're running (LavaCrossing,
DoorKey, KeyCorridor), at least some extrinsic signal exists, so the
caveat is bounded.

### DSC (point-based)

| Property | Preserved? | Details |
|---|---|---|
| Continuous per-step signal | Yes, *strengthened* | Multiplier `1 + λ·norm_dist·rank_weight` is always ≥ 1 |
| Novelty-correlated | Yes | Distance to nearest anchor is itself a feature-space novelty measure |
| No extrinsic dependence | Yes | Anchor insertion uses intrinsic-reward percentile, not extrinsic |
| Convergence over time | **No — anchors don't decay** | See below |

**The honest caveat in the opposite direction.** Unlike RND's predictor,
the anchor buffer doesn't "learn away" novelty. Anchors stay populated;
distances stay nonzero. So the DSC bonus *doesn't naturally decay* the way
RND does. This is mostly a feature — it means DSC retains exploration
signal long after the RND predictor would have saturated, which is exactly
why it addresses weakness #12. But it does mean the agent has a permanent
multiplicative drive to seek anchor-distant or new-anchor-near states,
even after extrinsic learning has converged in those regions.

In a no-reward limit, DSC alone would keep the agent perpetually exploring,
which is what you want from a curiosity bonus. Combined with Option B's
gate, the gate provides the "settle down once extrinsic is figured out"
mechanism that DSC by itself lacks.

**Bottom line for sparseness:** DSC **fully preserves RND's sparse-reward
benefits** and **enhances them via decay-resistance**.

### DSC type-learning (K-means clusters)

Same properties as point-DSC, with one additional consideration:

- Cluster centers shift as anchors update, which means the bonus
  landscape is mildly *dynamic* even at a fixed state. This could be a
  feature (keeps exploration moving when the agent re-visits) or a
  destabilizer (the policy is chasing a moving target). In smoke tests
  this hasn't been visible; if it matters empirically, increasing
  `ClusterRefreshSteps` would slow the dynamics.

**Bottom line:** sparseness-neutral relative to point-DSC.

### Inventory observation fusion

Sparseness-neutral. The fusion only changes the *representation* on which
RND, Option B, and DSC operate. It doesn't change the density or magnitude
of the intrinsic reward signal.

### Gymnasium migration + MiniGrid harness

Sparseness-neutral. Pure infrastructure.

---

## Summary: combined-stack effect on sparseness

For the full RND + Option B + DSC + DSC-typed + inventory stack on a
sparse-reward task (KeyCorridor):

| Phase of training | What dominates |
|---|---|
| Before any reward is seen (pure exploration) | DSC bonus carries the signal; Option B gate ≈ 0.5α uniform; RND predictor still active |
| Once first reward observed | Option B gate becomes discriminative (suppresses intrinsic in mature regions, opens it in novel ones); DSC continues providing decay-resistant landmark signal |
| Late training | RND predictor mostly saturated; Option B gate mostly closed in mature regions; DSC distance bonus is the primary remaining exploration drive |

So the augmentations preserve sparse-reward viability throughout, with DSC
specifically compensating for both the RND-predictor-saturation problem
(#12) and the Option B gate's pure-exploration weakness. The
counter-intuitive observation is that **DSC is doing important work even
on tasks where its sequencing/typing claims are weak** — it's also
serving as an anti-saturation backstop for the RND signal.

---

## Weakness coverage (post-fixes)

See § 6 for the final shipped enhancements. This table covers the original
Option B + DSC plan. The Option B row is accurate as implemented; the DSC
row reflects intended design rather than primary shipped result.

| Weakness | Vanilla RND | Option B | NovelD + SimHash (shipped) |
|---|---|---|---|
| #3 dancing with skulls (Pitfall) | — | Coded but inoperative — variance ≈ 0 pre-reward; see § 6 | **partial** — SimHash count suppresses re-visits to lava; NovelD episodic bonus disfavors already-explored regions |
| #9 Gravitar failure | — | Gate uniformly attenuates 5× (not state-specific) | Not directly addressed; IntCoef ablation is the cleanest fix |
| #4 local-only exploration | — | — | **partial** — SimHash global count creates pressure toward globally-unvisited states |
| #12 predictor saturation | — | partial | **partial** — SimHash signal is independent of predictor MSE |
| #1, #2, #5, #6, #7, #8, #10, #11 | — | — | — (orthogonal to all interventions tested) |

---

## 6. Final shipped enhancements

The following three enhancements replaced Option B + DSC as the primary
interventions after multi-seed analysis. Full mechanism analysis and
empirical results are in `reports/rnd_enhancement_constraints.md`.

### 6.1 NovelD with K-means cluster types (Zhang et al. 2021 + cluster extension)

**What it is**: At each step, the intrinsic reward is `max(RND(s') - α·RND(s), 0) × (1/sqrt(N(cluster_id(s'))))` where `cluster_id` is the nearest K=8 K-means center over the RND target features, and N is an episodic visit counter reset at each episode boundary.

**Files**: `utils.py:FeatureClusterer`, `train.py` (episodic counter, periodic recluster every `ClusterRefreshSteps=4096` steps).

**What it does**: Episodic count bonus suppresses re-exploration of already-visited feature-space regions within an episode. The cluster abstraction (K=8 over 512-d RND features) groups visually similar states, providing coarse semantic discrimination.

**Empirical finding**: Best time-to-goal on LavaCrossing (563k, seed 0) and DoorKey (97.3k, seed 0). Adds noise on KeyCorridor where vanilla RND already works — time-to-goal 360k vs baseline 325k (FAIL). The failure is attributable to coarse granularity (K=8 too few distinct regions for KeyCorridor's layout) and episodic decay rate mismatched to the baseline's already-effective exploration.

### 6.2 SimHash additive bonus (Tang et al. 2017)

**What it is**: A 64-bit random-projection hash `h(s) = sign(A·flatten(obs))` (A is a fixed random matrix) maps each observation to a binary vector. A global visit counter N(h) accumulates across all episodes. The bonus `λ/sqrt(N(h))` is added to the RND intrinsic reward each step.

**Files**: `utils.py:SimHashCounter`, `train.py` (global counter, additive combination with RND MSE).

**What it does**: Provides per-step coverage bonus at near-pixel resolution (25k unique hashes on LavaCrossing over 1M steps; 2.9k on KeyCorridor over 1.5M steps). Never saturates within the training budget. Additive form cannot suppress the RND signal.

**Empirical finding**: 16.5% faster convergence on KeyCorridor (n=3 matched seeds: 267.6k ±12.7k vs vanilla 320.5k ±3.6k). Consistent per-seed improvement (+13.5%, +21.2%, +14.8%). SimHash-only (without RND) fails completely — 0.000 extrinsic return at 1.5M steps despite 1.4M unique hash observations — confirming RND provides the essential directional signal and SimHash provides complementary coverage.

### 6.3 Posterior Sampling (Osband et al. 2016)

**What it is**: The K=5 extrinsic critic ensemble (built for Option B) is reused with a different aggregation: instead of pessimistic `min`, each env worker is assigned one randomly-sampled head index at the start of each rollout. That head's value estimate drives the extrinsic advantage for the entire rollout. Heads are resampled at rollout boundaries.

**Files**: `train.py` (`active_heads` array, resampled per rollout via `np.random.randint`).

**What it does**: Provides trajectory-level exploration commitment — each env "believes" in one value model for 128 steps, creating coherent episode-length behaviour diversity across the 8 workers. This is the mechanism Bootstrap DQN was originally designed for, as opposed to Option B's pessimistic use.

**Empirical finding**: Slight improvement alone (288.8k vs 325.6k baseline on KeyCorridor, FAIL by 12% threshold). Catastrophic when stacked with NovelD — 580k vs 360k for NovelD alone (FAIL). The stacking failure occurs because NovelD's persistently high episodic bonus (int_reward 4.53 → 2.77) combined with per-rollout head commitment amplifies signal noise: the agent commits to one value estimate while receiving a noisy count bonus that doesn't align with that head's value landscape, producing inconsistent gradients that slow convergence.
