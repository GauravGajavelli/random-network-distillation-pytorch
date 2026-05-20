# Future Work

This document catalogs open directions for extending the current RND + Option B + DSC stack. It is organized from near-term incremental improvements to longer-horizon architectural changes.

---

## 1. Subgoal typing: from anonymous anchors to named subgoal categories

### Current limitation

DSC anchors are individual points in a 512-d RND target-network embedding. The `UseClusters` flag partially addresses this by running K-means over the anchor buffer, but the resulting cluster centers are still anonymous — we can say "this state is near cluster 3" but not "cluster 3 means *key acquired*."

The architecture document explicitly flags this gap:

> *"Anchors are not types or categories. Each anchor is one specific point in a 512-d embedding, not a class or cluster center."*

### What subgoal typing would look like

A subgoal type is a predicate that is true of a set of states sharing a semantically meaningful property — door unlocked, key held, room entered. Getting there requires:

1. **Labeled anchor clusters.** Run a second-pass classifier (or contrastive probe) over the anchor buffer that maps cluster IDs to human-readable labels. For MiniGrid environments, the ground-truth observation dict gives us these labels for free (object type, carried item, door state).

2. **Type-conditional curiosity.** Instead of one DSC bonus, maintain a separate distance signal per subgoal type. Bonus for "how far from the nearest *door* anchor" is independent of "how far from the nearest *key* anchor." This gives the agent structured novelty across semantic dimensions rather than a single scalar distance.

3. **Curriculum over types.** Sort subgoal types by their current discovery rate and prioritize the frontier type — the one being discovered but not yet mastered. This addresses weakness #1 (short-term bias) more directly than the current rank-weight heuristic.

---

## 2. Skills: policies that navigate between subgoal types

### Current limitation

There is no policy that can be told "get to a state of type T." DSC modifies per-step reward but provides no mechanism for directed navigation to a specific subgoal. Reaching a new subgoal type is fully emergent from the curiosity bonus, not planned.

> *"Goal-conditioned policies (which DO let you say 'navigate to anchor 4') would require additional architecture — a goal-conditioned value head, hindsight-experience-replay, or hierarchical options."* — ARCHITECTURE.md

### Skill definition

A **skill** (in the options-framework sense) is a triple `(I, π_g, β)`:
- `I` — initiation set: the set of states from which the skill can be invoked.
- `π_g` — a goal-conditioned policy that drives the agent toward subgoal type `g`.
- `β` — termination condition: `β(s) = True` when the agent reaches any state of type `g`, or when a timeout expires.

### How to build skills over subgoal types

#### Step 1 — define subgoal types (prerequisite: § 1 above)

Each subgoal type becomes a potential skill goal. For a KeyCorridor task the natural type set is:
`{start, key_visible, key_picked_up, door_visible, door_unlocked, goal_reached}`.

#### Step 2 — collect inter-type transitions via HER

Run the current RND + DSC agent and log every episode as a sequence of subgoal-type visits. For each transition `(s_t, a_t, ..., s_{t+k})` where `s_{t+k}` is of a new type, record it as a training tuple for the goal-conditioned policy `π_g` with `g = type(s_{t+k})`. Apply **Hindsight Experience Replay (HER)** to relabel trajectories: any trajectory that ends at type `g'` can be relabeled as a successful demonstration for skill `g'`.

#### Step 3 — train goal-conditioned value heads

Augment the existing network with a goal-conditioned critic `V(s, g)` and a goal-conditioned actor `π(a | s, g)`. The goal `g` can be embedded as:
- A one-hot vector over subgoal types (simple, requires fixed type set).
- A prototype embedding: the mean anchor feature vector for type `g` (richer, generalizes to new types without architecture change).

The prototype embedding approach integrates cleanly with the existing anchor buffer: `goal_embed = mean(features[cluster == g])`.

#### Step 4 — compose skills into plans via a meta-controller

A high-level meta-controller (a second PPO agent operating at a slower timescale) selects which skill to invoke at each decision point. It observes the current subgoal-type and the task structure (available subgoal types, which have been visited) and outputs a skill index.

```
Meta-controller (slow): [state_type_t] → skill g
    ↓
Skill π_g (fast):        [s_t, goal=g] → a_t, a_{t+1}, ..., until β(s) or timeout
    ↓
Subgoal buffer:          record (type_at_skill_start, g, steps_taken, success)
```

This is essentially a **two-level MAXQ / Option-Critic** architecture. The meta-controller's action space is the set of known subgoal types; its reward is the sparse extrinsic reward of the original task.

#### Step 5 — intrinsic reward for skill transitions

Retain the DSC bonus but apply it at the skill level: the bonus for invoking skill `g` is proportional to how long since type `g` was last reached (recency weight) and how many novel inter-type transitions the skill would produce. This prevents the meta-controller from looping on easy skills.

---

## 3. Inter-type transition graph

Maintaining an explicit directed graph `G = (V, E)` where:
- `V` = discovered subgoal types
- `E` = `(type_i → type_j)` if the agent has ever transitioned between them
- edge weight = empirical success rate of skill `π_{type_j}` when invoked from a state of `type_i`

This graph serves multiple purposes:
- **Planning:** A* or Dijkstra over `G` gives a skill sequence for reaching the goal type from the current type.
- **Curriculum:** edges with low success rates identify skills that need more training.
- **Novelty detection:** a new subgoal type is an unseen node; a new skill path is an unseen edge. DSC-style bonuses can target edge novelty rather than state-space novelty.

---

## 4. Uncovered RND weaknesses that skills address

| Weakness | Current coverage | How skill-over-subgoal-types addresses it |
|---|---|---|
| #2 no episodic memory | partial (anchors are non-episodic) | Skill selection conditioned on subgoal-type visit history within episode gives genuine episodic memory |
| #4 local-only exploration | fixed by DSC anchor sequence | Meta-controller can plan globally across the type graph; no longer emergent |
| #5 recurrent policies hurt | unaddressed | Each skill can use a short-horizon recurrent policy; the meta-controller operates on type-level state (no long-range memory needed) |
| #10 episodic/non-episodic combination | heuristic | Natural separation: skills are episodic (reset on termination), meta-controller is non-episodic |

---

## 5. Near-term experiments

The following can be done incrementally without the full skill architecture:

| Experiment | What it tests | Config change |
|---|---|---|
| Labeled cluster probes on MiniGrid | Whether `NumClusters` K-means meaningfully separates semantic types | Add ground-truth label logging per anchor; measure cluster purity |
| Per-type DSC bonus (additive) | Whether separating distance by type helps KeyCorridor vs. single-distance DSC | New `UseTypedDSC` flag; sum of per-type distance bonuses |
| Prototype goal embedding (frozen policy) | Whether `mean(features[cluster == g])` is a useful goal representation for HER labeling | Off-policy evaluation only; no new training needed |
| Subgoal-type visit sequence logging | Characterize the natural order types are discovered under current DSC | Add `type_visit_log` to TensorBoard; no policy change |

---

## 6. Longer-horizon directions

- **Procedurally generated type sets.** For tasks with unknown structure (Atari), learn subgoal types from clustering without ground-truth labels, using a contrastive auxiliary loss to push temporally distant features apart and temporally close features together (similar to BYOL-Explore or SPR).
- **Skill reuse across tasks.** If the same type graph (e.g., *pick up object → unlock door → reach goal*) appears in multiple environments, skills trained on one environment should transfer to another. This requires a factored type representation that abstracts away environment-specific visual details.
- **Option-Critic end-to-end training.** Rather than the two-stage process above (train skills, then train meta-controller), train the full hierarchy jointly using the Option-Critic gradient, with subgoal types as a soft inductive bias rather than a hard constraint on the option set.
