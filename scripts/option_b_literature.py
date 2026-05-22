"""Published-method reference data for the Option B analysis report.

Hardcoded comparison points between our Option B variant and the published
methods it draws from or relates to. Used by `analyze_option_b.py` to
generate the "Comparison to Published Methods" section.

Key positioning claim:
    Option B is a hybrid of TD3 (pessimistic min) and Bootstrap DQN
    (per-sample bootstrap masks) applied to a setting (online PPO + RND
    for sparse-reward exploration) where they hadn't been combined before,
    with one novel ingredient: the variance-gated intrinsic reward
    (no published direct analog).
"""

LITERATURE = {
    "TD3": {
        "citation": "Fujimoto, van Hoof, Meger. \"Addressing Function Approximation "
                    "Error in Actor-Critic Methods.\" ICML 2018.",
        "K_critics": "2",
        "use_of_ensemble": "min(Q1, Q2) for value pessimism in continuous control",
        "bootstrap_masks": False,
        "variance_gate": False,
        "intrinsic_reward": False,
        "domain": "online, continuous control, dense reward",
        "what_we_borrow": "The pessimistic-min mechanism for value estimation. "
                           "min over heads provides estimation-error pessimism "
                           "that biases the policy away from over-optimistic states.",
        "what_we_change": "K = 2 -> K = 5 (more heads for stronger min effect); "
                           "added Bernoulli(0.8) per-sample bootstrap masks for "
                           "training diversity; applied to PPO (discrete actor-critic) "
                           "rather than TD3 (deterministic continuous); added "
                           "variance-gated intrinsic reward (RND-specific).",
        "alignment_with_our_results": "TD3 expects pessimism to help in settings "
                                       "with informative negative-extrinsic signal. "
                                       "Our LavaCrossing positive result (death_rate "
                                       "0.80 -> 0.002, extr_return 0.001 -> 0.834) is "
                                       "consistent: lava-death IS the negative signal "
                                       "that pessimistic min amplifies.",
    },

    "Bootstrap DQN": {
        "citation": "Osband, Blundell, Pritzel, Van Roy. \"Deep Exploration via "
                    "Bootstrapped DQN.\" NeurIPS 2016.",
        "K_critics": "10",
        "use_of_ensemble": "Posterior sampling for exploration (one head per "
                            "episode), NOT pessimism",
        "bootstrap_masks": True,
        "variance_gate": False,
        "intrinsic_reward": False,
        "domain": "online, discrete actions, exploration-driven",
        "what_we_borrow": "The per-sample bootstrap masking trick that creates "
                           "diverse heads. Each head sees ~80% of samples (Bernoulli "
                           "p=0.8) so heads' value estimates diverge.",
        "what_we_change": "We use the resulting ensemble for pessimistic value "
                           "estimation (TD3 style) instead of for posterior-sampled "
                           "exploration. This is the OPPOSITE of what Osband et al. "
                           "argue should be done with the same machinery.",
        "alignment_with_our_results": "Osband et al. explicitly warn that ensemble "
                                       "pessimism is the WRONG use of a critic "
                                       "ensemble in online RL — posterior sampling "
                                       "for exploration is correct. Our DoorKey "
                                       "negative result (extr 0.96 -> 0.29) "
                                       "rediscovers this insight: pessimism on a "
                                       "sparse-reward task where intrinsic is "
                                       "genuinely useful suppresses the wrong signal.",
    },

    "REDQ / EDAC": {
        "citation": "Chen, Wang, Zhou, Ross. \"Randomized Ensembled Double "
                    "Q-Learning: Learning Fast Without a Model.\" ICLR 2021. "
                    "An, Moon, Kim, Song. \"Uncertainty-Based Offline RL with "
                    "Diversified Q-Ensemble.\" NeurIPS 2021.",
        "K_critics": "10+ (REDQ uses K=10, subset M=2; EDAC uses K=10+)",
        "use_of_ensemble": "Pessimistic min over ensemble (or random subset) for "
                            "online sample efficiency (REDQ) or offline RL stability "
                            "(EDAC).",
        "bootstrap_masks": False,
        "variance_gate": False,
        "intrinsic_reward": False,
        "domain": "online SAC (REDQ) or offline RL (EDAC)",
        "what_we_borrow": "Confirmation that K > 2 ensemble pessimism is a known "
                           "and effective technique for value-based RL.",
        "what_we_change": "We use K = 5 (compromise for compute on M1 Pro); we "
                           "add bootstrap masks (neither REDQ nor EDAC mask); we "
                           "operate in PPO + RND (not SAC, not offline); we add "
                           "the variance gate on the intrinsic stream.",
        "alignment_with_our_results": "Both methods target value-overestimation "
                                       "in value-based RL. We target the same "
                                       "phenomenon in PPO + RND on sparse-reward "
                                       "exploration tasks. Consistent positive "
                                       "result on LavaCrossing.",
    },

    "Disagreement Curiosity": {
        "citation": "Pathak, Gandhi, Gupta. \"Self-Supervised Exploration via "
                    "Disagreement.\" ICML 2019.",
        "K_critics": "K predictor networks (forward dynamics models)",
        "use_of_ensemble": "Inter-predictor variance USED AS the intrinsic reward "
                            "(generator), driving exploration toward states where "
                            "models disagree.",
        "bootstrap_masks": False,
        "variance_gate": False,
        "intrinsic_reward": True,
        "domain": "online, model-based curiosity",
        "what_we_borrow": "The conceptual basis: ensemble variance is a meaningful "
                           "novelty/uncertainty signal.",
        "what_we_change": "We ensemble CRITICS (value), not predictors (forward "
                           "dynamics); we use the variance to GATE an externally-"
                           "computed intrinsic reward, not to GENERATE one; the "
                           "direction is structurally opposite — Pathak amplifies "
                           "intrinsic where variance is high, we suppress intrinsic "
                           "where variance is low.",
        "alignment_with_our_results": "Both methods treat ensemble disagreement "
                                       "as informative; the closest analog in spirit. "
                                       "But the gating direction matters: our gate "
                                       "with floor 0.2 prevents the suppression "
                                       "collapse that motivates Pathak's amplification "
                                       "approach.",
    },
}

NOVEL_INGREDIENTS = [
    "Variance-gated intrinsic reward with floor:"
    " r_int * clip(alpha * var / (var + EMA(var)), 0.2, 1.0)."
    " No published method gates an externally-computed intrinsic reward by"
    " an ensemble's value-variance signal. Disagreement Curiosity (Pathak"
    " et al. 2019) is the closest analog in spirit but uses variance as"
    " the generator, not as a regulator.",

    "Specific combination of TD3's pessimistic min + Bootstrap DQN's"
    " per-sample bootstrap masks applied to a curiosity-driven online"
    " agent (PPO + RND). Each ingredient is published individually;"
    " the specific combination plus application to sparse-reward"
    " curiosity-driven RL is not.",

    "Per-head 2-layer MLP critic heads on a shared CNN trunk (justified"
    " by the diagnostic finding that shared-trunk + linear heads gave"
    " effectively zero ensemble diversity, making min(V_k) approx mean(V_k)"
    " and defeating the pessimism mechanism). This architectural choice"
    " is more independent than TD3's twin heads but cheaper than fully"
    " separate ensemble networks.",
]


def positioning_summary():
    """One-paragraph elevator-pitch positioning of Option B vs the literature."""
    return (
        "Option B is best understood as a hybrid of TD3 (Fujimoto et al. 2018) "
        "and Bootstrap DQN (Osband et al. 2016) applied to a setting where they "
        "had not been combined: online PPO + RND for sparse-reward exploration. "
        "From TD3 we borrow the pessimistic-min mechanism; from Bootstrap DQN "
        "we borrow the per-sample bootstrap masks. Where Option B is genuinely "
        "novel is the variance-gated intrinsic reward (no direct published "
        "analog) and the specific composition of these mechanisms applied to "
        "curiosity-driven RL. No single published method is identical to Option B."
    )
