"""
Deterministic graders for the Content Moderation Environment.
All graders return a float in [-1.0, 1.1]. No randomness.
"""

from typing import Optional

_DECISION_PARTIAL = {
    ("allow",  "allow"):  1.0,
    ("warn",   "warn"):   1.0,
    ("remove", "remove"): 1.0,
    # Off-by-one on severity scale
    ("warn",   "allow"):  0.3,   # over-moderated safe post
    ("allow",  "warn"):   0.3,   # under-moderated warn-worthy post
    ("warn",   "remove"): 0.5,   # partial – better than nothing
    ("remove", "warn"):   0.5,   # stricter than needed, defensible
    # Dangerous mistakes
    ("allow",  "remove"): -1.0,  # allowed clearly harmful content
    ("remove", "allow"):  0.1,   # wrongly removed safe content (less bad)
}


def grade_classification(predicted: Optional[str], ground_truth: str) -> float:
    """Exact-match: 1.0 correct, 0.0 wrong or None."""
    if predicted is None:
        return 0.0
    return 1.0 if predicted == ground_truth else 0.0


def grade_decision(predicted: Optional[str], ground_truth: str) -> float:
    """Partial-credit decision grader. Range: [-1.0, 1.0]."""
    if predicted is None:
        return 0.0
    return _DECISION_PARTIAL.get((predicted, ground_truth), 0.0)


def compute_episode_reward(
    classification_score: float,
    decision_score: float,
    step_count: int,
    early_bonus_threshold: int = 2,
) -> float:
    """
    base_reward  = 0.4 * classification + 0.6 * decision
    step_penalty = extra_steps * 0.05
    early_bonus  = 0.1 if done within threshold AND base_reward > 0
    """
    base_reward = 0.4 * classification_score + 0.6 * decision_score
    extra_steps = max(0, step_count - early_bonus_threshold)
    step_penalty = extra_steps * 0.05
    early_bonus = 0.1 if (step_count <= early_bonus_threshold and base_reward > 0) else 0.0
    final = base_reward - step_penalty + early_bonus
    return round(max(-1.0, min(1.1, final)), 4)


def grade_step(
    action_type: str,
    agent_label: Optional[str],
    agent_decision: Optional[str],
    ground_truth_label: str,
    ground_truth_decision: str,
    step_count: int,
    is_done: bool,
) -> float:
    """Return an immediate step reward."""
    STEP_COST = -0.02

    if not is_done:
        if action_type == "classify" and agent_label is not None:
            clf_score = grade_classification(agent_label, ground_truth_label)
            return round(0.4 * clf_score + STEP_COST, 4)
        return round(STEP_COST, 4)

    clf_score = grade_classification(agent_label, ground_truth_label)
    dec_score = grade_decision(agent_decision, ground_truth_decision)
    return compute_episode_reward(clf_score, dec_score, step_count)
