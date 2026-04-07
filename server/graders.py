"""
Deterministic graders for the Content Moderation Environment.

All public score values are strictly within the open interval (0, 1).
No value ever equals exactly 0.0 or 1.0 — this satisfies the OpenEnv
Phase 2 validator requirement that task scores be strictly between 0 and 1.

Score design
------------
Raw internal scores are computed on a [-1, 1] scale and then mapped into
(0, 1) via a linear normalisation:

    normalised = (raw + 1) / 2          maps [-1, 1] -> [0, 1]

We then clamp away the endpoints by enforcing a minimum of SCORE_MIN and
a maximum of SCORE_MAX, both strictly inside (0, 1):

    SCORE_MIN = 0.01   (replaces 0.0)
    SCORE_MAX = 0.99   (replaces 1.0)

This preserves the full ordering of all scores while guaranteeing no
boundary value is ever returned.
"""

from typing import Optional

# Strict open-interval boundaries
SCORE_MIN: float = 0.01
SCORE_MAX: float = 0.99

# ── Raw decision scores (internal, before normalisation) ──────────────────
# Raw range is [-1.0, 1.0]; we normalise to (0, 1) before returning.
_DECISION_RAW = {
    ("allow",  "allow"):   1.0,
    ("warn",   "warn"):    1.0,
    ("remove", "remove"):  1.0,
    # Off-by-one on severity scale
    ("warn",   "allow"):  -0.4,   # over-moderated safe post
    ("allow",  "warn"):   -0.4,   # under-moderated warn-worthy post
    ("warn",   "remove"):  0.0,   # partial – better than nothing
    ("remove", "warn"):    0.0,   # stricter than needed, defensible
    # Dangerous mistake
    ("allow",  "remove"): -1.0,   # allowed clearly harmful content
    ("remove", "allow"):  -0.8,   # wrongly removed safe content
}


def _to_open(raw: float) -> float:
    """
    Map a raw score in [-1.0, 1.0] to the open interval (SCORE_MIN, SCORE_MAX).

        normalised = (raw + 1.0) / 2.0        [-1,1] -> [0,1]
        clamped    = clip(normalised, MIN, MAX) [0,1] -> (MIN, MAX)
    """
    normalised = (raw + 1.0) / 2.0
    return round(max(SCORE_MIN, min(SCORE_MAX, normalised)), 4)


def grade_classification(predicted: Optional[str], ground_truth: str) -> float:
    """
    Classification grader.

    Returns a value strictly in (SCORE_MIN, SCORE_MAX):
        correct  -> SCORE_MAX (0.99)
        wrong    -> SCORE_MIN (0.01)
        None     -> SCORE_MIN (0.01)
    """
    if predicted is None:
        return SCORE_MIN
    raw = 1.0 if predicted == ground_truth else -1.0
    return _to_open(raw)


def grade_decision(predicted: Optional[str], ground_truth: str) -> float:
    """
    Partial-credit moderation decision grader.

    Returns a value strictly in (SCORE_MIN, SCORE_MAX).
    The raw score is looked up from _DECISION_RAW and normalised via _to_open().
    """
    if predicted is None:
        return SCORE_MIN
    raw = _DECISION_RAW.get((predicted, ground_truth), -1.0)
    return _to_open(raw)


def compute_episode_reward(
    classification_score: float,
    decision_score: float,
    step_count: int,
    early_bonus_threshold: int = 2,
) -> float:
    """
    Combine classification and decision scores into a single episode reward
    strictly within (SCORE_MIN, SCORE_MAX).

    Both input scores are already normalised (0, 1) values from the graders above.

    Formula (in normalised space):
        base         = 0.4 * classification_score + 0.6 * decision_score
        step_penalty = max(0, step_count - threshold) * 0.02
        early_bonus  = 0.02 if step_count <= threshold and base > 0.5 else 0.0
        raw          = base - step_penalty + early_bonus

    Result is clamped to (SCORE_MIN, SCORE_MAX).
    """
    base = 0.4 * classification_score + 0.6 * decision_score
    extra_steps = max(0, step_count - early_bonus_threshold)
    step_penalty = extra_steps * 0.02
    early_bonus = 0.02 if (step_count <= early_bonus_threshold and base > 0.5) else 0.0
    final = base - step_penalty + early_bonus
    return round(max(SCORE_MIN, min(SCORE_MAX, final)), 4)


def grade_step(
    action_type: str,
    agent_label: Optional[str],
    agent_decision: Optional[str],
    ground_truth_label: str,
    ground_truth_decision: str,
    step_count: int,
    is_done: bool,
) -> float:
    """
    Return an immediate step reward, always strictly within (SCORE_MIN, SCORE_MAX).

    Intermediate steps receive partial credit for correct classification.
    The final step returns the full episode reward.
    """
    # Small per-step cost expressed as a reduction from the midpoint (0.5)
    STEP_COST = 0.01   # subtracted from intermediate score

    if not is_done:
        if action_type == "classify" and agent_label is not None:
            clf = grade_classification(agent_label, ground_truth_label)
            # Blend toward midpoint and penalise slightly for the step
            raw = 0.4 * clf + 0.6 * 0.5 - STEP_COST
            return round(max(SCORE_MIN, min(SCORE_MAX, raw)), 4)
        # No useful action taken
        mid = 0.5 - STEP_COST
        return round(max(SCORE_MIN, min(SCORE_MAX, mid)), 4)

    clf_score = grade_classification(agent_label, ground_truth_label)
    dec_score = grade_decision(agent_decision, ground_truth_decision)
    return compute_episode_reward(clf_score, dec_score, step_count)
