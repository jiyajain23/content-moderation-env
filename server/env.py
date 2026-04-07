"""
Core environment logic for the Context-Aware Content Moderation Environment.

Implements the OpenEnv contract:
    reset(task_id) -> Observation
    step(action)   -> (Observation, reward, done, info)
    state()        -> State
"""

from typing import Optional, Tuple, Dict, Any
 
from .models import Action, Observation, State
from .tasks import get_task, TASKS, get_all_task_ids
from .graders import (
    grade_step, grade_classification, grade_decision,
    compute_episode_reward, SCORE_MIN, SCORE_MAX,
)

MAX_STEPS = 10  # Safety cap to prevent infinite loops


class ModerationEnv:
    """
    Content moderation environment.

    Episode flow:
        1. reset(task_id) – load a task, return initial observation
        2. step(classify action) – agent labels the post
        3. step(moderate action) – agent decides enforcement; episode ends
    """

    def __init__(self) -> None:
        self._state: Optional[State] = None

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def reset(self, task_id: Optional[str] = None) -> Observation:
        """Load a task (default: first task) and return the initial observation."""
        if task_id is None:
            task_id = TASKS[0]["task_id"]

        task = get_task(task_id)

        self._state = State(
            task_id=task["task_id"],
            post_id=task["post_id"],
            content=task["content"],
            user_history=task["user_history"],
            flags=task["flags"],
            platform_rules=task["platform_rules"],
            ground_truth_label=task["ground_truth_label"],
            ground_truth_decision=task["ground_truth_decision"],
        )

        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Apply an action and return (observation, reward, done, info).

        Raises:
            RuntimeError: if reset() has not been called.
            ValueError: if the action is malformed or out of sequence.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        s = self._state
        error = action.validate_action()
        if error:
            raise ValueError(f"Invalid action: {error}")

        # Guard against acting in a finished episode
        if s.done:
            raise ValueError("Episode is already done. Call reset() to start a new episode.")

        # Guard against looping
        if s.step_count >= MAX_STEPS:
            s.done = True
            obs = self._build_observation()
            return obs, -0.1, True, {"error": "max_steps_exceeded"}

        # Enforce ordering: must classify before moderating
        if action.action_type == "moderate" and not s.classified:
            raise ValueError("Must classify the post before moderating it.")

        # Enforce no re-classification or re-moderation
        if action.action_type == "classify" and s.classified:
            raise ValueError("Post has already been classified.")
        if action.action_type == "moderate" and s.moderated:
            raise ValueError("Post has already been moderated.")

        # Apply action
        s.step_count += 1

        if action.action_type == "classify":
            s.agent_label = action.label
            s.classified = True

        elif action.action_type == "moderate":
            s.agent_decision = action.decision
            s.moderated = True
            s.done = True  # Episode ends after moderation

        # Compute step reward
        reward = grade_step(
            action_type=action.action_type,
            agent_label=s.agent_label,
            agent_decision=s.agent_decision,
            ground_truth_label=s.ground_truth_label,
            ground_truth_decision=s.ground_truth_decision,
            step_count=s.step_count,
            is_done=s.done,
        )
        s.episode_reward += reward

        obs = self._build_observation()
        info = self._build_info(reward)

        return obs, reward, s.done, info

    def state(self) -> State:
        """Return the full internal state (useful for debugging and evaluation)."""
        if self._state is None:
            raise RuntimeError("Call reset() before state().")
        return self._state

    # ──────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────

    def _build_observation(self) -> Observation:
        s = self._state
        available = []
        if not s.classified:
            available.append("classify")
        if s.classified and not s.moderated:
            available.append("moderate")

        return Observation(
            post_id=s.post_id,
            content=s.content,
            user_history=s.user_history,
            flags=s.flags,
            platform_rules=s.platform_rules,
            available_actions=available,
            step_number=s.step_count,
            classified=s.classified,
            moderated=s.moderated,
        )

    def _build_info(self, reward: float) -> Dict[str, Any]:
        s = self._state
        info: Dict[str, Any] = {
            "step_count": s.step_count,
            "classified": s.classified,
            "moderated": s.moderated,
            "step_reward": reward,
            "cumulative_reward": round(s.episode_reward, 4),
        }
        if s.done:
            clf_score = grade_classification(s.agent_label, s.ground_truth_label)
            dec_score = grade_decision(s.agent_decision, s.ground_truth_decision)
            info.update({
                "episode_done": True,
                "ground_truth_label": s.ground_truth_label,
                "ground_truth_decision": s.ground_truth_decision,
                "agent_label": s.agent_label,
                "agent_decision": s.agent_decision,
                "classification_score": clf_score,
                "decision_score": dec_score,
                "final_reward": compute_episode_reward(clf_score, dec_score, s.step_count),
            })
        return info

