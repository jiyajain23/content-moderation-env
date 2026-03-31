"""
Pydantic models for the Context-Aware Content Moderation Environment.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class Observation(BaseModel):
    """What the agent receives at each step."""
    post_id: str
    content: str
    user_history: List[str] = Field(default_factory=list)
    flags: List[str] = Field(default_factory=list)
    platform_rules: List[str] = Field(default_factory=list)
    available_actions: List[str] = Field(default_factory=list)
    step_number: int = 0
    classified: bool = False
    moderated: bool = False


class Action(BaseModel):
    """Action submitted by the agent."""
    action_type: Literal["classify", "moderate"]
    label: Optional[Literal["spam", "abusive", "safe"]] = None
    decision: Optional[Literal["allow", "warn", "remove"]] = None
    reasoning: Optional[str] = None

    def validate_action(self) -> Optional[str]:
        if self.action_type == "classify" and self.label is None:
            return "label is required when action_type is 'classify'"
        if self.action_type == "moderate" and self.decision is None:
            return "decision is required when action_type is 'moderate'"
        return None


class State(BaseModel):
    """Full internal state of the environment."""
    task_id: str
    post_id: str
    content: str
    user_history: List[str]
    flags: List[str]
    platform_rules: List[str]
    ground_truth_label: Literal["spam", "abusive", "safe"]
    ground_truth_decision: Literal["allow", "warn", "remove"]
    agent_label: Optional[str] = None
    agent_decision: Optional[str] = None
    classified: bool = False
    moderated: bool = False
    step_count: int = 0
    done: bool = False
    episode_reward: float = 0.0


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict


class ResetResult(BaseModel):
    observation: Observation


class StateResult(BaseModel):
    state: State
