"""
Content Moderation Agent — inference.py
----------------------------------------
Environment variables (all required):
    API_BASE_URL   LiteLLM proxy base URL  e.g. https://proxy.example.com/v1
    API_KEY        LiteLLM proxy API key
    ENV_BASE_URL   Moderation env server   e.g. http://localhost:7860
    MODEL_NAME     Model alias on proxy    default: llama3-70b-8192
"""

import json
import logging
import os
import sys

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "").rstrip("/")   # LiteLLM proxy → LLM calls ONLY
API_KEY      = os.getenv("API_KEY", "")                    # LiteLLM proxy key
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860").rstrip("/")  # FastAPI env server
MODEL_NAME   = os.getenv("MODEL_NAME", "llama3-70b-8192")

BENCHMARK = "content_moderation_env"
MAX_STEPS = 10
TASK_IDS  = ["task_easy_001", "task_medium_001", "task_hard_001"]

# ---------------------------------------------------------------------------
# OpenAI client — reads env vars at call time so validator-injected values are seen
# ---------------------------------------------------------------------------
def create_client() -> OpenAI:
    base_url = os.getenv("API_BASE_URL", "").rstrip("/")
    api_key  = os.getenv("API_KEY", "")
    # Provide a dummy key if missing so OpenAI() doesn't throw at construction —
    # a real auth error will surface naturally on the first LLM call instead.
    client = OpenAI(
        base_url=base_url or None,
        api_key=api_key or "dummy",
    )
    log.info("LLM client ready  → base_url=%s  model=%s", base_url or "(default)", MODEL_NAME)
    log.info("Env server        → %s", ENV_BASE_URL)
    return client

# ---------------------------------------------------------------------------
# Structured logging (format kept compatible with validator expectations)
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    log.info("[START] task=%s env=%s model=%s", task, env, model)

def log_step(step: int, action: str, reward: float, done: bool, error) -> None:
    log.info(
        "[STEP] step=%d action=%s reward=%.2f done=%s error=%s",
        step, action, reward, str(done).lower(), error if error else "null",
    )

def log_end(success: bool, steps: int, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    score = max(0.01, min(0.99, sum(rewards) / len(rewards))) if rewards else 0.5
    log.info(
        "[END] success=%s steps=%d score=%.3f rewards=%s",
        str(success).lower(), steps, score, rewards_str,
    )

# ---------------------------------------------------------------------------
# HTTP helpers — all env server calls go through here
# ---------------------------------------------------------------------------
def env_post(endpoint: str, payload: dict) -> dict:
    """POST to the moderation environment server (ENV_BASE_URL). Raises on failure."""
    url = f"{ENV_BASE_URL}{endpoint}"
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"POST {url} failed: {exc}") from exc

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------
def normalize_label(label: str | None) -> str:
    label = (label or "").lower()
    if "spam" in label:
        return "spam"
    if any(w in label for w in ("hate", "abuse", "abusive", "threat")):
        return "abusive"
    return "safe"

def normalize_decision(decision: str | None) -> str:
    decision = (decision or "allow").lower()
    if "remove" in decision:
        return "remove"
    if "warn" in decision:
        return "warn"
    return "allow"

def fallback_action(available_actions: list, reasoning: str = "fallback") -> dict:
    action_type = available_actions[0] if available_actions else "classify"
    return {
        "action_type": action_type,
        "label":       "safe"  if action_type == "classify" else None,
        "decision":    "allow" if action_type == "moderate" else None,
        "reasoning":   reasoning,
    }

def safe_json_load(text: str, fallback: dict) -> dict:
    try:
        text = text.strip()
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return fallback

# ---------------------------------------------------------------------------
# LLM action — raises on failure so the caller can decide what to do
# ---------------------------------------------------------------------------
def get_action(obs: dict, client: OpenAI) -> dict:
    available_actions = obs.get("available_actions") or []
    content           = obs.get("content", "")
    user_history      = obs.get("user_history") or []
    flags             = obs.get("flags") or []
    platform_rules    = obs.get("platform_rules") or []

    is_classify = "classify" in available_actions

    if is_classify:
        task_instruction = (
            "Your task: CLASSIFY this post.\n"
            'Return JSON: {"action_type": "classify", "label": "<spam|abusive|safe>", "reasoning": "<brief>"}'
        )
    else:
        task_instruction = (
            "Your task: MODERATE this post (already classified).\n"
            'Return JSON: {"action_type": "moderate", "decision": "<allow|warn|remove>", "reasoning": "<brief>"}'
        )

    history_lines = "\n".join(f"- {h}" for h in user_history) if user_history else "- No prior history"
    flag_lines    = "\n".join(f"- {f}" for f in flags)         if flags          else "- No flags"
    rule_lines    = "\n".join(f"- {r}" for r in platform_rules) if platform_rules else "- No rules provided"

    prompt = (
        "You are an expert content moderation agent for a social media platform.\n\n"
        f"POST CONTENT:\n{content}\n\n"
        f"USER HISTORY:\n{history_lines}\n\n"
        f"COMMUNITY FLAGS:\n{flag_lines}\n\n"
        f"PLATFORM RULES:\n{rule_lines}\n\n"
        f"AVAILABLE ACTIONS: {available_actions}\n\n"
        f"{task_instruction}\n\n"
        "Return ONLY valid JSON. No markdown, no extra text."
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = response.choices[0].message.content or "{}"
    return safe_json_load(text, fallback_action(available_actions, "json_parse_fallback"))

# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------
def run_task(task_id: str, client: OpenAI) -> None:
    log_start(task_id, BENCHMARK, MODEL_NAME)

    rewards:     list  = []
    steps_taken: int   = 0
    success:     bool  = False

    try:
        # ── Reset ──────────────────────────────────────────────────────────
        data = env_post("/reset", {"task_id": task_id})
        if "observation" not in data:
            raise RuntimeError(f"Reset returned no observation: {data}")
        obs = data["observation"]

        # ── Episode loop ───────────────────────────────────────────────────
        for step in range(1, MAX_STEPS + 1):
            available = obs.get("available_actions") or []
            if not available:
                log_step(step, "no_actions", 0.0, True, "no_available_actions")
                break

            # LLM call — raises on any API error
            action = get_action(obs, client)

            # Normalise action_type
            action_type = (action.get("action_type") or "").lower()
            if "classify" in action_type:
                action["action_type"] = "classify"
            elif "moderate" in action_type:
                action["action_type"] = "moderate"
            else:
                action["action_type"] = available[0]

            if action["action_type"] not in available:
                action["action_type"] = available[0]

            # Normalise label / decision
            if action["action_type"] == "classify":
                action["label"]    = normalize_label(action.get("label"))
                action["decision"] = None
            else:
                action["label"]    = None
                action["decision"] = normalize_decision(action.get("decision"))

            # ── Step ───────────────────────────────────────────────────────
            step_data = env_post("/step", {"action": action})
            reward    = float(step_data.get("reward", 0.0))
            done      = bool(step_data.get("done", False))
            error     = None if "observation" in step_data else "step_error"

            rewards.append(reward)
            steps_taken = step
            log_step(step, json.dumps(action), reward, done, error)

            if "observation" not in step_data:
                break

            obs = step_data["observation"]

            if done:
                success = True
                break

    except Exception as exc:
        log.error("[ERROR] task=%s  %s", task_id, exc, exc_info=True)
        log_step(steps_taken + 1, "exception", 0.0, True, str(exc))

    finally:
        log_end(success, steps_taken, rewards)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    llm_client = create_client()  # hard-fails if env vars are missing or wrong
    for task_id in TASK_IDS:
        run_task(task_id, llm_client)
