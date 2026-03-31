"""
Baseline inference script for the Content Moderation Environment.

Uses the OpenAI Python client (compatible with any OpenAI-spec endpoint)
to run all 3 tasks and report per-task scores.

Environment variables:
    OPENAI_API_KEY  – API key (required)
    API_BASE_URL    – Base URL of the environment server (default: http://localhost:7860)
    LLM_BASE_URL    – OpenAI-compatible LLM base URL (default: https://api.openai.com/v1)
    MODEL_NAME      – LLM model name (default: gpt-4o)
"""




import os
import json
import sys
import requests
from openai import OpenAI
from typing import Optional, Tuple

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
API_BASE_URL = "https://jiyajain23-content-moderation-env.hf.space"  # Environment server base URL
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")  # OpenAI-compatible LLM base URL
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    print("[ERROR] OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=LLM_BASE_URL,
)

TASK_IDS = ["task_easy_001", "task_medium_001", "task_hard_001"]

SYSTEM_PROMPT = """
You are a strict content moderation system.

Follow rules exactly:
- spam → remove
- abusive → warn or remove depending on severity
- safe → allow

Respond ONLY in JSON.
Be precise and deterministic.
"""

ACTION_SCHEMA_HINT = {
    "action_type": "classify | moderate",
    "label": "spam | abusive | safe (required when action_type=classify)",
    "decision": "allow | warn | remove (required when action_type=moderate)",
    "reasoning": "optional string",
}


def build_user_prompt(obs: dict, step: int) -> str:
    lines = [
        f"POST ID: {obs['post_id']}",
        f"CONTENT:\n{obs['content']}",
        "",
        "USER HISTORY:",
        *[f"  - {h}" for h in obs.get("user_history", [])],
        "",
        "COMMUNITY FLAGS:",
        *[f"  - {f}" for f in obs.get("flags", [])],
        "",
        "PLATFORM RULES:",
        *[f"  - {r}" for r in obs.get("platform_rules", [])],
        "",
        f"AVAILABLE ACTIONS: {obs.get('available_actions', [])}",
        f"STEP: {step}",
        "",
    ]
    if not obs.get("classified"):
        lines.append("Task: Classify this post. Respond with action_type='classify'.")
    else:
        lines.append("Task: You have classified the post. Now decide on moderation. Respond with action_type='moderate'.")
    return "\n".join(lines)


def validate_action_dict(action: dict, obs: dict) -> Tuple[bool, Optional[str]]:
    action_type = action.get("action_type")
    if action_type not in ("classify", "moderate"):
        return False, "action_type must be 'classify' or 'moderate'"

    available = obs.get("available_actions") or []
    if available and action_type not in available:
        return False, f"action_type must be one of available_actions={available}"

    if action_type == "classify":
        label = action.get("label")
        if label not in ("spam", "abusive", "safe"):
            return False, "label must be one of: spam, abusive, safe (required for classify)"
        return True, None

    decision = action.get("decision")
    if decision not in ("allow", "warn", "remove"):
        return False, "decision must be one of: allow, warn, remove (required for moderate)"
    return True, None


def coerce_action_defaults(action: dict, obs: dict) -> dict:
    """
    Last-resort coercion to avoid aborting the run if the model returns incomplete JSON.
    Defaults are conservative: classify->safe, moderate->warn.
    """
    action_type = action.get("action_type")
    if action_type not in ("classify", "moderate"):
        available = obs.get("available_actions") or []
        action_type = available[0] if available else "classify"
        action["action_type"] = action_type

    if action_type == "classify" and action.get("label") not in ("spam", "abusive", "safe"):
        action["label"] = "safe"
    if action_type == "moderate" and action.get("decision") not in ("allow", "warn", "remove"):
        action["decision"] = "warn"
    return action

def call_llm(messages: list) -> dict:
    """Call the LLM and parse JSON response."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,  # deterministic
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    import re

    def safe_parse(content):
        try:
            return json.loads(content)
        except:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise

    return safe_parse(content)


def run_task(task_id: str) -> dict:
    """Run a full episode for a task. Returns score info."""
    print(f"\n{'='*60}")
    print(f"  TASK: {task_id}")
    print(f"{'='*60}")

    # Reset environment
    reset_resp = requests.post(
        f"{API_BASE_URL}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
    reset_resp.raise_for_status()
    obs = reset_resp.json()["observation"]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    step_num = 0
    total_reward = 0.0
    final_info = {}

    while True:
        step_num += 1
        user_msg = build_user_prompt(obs, step_num)
        schema_hint = (
            "Return ONLY a JSON object with keys matching this schema:\n"
            f"{json.dumps(ACTION_SCHEMA_HINT, indent=2)}\n"
            "Do not return null for required fields.\n"
        )
        messages.append({"role": "user", "content": f"{user_msg}\n\n{schema_hint}"})

        action_dict = call_llm(messages)
        ok, err = validate_action_dict(action_dict, obs)
        if not ok:
            repair_prompt = (
                "Your previous JSON action was INVALID for this environment.\n"
                f"Error: {err}\n"
                f"available_actions={obs.get('available_actions')}\n\n"
                "Return ONLY a corrected JSON action (no markdown, no extra text)."
            )
            messages.append({"role": "user", "content": repair_prompt})
            action_dict = call_llm(messages)
            ok2, err2 = validate_action_dict(action_dict, obs)
            if not ok2:
                print(f"[WARN] Invalid LLM action after repair: {err2}. Coercing defaults.", file=sys.stderr)
                action_dict = coerce_action_defaults(action_dict, obs)
        print(f"  Step {step_num}: {action_dict.get('action_type')} -> "
              f"label={action_dict.get('label')} decision={action_dict.get('decision')} | "
              f"reasoning={str(action_dict.get('reasoning', ''))[:80]}...")

        messages.append({"role": "assistant", "content": json.dumps(action_dict)})

        # Remove 'reasoning' key before sending to server (server doesn't require it but accepts it)
        step_resp = requests.post(
            f"{API_BASE_URL}/step",
            json={"action": action_dict},
            timeout=30,
        )
        step_resp.raise_for_status()
        result = step_resp.json()

        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]
        info = result["info"]
        total_reward += reward

        print(f"  Reward: {reward:.4f} | Cumulative: {total_reward:.4f} | Done: {done}")

        if done:
            final_info = info
            break

    print(f"\n  Final scores for {task_id}:")
    print(f"    Classification score : {final_info.get('classification_score', 'N/A')}")
    print(f"    Decision score       : {final_info.get('decision_score', 'N/A')}")
    print(f"    Final reward         : {final_info.get('final_reward', 'N/A')}")
    print(f"    Agent label          : {final_info.get('agent_label')} "
          f"(GT: {final_info.get('ground_truth_label')})")
    print(f"    Agent decision       : {final_info.get('agent_decision')} "
          f"(GT: {final_info.get('ground_truth_decision')})")

    return {
        "task_id": task_id,
        "classification_score": final_info.get("classification_score", 0.0),
        "decision_score": final_info.get("decision_score", 0.0),
        "final_reward": final_info.get("final_reward", 0.0),
        "steps": step_num,
    }


def main():
    print("Context-Aware Content Moderation Environment – Baseline Inference")
    if "/openai/" in API_BASE_URL or "groq" in API_BASE_URL or "openai" in API_BASE_URL:
        print(
            "[WARN] API_BASE_URL looks like an LLM endpoint. "
            "Use API_BASE_URL for the local environment server, and set LLM_BASE_URL for the LLM.",
            file=sys.stderr,
        )
    print(f"Env Server : {API_BASE_URL}")
    print(f"LLM API     : {LLM_BASE_URL}")
    print(f"Model  : {MODEL_NAME}")

    results = []
    for task_id in TASK_IDS:
        try:
            result = run_task(task_id)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Task {task_id} failed: {e}", file=sys.stderr)
            results.append({"task_id": task_id, "final_reward": 0.0, "error": str(e)})

    # Summary table
    print(f"\n{'='*60}")
    print("  FINAL SCORES SUMMARY")
    print(f"{'='*60}")
    print(f"{'Task ID':<25} {'Classification':>15} {'Decision':>10} {'Reward':>10} {'Steps':>6}")
    print("-" * 70)
    total = 0.0
    for r in results:
        clf = r.get("classification_score", "ERR")
        dec = r.get("decision_score", "ERR")
        rew = r.get("final_reward", 0.0)
        steps = r.get("steps", "-")
        print(f"{r['task_id']:<25} {str(clf):>15} {str(dec):>10} {str(rew):>10} {str(steps):>6}")
        if isinstance(rew, float):
            total += rew
    print("-" * 70)
    avg = total / len(results) if results else 0.0
    print(f"{'AVERAGE'::<25} {'':>15} {'':>10} {avg:>10.4f}")
    print()


if __name__ == "__main__":
    main()
