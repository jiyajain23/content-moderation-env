import os
import json
import requests
from openai import OpenAI
# -----------------------------
# Config (STRICT)
# -----------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY      = os.environ.get("API_KEY", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

TASK_NAME  = "content_moderation"
BENCHMARK  = "content_moderation_env"
MAX_STEPS  = 10

# -----------------------------
# Logging
# -----------------------------
def compute_score(rewards):
    if not rewards:
        return 0.5
    raw = sum(rewards) / len(rewards)
    return max(0.01, min(0.99, raw))

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True
    )

def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    score = compute_score(rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True
    )

# -----------------------------
# Client — created lazily inside run_task
# so a crash here never kills the process
# -----------------------------
def make_client():
    try:
        if not API_BASE_URL or not API_KEY:
            raise ValueError("API_BASE_URL or API_KEY is empty")
        return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)  # ✅ correct
    except Exception as e:
        print(f"[DEBUG] Client init failed: {e}", flush=True)
        return None

# -----------------------------
# Safe Helpers
# -----------------------------
def safe_post(url, payload):
    try:
        res = requests.post(url, json=payload, timeout=30)
        return res.json()
    except Exception as e:
        print(f"[DEBUG] Request error: {e}", flush=True)
        return {}

def safe_json_load(text, fallback):
    try:
        text = text.strip()
        # Strip markdown code fences if the model wraps output
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception:
        return fallback

def get_safe_action(available_actions):
    action_type = available_actions[0] if available_actions else "classify"
    return {
        "action_type": action_type,
        "label": "safe",
        "decision": "allow",
        "reasoning": "safe_fallback"
    }

# -----------------------------
# Normalize helpers
# -----------------------------
def normalize_label(label):
    label = (label or "").lower()
    if "spam" in label:
        return "spam"
    if "abuse" in label or "abusive" in label or "hate" in label or "threat" in label:
        return "abusive"
    return "safe"

def normalize_decision(decision):
    decision = (decision or "allow").lower()
    if "remove" in decision:
        return "remove"
    if "warn" in decision:
        return "warn"
    return "allow"

# -----------------------------
# LLM Action
# -----------------------------
def get_action(obs, client):
    available_actions = obs.get("available_actions", []) or []

    if client is None:
        return get_safe_action(available_actions)

    content        = obs.get("content", "")
    user_history   = obs.get("user_history", [])
    flags          = obs.get("flags", [])
    platform_rules = obs.get("platform_rules", [])
    is_classify    = "classify" in available_actions

    if is_classify:
        task_instruction = (
            'Your task: CLASSIFY this post.\n'
            'Return JSON: {"action_type": "classify", "label": "<spam|abusive|safe>", "reasoning": "<brief>"}'
        )
    else:
        task_instruction = (
            'Your task: MODERATE this post (already classified).\n'
            'Return JSON: {"action_type": "moderate", "decision": "<allow|warn|remove>", "reasoning": "<brief>"}'
        )

    prompt = f"""You are an expert content moderation agent for a social media platform.

POST CONTENT:
{content}

USER HISTORY:
{chr(10).join(f"- {h}" for h in user_history) if user_history else "- No prior history"}

COMMUNITY FLAGS:
{chr(10).join(f"- {f}" for f in flags) if flags else "- No flags"}

PLATFORM RULES:
{chr(10).join(f"- {r}" for r in platform_rules) if platform_rules else "- No rules provided"}

AVAILABLE ACTIONS: {available_actions}

{task_instruction}

Return ONLY valid JSON. No markdown, no extra text.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        text = response.choices[0].message.content or "{}"
        return safe_json_load(text, get_safe_action(available_actions))
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return get_safe_action(available_actions)

# -----------------------------
# Main Runner
# -----------------------------
def run_task(task_id, client):
    log_start(task_id, BENCHMARK, MODEL_NAME)

    rewards     = []
    steps_taken = 0
    success     = False

    try:
        # ── Reset ──────────────────────────────────────────────
        data = safe_post(f"{ENV_BASE_URL}/reset", {"task_id": task_id})

        if "observation" not in data:
            print(f"[DEBUG] Reset failed for {task_id}: {data}", flush=True)
            log_step(0, "reset_failed", 0.0, True, "reset_error")
            log_end(False, 0, [])
            return                          # only exits on genuine reset failure

        obs = data["observation"]           # reachable on success

        # ── Episode loop ───────────────────────────────────────
        for step in range(1, MAX_STEPS + 1):
            available = obs.get("available_actions", []) or []

            if not available:
                log_step(step, "no_actions", 0.0, True, "no_available_actions")
                break

            # Get action from LLM (client may be None → safe fallback)
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

            # ── Send step ──────────────────────────────────────
            step_data = safe_post(f"{ENV_BASE_URL}/step", {"action": action})

            reward = float(step_data.get("reward", 0.0))
            done   = bool(step_data.get("done", False))
            error  = None if "observation" in step_data else "step_error"

            rewards.append(reward)
            steps_taken = step

            log_step(step, json.dumps(action), reward, done, error)

            if "observation" not in step_data:
                break

            obs = step_data["observation"]

            if done:
                success = True
                break

    except Exception as e:
        log_step(steps_taken + 1, "exception", 0.0, True, str(e))

    finally:
        log_end(success, steps_taken, rewards)

# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    try:
        client = make_client()          # safe — never crashes the process
        tasks = ["task_easy_001", "task_medium_001", "task_hard_001"]
        for t in tasks:
            run_task(t, client)
    except Exception as e:
        print(f"[FATAL] Unhandled exception: {e}", flush=True)
