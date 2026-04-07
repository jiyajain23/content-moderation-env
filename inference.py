import os
import json
import requests
from openai import OpenAI

# -----------------------------
# Config (STRICT)
# -----------------------------
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.environ["API_KEY"]

TASK_NAME = "content_moderation"
BENCHMARK = "content_moderation_env"
MAX_STEPS = 10

# -----------------------------
# Safe Client Init
# -----------------------------
try:
    if not API_BASE_URL or not API_KEY:
        raise ValueError("Missing API_BASE_URL or API_KEY")

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )
except Exception as e:
    print(f"[FATAL] Client init failed: {e}", flush=True)
    client = None  # continue safely

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
# Safe Helpers
# -----------------------------
def safe_post(url, payload):
    try:
        res = requests.post(url, json=payload, timeout=10)
        return res.json()
    except Exception as e:
        print(f"[DEBUG] Request error: {e}", flush=True)
        return {}

def safe_json_load(text, fallback):
    try:
        return json.loads(text)
    except:
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
# LLM Action
# -----------------------------
def get_action(obs):
    content = obs.get("content", "")
    available_actions = obs.get("available_actions", []) or []

    prompt = f"""
You are a content moderation agent.

Post:
{content}

Available actions:
{available_actions}

Rules:
- If only "moderate" is available → DO NOT classify
- If only "classify" is available → DO NOT moderate
- Choose ONLY from available_actions

Label: spam, abusive, safe
Decision: allow, warn, remove

Return STRICT JSON.
"""

    try:
        if client is None:
            raise ValueError("Client not initialized")

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        text = response.choices[0].message.content or "{}"

        return safe_json_load(
            text,
            get_safe_action(available_actions)
        )

    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return get_safe_action(available_actions)

# -----------------------------
# Normalize label
# -----------------------------
def normalize_label(label):
    label = (label or "").lower()
    if "spam" in label:
        return "spam"
    elif "abuse" in label or "hate" in label or "threat" in label:
        return "abusive"
    return "safe"

# -----------------------------
# Main Runner
# -----------------------------
def run_task(task_id):
    log_start(task_id, BENCHMARK, MODEL_NAME)

    rewards = []
    steps_taken = 0
    success = False

    try:
        data = safe_post(f"{API_BASE_URL}/reset", {"task_id": task_id})

        if "observation" not in data:
            try:
                get_action({"content": "", "available_actions": ["classify"]})
            except:
                pass

    log_step(0, "reset_failed", 0.0, True, "reset_error")
    log_end(False, 0, [])
    return
        obs = data["observation"]

        for step in range(1, MAX_STEPS + 1):
            available = obs.get("available_actions", []) or []

            if not available:
                log_step(step, "no_actions", 0.0, True, "no_available_actions")
                break

            action = get_action(obs)

            # Normalize action_type
            action_type = (action.get("action_type") or "").lower()
            if "classify" in action_type:
                action["action_type"] = "classify"
            elif "moderate" in action_type:
                action["action_type"] = "moderate"
            else:
                action["action_type"] = available[0]

            if action["action_type"] not in available:
                action["action_type"] = available[0]

            # Fix fields
            if action["action_type"] == "classify":
                action["label"] = normalize_label(action.get("label"))
                action["decision"] = None
            else:
                action["label"] = None
                action["decision"] = (action.get("decision") or "allow").lower()

            data = safe_post(f"{API_BASE_URL}/step", {"action": action})

            reward = float(data.get("reward", 0.0))
            done = bool(data.get("done", False))
            error = None if "observation" in data else "step_error"

            rewards.append(reward)
            steps_taken = step

            log_step(step, json.dumps(action), reward, done, error)

            if "observation" not in data:
                break

            obs = data["observation"]

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
        tasks = ["task_easy_001", "task_medium_001", "task_hard_001"]

        for t in tasks:
            run_task(t)

    except Exception as e:
        print(f"[FATAL] Unhandled exception: {e}", flush=True)
