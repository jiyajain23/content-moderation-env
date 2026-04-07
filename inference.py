import os
import json
import requests
from openai import OpenAI

# -----------------------------
# Config (Submission-compliant)
# -----------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3-70b-8192")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TASK_NAME = "content_moderation"
BENCHMARK = "content_moderation_env"
MAX_STEPS = 10


def create_client():
    if not HF_TOKEN:
        print("[DEBUG] Missing HF_TOKEN, running in fallback mode", flush=True)
        return None

    try:
        return OpenAI(
            api_key=HF_TOKEN,
            base_url="https://api.groq.com/openai/v1"
        )
    except Exception as e:
        print(f"[DEBUG] Client init failed: {e}", flush=True)
        return None


client = create_client()

# -----------------------------
# Logging (STRICT FORMAT)
# -----------------------------
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
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True
    )

# -----------------------------
# LLM Action
# -----------------------------

def get_action(obs):
    content = obs.get("content", "")
    available_actions = obs.get("available_actions", []) or []

    # Fallback if no client (HF_TOKEN missing / init failed)
    if client is None:
        action_type = available_actions[0] if available_actions else "classify"
        return {
            "action_type": action_type,
            "label": "safe" if action_type == "classify" else None,
            "decision": "allow" if action_type == "moderate" else None,
            "reasoning": "fallback_no_client"
        }

    prompt = f"""
You are a content moderation agent.

Post:
{content}

Available actions:
{available_actions}

Rules:
- If only "moderate" is available → you MUST NOT classify
- If only "classify" is available → you MUST NOT moderate
- Choose ONLY from available_actions

Label must be one of: "spam", "abusive", "safe"
Decision must be one of: "allow", "warn", "remove"

Return STRICT JSON:
{{
  "action_type": "...",
  "label": "...",
  "decision": "...",
  "reasoning": "..."
}}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        content = response.choices[0].message.content or "{}"
        return json.loads(content)

    except Exception:
        action_type = available_actions[0] if available_actions else "classify"
        return {
            "action_type": action_type,
            "label": "safe" if action_type == "classify" else None,
            "decision": "allow" if action_type == "moderate" else None,
            "reasoning": "fallback_exception"
        }
# -----------------------------
# Helpers
# -----------------------------
def normalize_label(label):
    label = (label or "").lower()
    if "spam" in label:
        return "spam"
    elif "hate" in label or "abuse" in label or "threat" in label:
        return "abusive"
    return "safe"

# -----------------------------
# Main Task Runner
# -----------------------------
def run_task(task_id):
    log_start(task_id, BENCHMARK, MODEL_NAME)

    rewards = []
    steps_taken = 0
    success = False

    try:
        res = requests.post(f"{API_BASE_URL}/reset", json={"task_id": task_id})
        data = res.json()

        if "observation" not in data:
            log_step(0, "reset_failed", 0.0, True, "reset_error")
            log_end(False, 0, [])
            return

        obs = data["observation"]

        for step in range(1, MAX_STEPS + 1):
            available = obs.get("available_actions", []) or []

            if not available:
                log_step(step, "no_available_actions", 0.0, True, "no_actions")
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

            # Step API
            res = requests.post(f"{API_BASE_URL}/step", json={"action": action})
            data = res.json()

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
    tasks = ["task_easy_001", "task_medium_001", "task_hard_001"]

    for t in tasks:
        run_task(t)
