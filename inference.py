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

# -----------------------------
# Client
# -----------------------------
client = OpenAI(
    api_key=HF_TOKEN,
    base_url="https://api.groq.com/openai/v1"
)

# -----------------------------
# LLM Action
# -----------------------------
def get_action(obs):
    prompt = f"""
You are a content moderation agent.

Post:
{obs['content']}

Available actions:
{obs['available_actions']}

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

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return json.loads(response.choices[0].message.content)

# -----------------------------
# Helpers
# -----------------------------
def normalize_label(label):
    label = label.lower()
    if "spam" in label:
        return "spam"
    elif "hate" in label or "abuse" in label or "threat" in label:
        return "abusive"
    return "safe"

# -----------------------------
# Main Task Runner
# -----------------------------
def run_task(task_id):
    print(f"[START] Task={task_id}")

    res = requests.post(f"{API_BASE_URL}/reset", json={"task_id": task_id})
    obs = res.json()["observation"]

    total_reward = 0

    for step in range(10):
        available = obs.get("available_actions", []) or []
        if not available:
            print(f"[STEP] No available actions → stopping")
            break

        action = get_action(obs)

        action_type = (action.get("action_type") or "").lower()
        if "classify" in action_type:
            action["action_type"] = "classify"
        elif "moderate" in action_type:
            action["action_type"] = "moderate"
        else:
            action["action_type"] = available[0]

        if action["action_type"] not in available:
            action["action_type"] = available[0]

        if action["action_type"] == "classify":
            action["label"] = normalize_label(action.get("label", "safe"))
            action["decision"] = None
        else:
            action["label"] = None
            action["decision"] = (action.get("decision") or "allow").lower()

        res = requests.post(f"{API_BASE_URL}/step", json={"action": action})
        data = res.json()

        print(f"[STEP] step={step} action={action} reward={data.get('reward')}")

        if "observation" not in data:
            print(f"[STEP] Error response: {data}")
            break

        total_reward += data["reward"]
        obs = data["observation"]

        if data.get("done"):
            break

    print(f"[END] Task={task_id} TotalReward={total_reward}")
    return total_reward

# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    tasks = ["task_easy_001", "task_medium_001", "task_hard_001"]

    for t in tasks:
        run_task(t)
