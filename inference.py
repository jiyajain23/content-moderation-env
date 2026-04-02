import os
from openai import OpenAI
import requests
# -----------------------------
# Config (robust)
# -----------------------------
try:
    import streamlit as st
    API_BASE = st.secrets.get("API_BASE_URL")
    MODEL_NAME = st.secrets.get("MODEL_NAME", "llama3-70b-8192")
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
except:
    API_BASE = os.getenv("API_BASE_URL")
    MODEL_NAME = os.getenv("MODEL_NAME", "llama3-70b-8192")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 🔥 Fail fast if missing
st.write("GROQ KEY EXISTS:", "GROQ_API_KEY" in st.secrets)
st.write("KEY VALUE:", st.secrets.get("GROQ_API_KEY"))

# -----------------------------
# Client
# -----------------------------
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)
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

    import json
    return json.loads(response.choices[0].message.content)
def normalize_label(label):
    label = label.lower()

    if "spam" in label:
        return "spam"
    elif "hate" in label or "abuse" in label or "threat" in label:
        return "abusive"
    else:
        return "safe"

def run_task(task_id):
    res = requests.post(f"{API_BASE}/reset", json={"task_id": task_id})
    obs = res.json()["observation"]

    total_reward = 0

    for _ in range(10):
        available = obs.get("available_actions", obs.get("available", [])) or []
        if not isinstance(available, list):
            available = []
        if not available:
            print("❌ No available actions in observation:", obs)
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

        res = requests.post(f"{API_BASE}/step", json={"action": action})
        data = res.json()
        print("STEP RESPONSE:", data)
        if "observation" not in data:
            print("❌ Step failed:", data)
            break
        total_reward += data["reward"]
        obs = data["observation"]

        if data["done"]:
            break
        
    return total_reward


if __name__ == "__main__":
    tasks = ["task_easy_001", "task_medium_001", "task_hard_001"]

    for t in tasks:
        score = run_task(t)
        print(f"{t}: {score}")


