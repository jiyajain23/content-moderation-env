import os
import requests
from openai import OpenAI

API_BASE = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(
    api_key=HF_TOKEN,
    base_url="https://api.groq.com/openai/v1"  # Groq-compatible
)

def get_action(obs):
    prompt = f"""
You are a content moderation agent.

Post:
{obs['content']}

User history:
{obs['user_history']}

Flags:
{obs['flags']}

Rules:
{obs['platform_rules']}

Return JSON:
{{
 "action_type": "classify or moderate",
 "label": "spam/abusive/safe",
 "decision": "allow/warn/remove",
 "reasoning": "short explanation"
}}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return eval(response.choices[0].message.content)


def run_task(task_id):
    res = requests.post(f"{API_BASE}/reset", json={"task_id": task_id})
    obs = res.json()["observation"]

    total_reward = 0

    for _ in range(10):
        action = get_action(obs)

        res = requests.post(f"{API_BASE}/step", json={"action": action})
        data = res.json()

        obs = data["observation"]
        total_reward += data["reward"]

        if data["done"]:
            break

    return total_reward


if __name__ == "__main__":
    tasks = ["task_easy_001", "task_medium_001", "task_hard_001"]

    for t in tasks:
        score = run_task(t)
        print(f"{t}: {score}")
