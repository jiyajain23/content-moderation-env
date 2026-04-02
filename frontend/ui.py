import streamlit as st
import requests
import json
import time
from openai import OpenAI
client = OpenAI(
    api_key=st.secrets.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

st.set_page_config(page_title="AI Moderation Agent", layout="wide")

st.title("🛡️ AI Content Moderation Simulator")
st.caption("Watch an AI agent analyze and moderate content in real-time")

# -----------------------------
# Session State
# -----------------------------
if "observation" not in st.session_state:
    st.session_state.observation = None
if "done" not in st.session_state:
    st.session_state.done = False
if "score" not in st.session_state:
    st.session_state.score = 0

# -----------------------------
# Helper Functions
# -----------------------------
def normalize_label(label):
    label = (label or "").lower()
    if "spam" in label:
        return "spam"
    elif "hate" in label or "abuse" in label or "threat" in label:
        return "abusive"
    else:
        return "safe"

def get_llm_action(obs):
    prompt = f"""
You are a strict content moderation agent.

RULES:
- If "classify" is available → you MUST classify
- If "moderate" is available → you MUST moderate
- NEVER do both

VALID OUTPUT:

For classify:
{{
  "action_type": "classify",
  "label": "spam | abusive | safe",
  "reasoning": "short explanation"
}}

For moderate:
{{
  "action_type": "moderate",
  "decision": "allow | warn | remove",
  "reasoning": "short explanation"
}}

Observation:
{json.dumps(obs, indent=2)}

Return ONLY JSON.
"""

    try:
        res = client.chat.completions.create(
            model=st.secrets.get("MODEL_NAME", "llama3-70b-8192"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        output = res.choices[0].message.content.strip()

        # 🔥 Handle ```json blocks
        if "```" in output:
            output = output.split("```")[1]
            if output.startswith("json"):
                output = output[4:]

        action = json.loads(output)

        # 🔧 Enforce valid action_type
        if action["action_type"] not in obs["available_actions"]:
            action["action_type"] = obs["available_actions"][0]

        # 🔧 Normalize label
        if action.get("label"):
            action["label"] = normalize_label(action["label"])

        # 🔧 Fix structure
        if action["action_type"] == "classify":
            action.pop("decision", None)
        else:
            action.pop("label", None)

        return action

    except Exception as e:
        st.error(f"LLM Error: {e}")
        return None

# -----------------------------
# Controls
# -----------------------------
st.subheader("⚙️ Controls")

col_ctrl1, col_ctrl2 = st.columns(2)

with col_ctrl1:
    task = st.selectbox(
        "Select Task",
        ["task_easy_001", "task_medium_001", "task_hard_001"]
    )

with col_ctrl2:
    if st.button("🔄 Reset Environment"):
        res = requests.post(f"{API_BASE}/reset", json={"task_id": task})
        data = res.json()

        if "observation" in data:
            st.session_state.observation = data["observation"]
            st.session_state.done = False
            st.session_state.score = 0
        else:
            st.error(data)

# -----------------------------
# Display Observation
# -----------------------------
if st.session_state.observation:
    obs = st.session_state.observation

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📄 Content")
        st.info(obs["content"])

        st.subheader("📜 Context")

        st.markdown("**👤 User History**")
        for h in obs["user_history"]:
            st.markdown(f"- {h}")

        st.markdown("**🚩 Flags**")
        for f in obs["flags"]:
            st.markdown(f"⚠️ {f}")

    with col2:
        st.subheader("🤖 Agent Status")

        st.metric("Step", obs["step_number"])
        st.metric("Available Actions", ", ".join(obs["available_actions"]))

        progress = min(obs["step_number"] / 5, 1.0)
        st.progress(progress)

# -----------------------------
# AI Agent Action
# -----------------------------
if st.session_state.observation and not st.session_state.done:
    obs = st.session_state.observation

    st.subheader("🤖 AI Agent")

    if st.button("⚡ Let AI Decide"):
        with st.spinner("AI is analyzing..."):
            time.sleep(1)
            action = get_llm_action(obs)

            if not action:
                st.stop()

        st.success("Decision Generated ✅")
        st.json(action)

        with st.expander("🧠 AI Reasoning"):
            st.write(action.get("reasoning", "No reasoning"))

        res = requests.post(f"{API_BASE}/step", json={"action": action})
        data = res.json()

        if "observation" in data:
            st.session_state.observation = data["observation"]
            st.session_state.done = data["done"]
            st.session_state.score += data["reward"]

            st.success(f"Reward: {data['reward']}")
        else:
            st.error(data)

# -----------------------------
# Manual Action
# -----------------------------
if st.session_state.observation and not st.session_state.done:
    obs = st.session_state.observation

    st.subheader("🎮 Manual Override")

    action_type = st.selectbox(
        "Action Type",
        obs["available_actions"]
    )

    label = None
    decision = None

    if action_type == "classify":
        label = st.selectbox("Label", ["spam", "abusive", "safe"])
    else:
        decision = st.selectbox("Decision", ["allow", "warn", "remove"])

    if st.button("🚀 Submit Manual Action"):
        payload = {
            "action": {
                "action_type": action_type,
                "label": label,
                "decision": decision
            }
        }

        res = requests.post(f"{API_BASE}/step", json=payload)
        data = res.json()

        if "observation" in data:
            st.session_state.observation = data["observation"]
            st.session_state.done = data["done"]
            st.session_state.score += data["reward"]

            st.success(f"Reward: {data['reward']}")
        else:
            st.error(data)

# -----------------------------
# Auto Run Agent
# -----------------------------
if st.session_state.observation and not st.session_state.done:
    if st.button("⚡ Auto Run Agent"):
        obs = st.session_state.observation

    for i in range(5):
        action = get_llm_action(obs)

        if not action:
            break

        res = requests.post(f"{API_BASE}/step", json={"action": action})
        data = res.json()

        if "observation" not in data:
            st.error(data)
            break

        obs = data["observation"]
        st.session_state.observation = obs
        st.session_state.score += data["reward"]

        st.write(f"Step {i+1}: {action}")
        time.sleep(1)

        if data["done"]:
            st.session_state.done = True
            break

# -----------------------------
# Score + Done
# -----------------------------
st.subheader("📊 Performance")

st.metric("Cumulative Score", round(st.session_state.score, 2))

if st.session_state.done:
    st.success("✅ Episode Finished!")
