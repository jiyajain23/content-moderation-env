import streamlit as st
import requests
import json

API_BASE = st.secrets.get("API_BASE_URL", "http://localhost:7860")

st.set_page_config(page_title="Content Moderation Env", layout="wide")

st.title("🤖 Content Moderation Environment")
st.markdown("Interact with the moderation environment like an AI agent.")

# -----------------------------
# Session State
# -----------------------------
if "observation" not in st.session_state:
    st.session_state.observation = None
if "done" not in st.session_state:
    st.session_state.done = False

# -----------------------------
# Reset Environment
# -----------------------------
st.sidebar.header("⚙️ Controls")

task = st.sidebar.selectbox(
    "Select Task",
    ["task_easy", "task_medium", "task_hard"]
)

if st.sidebar.button("🔄 Reset Environment"):
    res = requests.post(f"{API_BASE}/reset", json={"task_id": task})
    st.session_state.observation = res.json()["observation"]
    st.session_state.done = False

# -----------------------------
# Display Observation
# -----------------------------
if st.session_state.observation:
    obs = st.session_state.observation or {}

    st.subheader("📥 Observation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📝 Content")
        st.info(obs["content"])

        st.markdown("### 🚩 Flags")
        st.write(obs["flags"])

        st.markdown("### 📜 Rules")
        st.write(obs["platform_rules"])

    with col2:
        st.markdown("### 👤 User History")
        st.write(obs["user_history"])

        st.markdown("### ⚡ Available Actions")
        st.write(obs["available_actions"])

# -----------------------------
# Actions
# -----------------------------
if st.session_state.observation and not st.session_state.done:

    st.subheader("🎯 Take Action")

    action_type = st.selectbox("Action Type", ["classify", "moderate"])

    label = None
    decision = None

    if action_type == "classify":
        label = st.selectbox("Label", ["spam", "abusive", "safe"])
    else:
        decision = st.selectbox("Decision", ["allow", "warn", "remove"])

    if st.button("🚀 Submit Action"):
       payload = {
           "action": {
               "action_type": action_type,
               "label": label,
               "decision": decision,
               "reasoning": reasoning
           }
        }

    res = requests.post(f"{API_BASE}/step", json=payload)
    data = res.json()

    st.session_state.observation = data["observation"]
    st.session_state.done = data["done"]

    st.success(f"Reward: {data['reward']}")
    st.json(data["info"])

# -----------------------------
# Done
# -----------------------------
if st.session_state.done:
    st.success("✅ Episode Finished!")
