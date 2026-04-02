import streamlit as st
import requests
import json
import time
import re
from openai import OpenAI
import os

# -----------------------------
# Configuration & Secrets
# -----------------------------
try:
    API_BASE = st.secrets["API_BASE_URL"]
except:
    API_BASE = os.getenv("API_BASE_URL", "http://localhost:7860")

# Initialize Groq Client
client = OpenAI(
    api_key=st.secrets.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

st.set_page_config(page_title="AI Moderation Agent", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .status-box { padding: 20px; border-radius: 10px; margin-bottom: 20px; }
    </style>
    """, unsafe_content_allowed=True)

st.title("🛡️ AI Content Moderation Simulator")
st.caption("Context-Aware RL Environment powered by Groq & OpenEnv")

# -----------------------------
# Session State
# -----------------------------
if "observation" not in st.session_state:
    st.session_state.observation = None
if "done" not in st.session_state:
    st.session_state.done = False
if "score" not in st.session_state:
    st.session_state.score = 0
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# Helper Functions
# -----------------------------
def normalize_label(label):
    label = (label or "").lower()
    if "spam" in label: return "spam"
    if any(word in label for word in ["hate", "abuse", "threat", "abusive"]): return "abusive"
    return "safe"

def clean_json_output(output):
    """Extracts JSON from LLM string even if it contains conversational filler."""
    match = re.search(r'\{.*\}', output, re.DOTALL)
    return match.group(0) if match else output

def get_llm_action(obs):
    prompt = f"""
You are a content moderation agent. Analyze context and rules.
RULES:
- Available Actions: {obs['available_actions']}
- You MUST choose ONE action from the list.
- Return ONLY valid JSON.

Format for 'classify': {{"action_type": "classify", "label": "spam|abusive|safe", "reasoning": "..."}}
Format for 'moderate': {{"action_type": "moderate", "decision": "allow|warn|remove", "reasoning": "..."}}

Observation:
{json.dumps(obs, indent=2)}
"""
    try:
        res = client.chat.completions.create(
            model=st.secrets.get("MODEL_NAME", "llama3-70b-8192"),
            messages=[{"role": "system", "content": "You are a JSON-only response bot."},
                      {"role": "user", "content": prompt}],
            temperature=0
        )
        raw_content = res.choices[0].message.content.strip()
        action = json.loads(clean_json_output(raw_content))
        
        # Validation & Normalization
        if action["action_type"] not in obs["available_actions"]:
            action["action_type"] = obs["available_actions"][0]
        if action.get("label"):
            action["label"] = normalize_label(action["label"])
        return action
    except Exception as e:
        st.error(f"LLM Reasoning Error: {e}")
        return None

# -----------------------------
# Sidebar Controls
# -----------------------------
with st.sidebar:
    st.header("⚙️ Environment Settings")
    task = st.selectbox("Scenario Difficulty", ["task_easy_001", "task_medium_001", "task_hard_001"])
    
    if st.button("🔄 Reset & Load New Task", use_container_width=True, type="primary"):
        res = requests.post(f"{API_BASE}/reset", json={"task_id": task})
        data = res.json()
        if "observation" in data:
            st.session_state.observation = data["observation"]
            st.session_state.done = False
            st.session_state.score = 0
            st.session_state.history = []
            st.toast("Environment Reset!", icon="🔄")
        else:
            st.error("Connection Failed")

    st.divider()
    st.subheader("📊 Session Statistics")
    st.metric("Total Reward", f"{st.session_state.score:.2f}")
    if st.session_state.done:
        st.success("Target Reached")

# -----------------------------
# Main Dashboard
# -----------------------------
if st.session_state.observation:
    obs = st.session_state.observation
    
    # Header Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Post ID", obs["post_id"])
    m2.metric("Current Step", obs["step_number"])
    status_label = "🟡 Classifying" if not obs["classified"] else "🟠 Moderating"
    m3.metric("State", status_label)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        with st.container(border=True):
            st.subheader("📄 Reported Content")
            st.markdown(f"#### \"{obs['content']}\"")
            
        st.subheader("🕵️ Investigation Context")
        tab1, tab2, tab3 = st.tabs(["👤 User History", "🚩 System Flags", "⚖️ Platform Rules"])
        
        with tab1:
            for h in obs["user_history"]:
                st.write(f"• {h}")
        with tab2:
            for f in obs["flags"]:
                st.warning(f)
        with tab3:
            for r in obs["platform_rules"]:
                st.caption(f"Rule: {r}")

    with col_right:
        st.subheader("🎮 Actions")
        
        # AI AGENT BOX
        with st.expander("🤖 AI Agent Control", expanded=not st.session_state.done):
            if st.button("⚡ Run AI Decision", use_container_width=True):
                with st.spinner("Llama-3 analyzing..."):
                    action = get_llm_action(obs)
                    if action:
                        res = requests.post(f"{API_BASE}/step", json={"action": action})
                        data = res.json()
                        st.session_state.observation = data["observation"]
                        st.session_state.done = data["done"]
                        st.session_state.score += data.get("reward", 0)
                        st.rerun()

        # MANUAL OVERRIDE BOX
        with st.expander("🛠️ Manual Override"):
            action_type = st.radio("Step", obs["available_actions"], horizontal=True)
            if action_type == "classify":
                label = st.selectbox("Set Label", ["safe", "spam", "abusive"])
                decision = None
            else:
                label = None
                decision = st.selectbox("Set Action", ["allow", "warn", "remove"])
            
            if st.button("🚀 Submit Step", use_container_width=True):
                payload = {"action": {"action_type": action_type, "label": label, "decision": decision}}
                res = requests.post(f"{API_BASE}/step", json=payload)
                data = res.json()
                st.session_state.observation = data["observation"]
                st.session_state.done = data["done"]
                st.session_state.score += data.get("reward", 0)
                st.rerun()

else:
    st.info("Select a task from the sidebar and click Reset to begin.")
