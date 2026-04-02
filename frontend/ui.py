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

client = OpenAI(
    api_key=st.secrets.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

st.set_page_config(page_title="AI Moderation Agent", layout="wide", initial_sidebar_state="expanded")

# 🛠️ FIXED CSS: Works in both Dark/Light mode and fixed the crash
st.markdown("""
    <style>
    [data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.1); 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .stMarkdown, .stCaption, h1, h2, h3, h4 {
        color: inherit !important;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# Session State
# -----------------------------
if "observation" not in st.session_state:
    st.session_state.observation = None
if "done" not in st.session_state:
    st.session_state.done = False
if "score" not in st.session_state:
    st.session_state.score = 0
if "last_reasoning" not in st.session_state:
    st.session_state.last_reasoning = ""

# -----------------------------
# Helper Functions (Variables Unchanged)
# -----------------------------
def normalize_label(label):
    label = (label or "").lower()
    if "spam" in label: return "spam"
    if any(word in label for word in ["hate", "abuse", "threat", "abusive"]): return "abusive"
    return "safe"

def clean_json_output(output):
    match = re.search(r'\{.*\}', output, re.DOTALL)
    return match.group(0) if match else output

def get_llm_action(obs):
    prompt = f"""You are a content moderation agent. Analyze context and rules.
Return ONLY valid JSON.
Observation: {json.dumps(obs, indent=2)}
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
        if action["action_type"] not in obs["available_actions"]:
            action["action_type"] = obs["available_actions"][0]
        if action.get("label"):
            action["label"] = normalize_label(action["label"])
        return action
    except Exception as e:
        st.error(f"LLM Error: {e}")
        return None

# -----------------------------
# Sidebar: Stats & Original Reset
# -----------------------------
with st.sidebar:
    st.title("🛡️ Settings")
    task = st.selectbox("Scenario Difficulty", ["task_easy_001", "task_medium_001", "task_hard_001"])
    
    if st.button("🔄 Reset Environment", use_container_width=True, type="primary"):
        res = requests.post(f"{API_BASE}/reset", json={"task_id": task})
        data = res.json()
        if "observation" in data:
            st.session_state.observation = data["observation"]
            st.session_state.done = False
            st.session_state.score = 0
            st.session_state.last_reasoning = ""
            st.toast("New Task Loaded!")

    st.divider()
    st.metric("Total Reward", f"{st.session_state.score:.2f}")
    if st.session_state.done:
        st.success("Target Reached ✅")

# -----------------------------
# Main UI Logic
# -----------------------------
st.title("🛡️ AI Content Moderation Simulator")

# NEW FEATURE: Add Own Content
with st.expander("📝 Add Your Own Custom Post"):
    custom_text = st.text_area("Type content here:", placeholder="e.g. You guys are idiots!")
    if st.button("📥 Load into Simulator"):
        st.session_state.observation = {
            "post_id": "manual_input",
            "content": custom_text,
            "user_history": ["No previous data"],
            "flags": ["User-defined content"],
            "platform_rules": ["Standard spam and harassment rules."],
            "available_actions": ["classify"],
            "step_number": 0,
            "classified": False,
            "moderated": False
        }
        st.session_state.done = False
        st.session_state.score = 0
        st.rerun()

if st.session_state.observation:
    obs = st.session_state.observation
    
    # Header Info
    m1, m2, m3 = st.columns(3)
    m1.metric("Post ID", obs["post_id"])
    m2.metric("Step", obs["step_number"])
    m3.metric("State", "Classifying" if not obs["classified"] else "Moderating")

    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.info(f"**Reported Content:**\n\n{obs['content']}")
        
        # AI THOUGHT PROCESS - Shows what the AI is thinking
        if st.session_state.last_reasoning:
            with st.chat_message("assistant", avatar="🧠"):
                st.write("**AI Reasoning Process:**")
                st.write(st.session_state.last_reasoning)

        st.subheader("🕵️ Investigation Context")
        t1, t2, t3 = st.tabs(["👤 History", "🚩 Flags", "⚖️ Rules"])
        with t1: [st.write(f"• {h}") for h in obs["user_history"]]
        with t2: [st.warning(f) for f in obs["flags"]]
        with t3: [st.caption(r) for r in obs["platform_rules"]]

    with col_r:
        st.subheader("⚡ Actions")
        
        # AI Run
        if not st.session_state.done:
            if st.button("🤖 Let AI Decide", use_container_width=True):
                with st.spinner("Analyzing..."):
                    action = get_llm_action(obs)
                    if action:
                        st.session_state.last_reasoning = action.get("reasoning", "No reasoning provided.")
                        res = requests.post(f"{API_BASE}/step", json={"action": action})
                        data = res.json()
                        st.session_state.observation = data["observation"]
                        st.session_state.done = data["done"]
                        st.session_state.score += data.get("reward", 0)
                        st.rerun()

        # Manual Action Override
        with st.expander("🛠️ Manual Override"):
            act_type = st.radio("Step", obs["available_actions"], horizontal=True)
            lbl = st.selectbox("Label", ["safe", "spam", "abusive"]) if act_type == "classify" else None
            dec = st.selectbox("Decision", ["allow", "warn", "remove"]) if act_type == "moderate" else None
            
            if st.button("🚀 Submit Step", use_container_width=True):
                payload = {"action": {"action_type": act_type, "label": lbl, "decision": dec}}
                res = requests.post(f"{API_BASE}/step", json=payload)
                data = res.json()
                st.session_state.observation = data["observation"]
                st.session_state.done = data["done"]
                st.session_state.score += data.get("reward", 0)
                st.rerun()
