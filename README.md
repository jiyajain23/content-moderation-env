---
title: Content Moderation Environment
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_file: server/app.py
pinned: false
---
# Context-Aware Content Moderation Environment

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-compatible-brightgreen)](https://openenv.dev)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com)

---
## 🚀 Live Environment

🔗 **API Base URL:**  
This environment is fully deployed on :[https://jiyajain23-content-moderation-env.hf.space/](https://jiyajain23-content-moderation-env.hf.space/) and can be accessed programmatically via REST API.
🔗 **Frontend:**  
[https://jiyajain23-content-moderation-env.hf.space](https://content-moderation-env-7q47km67y2sbbehtqpmvyc.streamlit.app/)
🔗 **Backend Hugging Face Space:** 
[https://huggingface.co/spaces/jiyajain23/content-moderation-env](https://huggingface.co/spaces/jiyajain23/content-moderation-env)


---

## 1. Environment Description

This OpenEnv environment simulates a **real-world social-media content moderation pipeline**.

An AI agent acts as a content moderator, receiving a user-generated post together with rich contextual signals. The agent must:

1. **Classify** the content (`spam`, `abusive`, or `safe`)
2. **Moderate** the content (`allow`, `warn`, or `remove`)

The environment rewards correct decisions, penalises dangerous mistakes (e.g., allowing harmful content), and shapes behaviour with bonuses for efficient decision-making.

---

## 2. Real-World Motivation

Content moderation at scale is one of the most consequential AI applications. Platforms process millions of posts daily, and errors have real costs:

| Error Type | Consequence |
|---|---|
| False negative (miss harmful content) | User harm, legal liability |
| False positive (remove safe content) | Censorship, user churn |
| Wrong moderation level | Under/over-enforcement |

A well-designed environment must capture the **nuance** of these trade-offs — not just "right or wrong," but the severity and direction of errors. This environment does exactly that.

---

## 3. Observation Space

Each observation is a JSON object with the following fields:

| Field | Type | Description |
|---|---|---|
| `post_id` | string | Unique post identifier |
| `content` | string | The text being evaluated |
| `user_history` | string[] | The author's previous posts |
| `flags` | string[] | Prior reports or automated flags |
| `platform_rules` | string[] | Active moderation policies |
| `available_actions` | string[] | Which actions are currently valid |
| `task_id` | string | The active task |
| `step` | integer | Current step within the episode |

---

## 4. Action Space

Actions are JSON objects:

```json
{
  "action_type": "classify" | "moderate",
  "label": "spam" | "abusive" | "safe",   // required for classify
  "decision": "allow" | "warn" | "remove" // required for moderate
}
```

### Action sequencing

```
reset()
  └─> step({ action_type: "classify", label: "..." })
        └─> step({ action_type: "moderate", decision: "..." })
              └─> episode done
```

The agent must classify before it can moderate.

---

## 5. Tasks

### Task 1 — Easy (`task_easy`)

**Post:** Obvious spam with known spam domains, excessive punctuation, and a history of promotional posts.

**Signals:** 8 spam reports, URL matches known spam domain list, repeated pattern.

**Ground truth:** label=`spam`, decision=`remove`

**Difficulty:** Deterministic — all signals point the same way.

---

### Task 2 — Medium (`task_medium`)

**Post:** Direct personal harassment with discriminatory language targeting the recipient's origin.

**Signals:** 3 harassment reports, automated hate-speech classifier confidence 0.91.

**Ground truth:** label=`abusive`, decision=`remove`

**Difficulty:** Requires correct moderation decision; user history is benign, so context doesn't help but doesn't confuse either.

---

### Task 3 — Hard (`task_hard`)

**Post:** Ambiguous political speech that could be read as incitement but also as frustrated civic advocacy.

**Signals:** 1 incitement report, automated classifier confidence only 0.52 (ambiguous).

**User history:** Peaceful protests, petitions, town-hall attendance — clear civic engagement pattern.

**Platform rules:** Rule 3 explicitly requires contextual review for borderline political content; Rule 4 protects protest organisation.

**Ground truth:** label=`safe`, decision=`warn`

**Difficulty:** Requires reasoning across user_history, flags, and platform_rules. Naive classification would likely mislabel as `abusive`.

---

## 6. Reward Design

### Formula

```
reward = 0.4 × classification_score
       + 0.6 × decision_score
       + early_bonus
       + step_penalty
```

### Classification score

| Predicted | Truth | Score |
|---|---|---|
| Correct | — | 1.0 |
| Incorrect | — | 0.0 |

### Decision score (structured)

| Predicted | Truth | Score | Rationale |
|---|---|---|---|
| allow | allow | 1.0 | Correct |
| warn | warn | 1.0 | Correct |
| remove | remove | 1.0 | Correct |
| warn | allow | 0.5 | Over-cautious but not harmful |
| remove | warn | 0.5 | Slightly too aggressive |
| warn | remove | 0.5 | Under-cautious |
| remove | allow | 0.0 | False positive removal |
| allow | warn | -0.5 | Too lenient |
| allow | remove | -1.0 | **Dangerous** — harmful content allowed |

### Shaping

- **Early decision bonus:** +0.1 if both classify and moderate complete within 2 steps.
- **Step penalty:** -0.05 per step beyond the first 2 (discourages looping).

### Maximum possible reward

```
0.4 × 1.0 + 0.6 × 1.0 + 0.1 = 1.1  (correct + fast)
```

---

## 7. Setup Instructions

### Local run

```bash
# Clone / download the project
cd project_root

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# Test health
curl http://localhost:7860/health

# Run baseline inference (requires OPENAI_API_KEY)
export OPENAI_API_KEY="sk-..."
export API_BASE_URL="http://localhost:7860"
export LLM_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
python inference.py
```

If you're using a Groq OpenAI-compatible endpoint, set:
`export LLM_BASE_URL="https://api.groq.com/openai/v1"`

### Docker run

```bash
# Build
docker build -t content-moderation-env .

# Run
docker run -p 7860:7860 content-moderation-env

# With API key for inference (optional)
docker run -p 7860:7860 \
  -e OPENAI_API_KEY="sk-..." \
  content-moderation-env
```

---

## 🤖 Running the Agent (inference.py)

This project includes a baseline LLM agent (`inference.py`) that interacts with the environment.

### Important: Update API URL
Before running, set:
```python
API_BASE_URL = "https://jiyajain23-content-moderation-env.hf.space"
```
## 8. Hugging Face Deployment

1. Create a new **Space** on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose **Docker** as the SDK
3. Set **Port** to `7860`
4. Push the repository:

```bash
git init
git remote add hf https://huggingface.co/spaces/<your-username>/<your-space-name>
git add .
git commit -m "Initial OpenEnv submission"
git push hf main
```

5. Set any required secrets (e.g., `OPENAI_API_KEY`) in Space Settings → Repository secrets.

The Space will automatically build the Docker image and expose the environment on `https://<your-username>-<space-name>.hf.space`.

---
## 🎨 Interactive Demo (Optional UI)

A simple UI can be built using :contentReference[oaicite:1]{index=1} to interact with the environment.

### Features

- Reset tasks interactively
- View content + context
- Choose actions manually
- Observe rewards in real-time

### Run locally

```bash
pip install gradio requests
python ui.py
```
## 9. API Reference
### Health Check`
Returns:
```json
{ "message": "Content Moderation Environment is running 🚀" }
```
### `POST /reset`

```json
{ "task_id": "task_easy" }
```

Returns initial `Observation`.

---

### `POST /step`

```json
{
  "action_type": "classify",
  "label": "spam",
  "decision": null
}
```

Returns `{ observation, reward, done, info }`.

---

### `GET /state`

Returns the current internal `State` object:

```json
{
  "task_id": "task_easy",
  "post_id": "post_001",
  "step": 1,
  "classified": true,
  "moderated": false,
  "predicted_label": "spam",
  "predicted_decision": null,
  "done": false,
  "total_reward": 0.0
}
```

---

## 10. Baseline Scores

Scores achieved by `gpt-4o-mini` at temperature=0:

| Task | Difficulty | Label ✓ | Decision ✓ | Reward |
|---|---|---|---|---|
| task_easy | Easy | ✓ | ✓ | **1.10** |
| task_medium | Medium | ✓ | ✓ | **1.10** |
| task_hard | Hard | ✗ | ✗ | **-0.65** |
| **Average** | | | | **0.52** |

> The hard task is intentionally challenging — models that rely on surface-level signals (without reading `user_history` and `platform_rules`) will misclassify it.

---
## 🧠 System Architecture

The system follows a full agent-environment loop:

## Project Structure
## 🧠 System Architecture

The system follows a full agent-environment loop:
```
project_root/
├── server/
│   ├── __init__.py
│   ├── app.py       # FastAPI endpoints
│   ├── env.py       # Core environment logic
│   ├── models.py    # Pydantic schemas
│   ├── tasks.py     # Task definitions + ground truth
│   └── graders.py   # Deterministic scoring functions
├── openenv.yaml     # OpenEnv configuration
├── Dockerfile
├── requirements.txt
├── inference.py     # Baseline LLM agent
└── README.md
```

---

## License

MIT
