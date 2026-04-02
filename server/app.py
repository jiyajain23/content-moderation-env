"""FastAPI server exposing the Content Moderation Environment via HTTP."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware

# 1. IMPORT MODELS FIRST
# We use the 'server.models' path to ensure Docker recognizes the package4
try:
    from .models import Action, ResetResult, StepResult, StateResult
    from .env import ModerationEnv
except ImportError:
    from server.models import Action, ResetResult, StepResult, StateResult
    from server.env import ModerationEnv

app = FastAPI(
    title="Context-Aware Content Moderation Environment",
    description="An OpenEnv-compatible environment for social media moderation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. DEFINE REQUEST SCHEMAS
# These must come AFTER 'Action' is imported but BEFORE the endpoints
class ResetRequest(BaseModel):
    task_id: Optional[str] = None

class StepRequest(BaseModel):
    action: Action  # This works now because Action is imported at the top

# 3. INITIALIZE ENVIRONMENT
env = ModerationEnv()

# 4. ENDPOINTS
@app.get("/")
def home():
    return {
        "message": "Content Moderation Environment is running 🚀",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/tasks")
def list_tasks():
    from .tasks import TASKS
    return {
        "tasks": [
            {
                "task_id": t["task_id"],
                "difficulty": t["difficulty"],
                "description": t["description"],
            }
            for t in TASKS
        ]
    }

@app.post("/reset", response_model=ResetResult)
def reset(request: Optional[ResetRequest] = None):
    task_id = request.task_id if request else None
    try:
        obs = env.reset(task_id=task_id)
        return ResetResult(observation=obs)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/step", response_model=StepResult)
def step(request: StepRequest):
    print("👉 Incoming action:", request.action)

    validation_error = request.action.validate_action()
    if validation_error:
        print("❌ Validation error:", validation_error)
        raise HTTPException(status_code=422, detail=validation_error)
        
    try:
        obs, reward, done, info = env.step(request.action)
        print("✅ Step success:", obs, reward, done)
        return StepResult(observation=obs, reward=reward, done=done, info=info)
    except Exception as e:
        print("🔥 STEP ERROR:", str(e))   # 👈 IMPORTANT
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state", response_model=StateResult)
def state():
    try:
        s = env.state()
        return StateResult(state=s)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)
if __name__ == "__main__":
    main()
