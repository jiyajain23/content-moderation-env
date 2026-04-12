"""FastAPI server exposing the Content Moderation Environment via HTTP."""

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

try:
    from .models import Action, ResetResult, StepResult, StateResult
    from .env import ModerationEnv
    from .tasks import TASKS
except ImportError:
    from server.models import Action, ResetResult, StepResult, StateResult
    from server.env import ModerationEnv
    from server.tasks import TASKS

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------
class ResetRequest(BaseModel):
    task_id: Optional[str] = None

class StepRequest(BaseModel):
    action: Action

# ---------------------------------------------------------------------------
# Environment (module-level singleton)
# ---------------------------------------------------------------------------
env = ModerationEnv()

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def home():
    return {
        "message": "Content Moderation Environment is running",
        "docs":    "/docs",
        "health":  "/health",
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "task_id":     t["task_id"],
                "difficulty":  t["difficulty"],
                "description": t["description"],
            }
            for t in TASKS
        ]
    }

@app.post("/reset", response_model=ResetResult)
def reset(request: ResetRequest = ResetRequest()):
    """Reset the environment. Optionally supply a task_id."""
    task_id = request.task_id
    try:
        obs = env.reset(task_id=task_id)
        return ResetResult(observation=obs)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"Task not found: {e}")
    except Exception as e:
        log.exception("Reset failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step", response_model=StepResult)
def step(request: StepRequest):
    log.info("Incoming action: %s", request.action)

    validation_error = request.action.validate_action()
    if validation_error:
        log.warning("Validation error: %s", validation_error)
        raise HTTPException(status_code=422, detail=validation_error)

    try:
        obs, reward, done, info = env.step(request.action)
        log.info("Step result: reward=%.2f done=%s", reward, done)
        return StepResult(observation=obs, reward=reward, done=done, info=info)
    except Exception as e:
        log.exception("Step failed")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state", response_model=StateResult)
def state():
    try:
        s = env.state()
        return StateResult(state=s)
    except Exception as e:
        log.exception("State fetch failed")
        raise HTTPException(status_code=400, detail=str(e))

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
