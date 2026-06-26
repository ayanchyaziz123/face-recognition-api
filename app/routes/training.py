import threading

from fastapi import APIRouter, HTTPException

from app.dependencies import recognizer, train_status
from trainer import train as run_train

router = APIRouter(tags=["Training"])


@router.post("/train")
def train():
    if train_status.get("state") == "training":
        return {"message": "Training already in progress.", "status": train_status}

    def _run():
        try:
            result = run_train(status=train_status)
            recognizer.load()
            train_status.update(result)
        except Exception as e:
            train_status["state"] = "error"
            train_status["error"] = str(e)

    train_status.clear()
    train_status["state"] = "starting"
    threading.Thread(target=_run, daemon=True).start()
    return {"message": "Training started in background.", "status": train_status}


@router.get("/train/status")
def get_train_status():
    return train_status


@router.get("/reload")
def reload():
    recognizer.load()
    if recognizer.model is None:
        raise HTTPException(503, "Model not found. Train first.")
    return {"message": "Model reloaded.", "classes": recognizer.classes}
