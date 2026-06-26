import threading

from fastapi import APIRouter, Depends, HTTPException

from app.auth import get_current_org
from app.dependencies import reload_recognizer, train_status
from trainer import train as run_train

router = APIRouter(tags=["Training"])


@router.post("/train")
def train(org_id: str = Depends(get_current_org)):
    if train_status.get("state") == "training":
        return {"message": "Training already in progress.", "status": train_status}

    def _run():
        try:
            result = run_train(org_id=org_id, status=train_status)
            reload_recognizer(org_id)
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
def reload(org_id: str = Depends(get_current_org)):
    recognizer = reload_recognizer(org_id)
    if recognizer.model is None:
        raise HTTPException(503, "Model not found. Add members and train first.")
    return {"message": "Model reloaded.", "classes": recognizer.classes}
