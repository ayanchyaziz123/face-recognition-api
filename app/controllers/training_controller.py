from fastapi import APIRouter, Depends

from app.core.auth import get_current_org
from app.services import training_service

router = APIRouter(tags=["Training"])


@router.post("/train")
def train(org_id: str = Depends(get_current_org)):
    return training_service.start_training(org_id)


@router.get("/train/status")
def train_status():
    return training_service.get_train_status()


@router.get("/reload")
def reload(org_id: str = Depends(get_current_org)):
    return training_service.reload_model(org_id)
