from fastapi import APIRouter, Depends, File, UploadFile

from app.core.auth import get_current_org
from app.services import recognition_service

router = APIRouter(tags=["Recognition"])


@router.post("/predict")
async def predict(
    file:   UploadFile = File(...),
    org_id: str        = Depends(get_current_org),
):
    return await recognition_service.run_prediction(org_id, file)


@router.get("/unknown-list")
def unknown_list(org_id: str = Depends(get_current_org)):
    return recognition_service.list_unknown_snapshots(org_id)
