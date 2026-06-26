from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.core.auth import get_current_org
from app.services import member_service, training_service

router = APIRouter(prefix="/members", tags=["Members"])
DATASET_DIR = Path("dataset")
_IMAGE_EXTS  = {".jpg", ".jpeg", ".png"}


class RenameBody(BaseModel):
    new_name: str


@router.post("/add", status_code=201)
async def add_member(
    name:   str,
    photos: list[UploadFile] = File(...),
    org_id: str = Depends(get_current_org),
):
    result = await member_service.save_member_photos(org_id, name, photos)
    training_service.start_training(org_id)
    result["training"] = "Model retraining started in the background."
    return result


@router.get("/")
def get_members(org_id: str = Depends(get_current_org)):
    return member_service.list_members(org_id)


# ── Per-member detail ──────────────────────────────────────────────────────────

@router.get("/{name}")
def get_member(name: str, org_id: str = Depends(get_current_org)):
    return member_service.get_member_detail(org_id, name)


@router.get("/{name}/photo")
def get_member_dp(name: str, org_id: str = Depends(get_current_org)):
    """Returns the first photo for use as an avatar."""
    member_dir = DATASET_DIR / org_id / name
    if member_dir.exists():
        for f in sorted(member_dir.iterdir()):
            if f.suffix.lower() in _IMAGE_EXTS:
                return FileResponse(str(f), media_type="image/jpeg")
    raise HTTPException(404, "No photo found.")


@router.get("/{name}/photos/{filename}")
def get_member_photo_file(name: str, filename: str, org_id: str = Depends(get_current_org)):
    path = DATASET_DIR / org_id / name / filename
    if not path.exists() or path.suffix.lower() not in _IMAGE_EXTS:
        raise HTTPException(404, "Photo not found.")
    return FileResponse(str(path), media_type="image/jpeg")


@router.delete("/{name}/photos/{filename}")
def delete_member_photo(name: str, filename: str, org_id: str = Depends(get_current_org)):
    return member_service.delete_member_photo(org_id, name, filename)


@router.put("/{name}/rename")
def rename_member(name: str, body: RenameBody, org_id: str = Depends(get_current_org)):
    return member_service.rename_member(org_id, name, body.new_name)


@router.delete("/{name}")
def delete_member(name: str, org_id: str = Depends(get_current_org)):
    return member_service.remove_member(org_id, name)
