import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.auth import get_current_org
from app.database import MemberQuery, members_table

router      = APIRouter(prefix="/members", tags=["Members"])
DATASET_DIR = Path("dataset")


@router.post("/add", status_code=201)
async def add_member(
    name:   str,
    photos: list[UploadFile] = File(...),
    org_id: str = Depends(get_current_org),
):
    member_dir = DATASET_DIR / org_id / name
    member_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    existing_count = len(list(member_dir.glob("*")))
    for photo in photos:
        if not photo.content_type or not photo.content_type.startswith("image/"):
            continue
        ext  = (photo.filename or "img.jpg").rsplit(".", 1)[-1]
        path = member_dir / f"{existing_count + saved + 1:04d}.{ext}"
        path.write_bytes(await photo.read())
        saved += 1

    if not members_table.search((MemberQuery.org_id == org_id) & (MemberQuery.name == name)):
        members_table.insert({"id": str(uuid.uuid4()), "org_id": org_id, "name": name})

    return {"message": f"Saved {saved} photo(s) for {name}."}


@router.get("/")
def list_members(org_id: str = Depends(get_current_org)):
    members = members_table.search(MemberQuery.org_id == org_id)
    return {"members": [m["name"] for m in members]}


@router.delete("/{name}")
def delete_member(name: str, org_id: str = Depends(get_current_org)):
    members_table.remove((MemberQuery.org_id == org_id) & (MemberQuery.name == name))
    member_dir = DATASET_DIR / org_id / name
    if member_dir.exists():
        import shutil
        shutil.rmtree(member_dir)
    return {"message": f"{name} removed."}
