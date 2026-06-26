import shutil
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.core.database import MemberQuery, members_table

DATASET_DIR = Path("dataset")
_IMAGE_EXTS  = {".jpg", ".jpeg", ".png"}


async def save_member_photos(org_id: str, name: str, photos: list[UploadFile]) -> dict:
    member_dir = DATASET_DIR / org_id / name
    member_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(member_dir.glob("*")))
    saved = 0
    for photo in photos:
        if not photo.content_type or not photo.content_type.startswith("image/"):
            continue
        ext  = (photo.filename or "img.jpg").rsplit(".", 1)[-1]
        path = member_dir / f"{existing + saved + 1:04d}.{ext}"
        path.write_bytes(await photo.read())
        saved += 1

    if not members_table.search((MemberQuery.org_id == org_id) & (MemberQuery.name == name)):
        members_table.insert({"id": str(uuid.uuid4()), "org_id": org_id, "name": name})

    total = existing + saved
    return {"message": f"Saved {saved} photo(s) for {name}.", "total_photos": total}


def list_members(org_id: str) -> dict:
    rows = members_table.search(MemberQuery.org_id == org_id)
    members = []
    for m in rows:
        member_dir  = DATASET_DIR / org_id / m["name"]
        photo_count = len([f for f in member_dir.iterdir() if f.suffix.lower() in _IMAGE_EXTS]) \
                      if member_dir.exists() else 0
        members.append({"name": m["name"], "photo_count": photo_count})
    return {"members": members}


def get_member_detail(org_id: str, name: str) -> dict:
    rows = members_table.search((MemberQuery.org_id == org_id) & (MemberQuery.name == name))
    if not rows:
        raise HTTPException(404, f"Member '{name}' not found.")
    member_dir = DATASET_DIR / org_id / name
    photos = sorted(
        f.name for f in member_dir.iterdir() if f.suffix.lower() in _IMAGE_EXTS
    ) if member_dir.exists() else []
    return {"name": name, "photo_count": len(photos), "photos": photos}


def delete_member_photo(org_id: str, name: str, filename: str) -> dict:
    path = DATASET_DIR / org_id / name / filename
    if not path.exists() or path.suffix.lower() not in _IMAGE_EXTS:
        raise HTTPException(404, "Photo not found.")
    path.unlink()
    return {"message": f"Deleted {filename}."}


def rename_member(org_id: str, old_name: str, new_name: str) -> dict:
    new_name = new_name.strip()
    if not new_name:
        raise HTTPException(400, "Name cannot be empty.")
    if old_name == new_name:
        return {"message": "No change."}
    if members_table.search((MemberQuery.org_id == org_id) & (MemberQuery.name == new_name)):
        raise HTTPException(400, f"A member named '{new_name}' already exists.")

    old_dir = DATASET_DIR / org_id / old_name
    new_dir = DATASET_DIR / org_id / new_name
    if old_dir.exists():
        old_dir.rename(new_dir)

    members_table.update(
        {"name": new_name},
        (MemberQuery.org_id == org_id) & (MemberQuery.name == old_name),
    )
    return {"message": f"Renamed '{old_name}' to '{new_name}'."}


def remove_member(org_id: str, name: str) -> dict:
    members_table.remove((MemberQuery.org_id == org_id) & (MemberQuery.name == name))
    member_dir = DATASET_DIR / org_id / name
    if member_dir.exists():
        shutil.rmtree(member_dir)
    return {"message": f"{name} removed."}
