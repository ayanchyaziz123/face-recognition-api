import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.auth import get_current_org
from app.database import AttQuery, attendance_table
from app.dependencies import get_recognizer

router = APIRouter(tags=["Recognition"])

_log_cooldown: dict[str, float] = {}
_unk_cooldown: dict[str, float] = {}
LOG_INTERVAL = 60.0
UNK_INTERVAL = 10.0


def _log_attendance(org_id: str, name: str, confidence: float):
    key = f"{org_id}:{name}"
    now = time.time()
    if now - _log_cooldown.get(key, 0) >= LOG_INTERVAL:
        _log_cooldown[key] = now
        attendance_table.insert({
            "org_id":     org_id,
            "name":       name,
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "confidence": round(confidence, 3),
        })


def _save_unknown(org_id: str, image_bytes: bytes, bbox: dict):
    now = time.time()
    if now - _unk_cooldown.get(org_id, 0) < UNK_INTERVAL:
        return
    _unk_cooldown[org_id] = now

    arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return

    x, y, w, h  = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    unknown_dir  = Path("unknown") / org_id
    unknown_dir.mkdir(parents=True, exist_ok=True)
    filename     = unknown_dir / f"unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(filename), bgr[y:y+h, x:x+w])


@router.post("/predict")
async def predict(
    file:   UploadFile = File(...),
    org_id: str        = Depends(get_current_org),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")

    image_bytes = await file.read()
    recognizer  = get_recognizer(org_id)

    try:
        result = recognizer.predict(image_bytes)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))

    for r in result["results"]:
        if r["name"] != "Unknown":
            _log_attendance(org_id, r["name"], r["confidence"])
        else:
            _save_unknown(org_id, image_bytes, r["bbox"])

    return result
