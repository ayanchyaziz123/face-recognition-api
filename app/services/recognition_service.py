import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

from app.core.database import AttQuery, attendance_table
from app.core.dependencies import get_recognizer
from app.services.liveness_service import check_liveness

_log_cooldown:   dict[str, float] = {}
_unk_cooldown:   dict[str, float] = {}
_spoof_cooldown: dict[str, float] = {}
LOG_INTERVAL   = 60.0
UNK_INTERVAL   = 10.0
SPOOF_INTERVAL = 10.0


def log_attendance(org_id: str, name: str, confidence: float) -> None:
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


def _save_snapshot(dest_dir: Path, bgr: np.ndarray, bbox: dict, prefix: str) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = dest_dir / f"{prefix}_{ts}.jpg"
    cv2.imwrite(str(path), bgr[y:y+h, x:x+w])


def save_unknown_snapshot(org_id: str, image_bytes: bytes, bbox: dict) -> None:
    now = time.time()
    if now - _unk_cooldown.get(org_id, 0) < UNK_INTERVAL:
        return
    _unk_cooldown[org_id] = now
    arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is not None:
        _save_snapshot(Path("unknown") / org_id, bgr, bbox, "unknown")


def save_spoof_snapshot(org_id: str, image_bytes: bytes, bbox: dict) -> None:
    now = time.time()
    if now - _spoof_cooldown.get(org_id, 0) < SPOOF_INTERVAL:
        return
    _spoof_cooldown[org_id] = now
    arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is not None:
        _save_snapshot(Path("spoofs") / org_id, bgr, bbox, "spoof")


async def run_prediction(org_id: str, file: UploadFile) -> dict:
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

    arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    for r in result["results"]:
        bbox = r["bbox"]

        # Run liveness on the face crop
        liveness = {"is_live": True, "blink_verified": True, "blink_count": 0, "eyes_open": True}
        if bgr is not None:
            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
            face_roi   = bgr[y:y+h, x:x+w]
            liveness   = check_liveness(face_roi, org_id)

        r["liveness"] = liveness

        if not liveness["is_live"]:
            r["name"]     = "Spoof Detected"
            r["greeting"] = "Please use your real face."
            save_spoof_snapshot(org_id, image_bytes, bbox)
            continue

        if r["name"] != "Unknown":
            log_attendance(org_id, r["name"], r["confidence"])
        else:
            save_unknown_snapshot(org_id, image_bytes, bbox)

    return result


def list_unknown_snapshots(org_id: str) -> dict:
    unknown_dir = Path("unknown") / org_id
    if not unknown_dir.exists():
        return {"images": []}
    images = sorted(
        f"{org_id}/{p.name}"
        for p in unknown_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    return {"images": images}
