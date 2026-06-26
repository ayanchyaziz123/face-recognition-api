import time

import cv2
import numpy as np

# ── Texture thresholds (tuned for webcam JPEG streams) ────────────────────────
TEXTURE_THRESHOLD  = 50.0
GRADIENT_THRESHOLD = 8.0
STDDEV_THRESHOLD   = 20.0

# ── Blink tracking ────────────────────────────────────────────────────────────
BLINKS_NEEDED = 2
BLINK_WINDOW  = 12.0   # seconds — blinks must happen within this window

_eye_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
)

# Per-org blink state: {org_id: {"was_open": bool, "blinks": [timestamps]}}
_blink_state: dict[str, dict] = {}


# ── Public API ─────────────────────────────────────────────────────────────────

def check_liveness(face_bgr: np.ndarray, org_id: str = "") -> dict:
    """
    Combined liveness check:
      1. Texture analysis   — catches printed photos / screens
      2. Eye / blink detect — catches video replay attacks
    Returns a dict merged from both checks.
    """
    texture = _texture_check(face_bgr)
    blink   = _blink_check(face_bgr, org_id) if org_id else {"blink_verified": False, "blink_count": 0, "eyes_open": True}

    is_live = texture["texture_ok"] and blink["blink_verified"]

    return {
        "is_live":          is_live,
        "texture_ok":       texture["texture_ok"],
        "texture_variance": texture["texture_variance"],
        "gradient_mean":    texture["gradient_mean"],
        "pixel_stddev":     texture["pixel_stddev"],
        "blink_verified":   blink["blink_verified"],
        "blink_count":      blink["blink_count"],
        "eyes_open":        blink["eyes_open"],
    }


def reset_blink_state(org_id: str) -> None:
    _blink_state.pop(org_id, None)


# ── Internal ───────────────────────────────────────────────────────────────────

def _texture_check(face_bgr: np.ndarray) -> dict:
    if face_bgr is None or face_bgr.size == 0:
        return {"texture_ok": False, "texture_variance": 0.0, "gradient_mean": 0.0, "pixel_stddev": 0.0}

    gray      = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    lap_var   = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sobelx    = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely    = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mean = float(np.sqrt(sobelx**2 + sobely**2).mean())
    stddev    = float(gray.std())

    passes = [
        lap_var   > TEXTURE_THRESHOLD,
        grad_mean > GRADIENT_THRESHOLD,
        stddev    > STDDEV_THRESHOLD,
    ]
    return {
        "texture_ok":       sum(passes) >= 2,
        "texture_variance": round(lap_var,   2),
        "gradient_mean":    round(grad_mean, 2),
        "pixel_stddev":     round(stddev,    2),
    }


def _eyes_open(face_bgr: np.ndarray) -> bool:
    """True when at least one eye is visible (i.e. not mid-blink)."""
    if face_bgr is None or face_bgr.size == 0:
        return True
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    # Only look at the upper half of the face — avoids mouth/nose false positives
    h = gray.shape[0]
    roi = gray[: h // 2, :]
    eyes = _eye_detector.detectMultiScale(
        roi, scaleFactor=1.1, minNeighbors=4, minSize=(15, 15)
    )
    return len(eyes) >= 1


def _blink_check(face_bgr: np.ndarray, org_id: str) -> dict:
    open_now = _eyes_open(face_bgr)
    state    = _blink_state.setdefault(org_id, {"was_open": True, "blinks": []})
    now      = time.time()

    # Falling edge (open → closed) registers one blink
    if state["was_open"] and not open_now:
        state["blinks"].append(now)

    state["was_open"] = open_now

    # Prune blinks outside the window
    state["blinks"] = [t for t in state["blinks"] if now - t < BLINK_WINDOW]
    count = len(state["blinks"])

    return {
        "blink_count":    count,
        "blink_verified": count >= BLINKS_NEEDED,
        "eyes_open":      open_now,
    }
