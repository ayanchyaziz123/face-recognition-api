import csv
import threading
from collections import Counter
from datetime import date
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from model import ATTENDANCE_CSV, UNKNOWN_DIR, Recognizer
from trainer import train as run_train

app       = FastAPI(title="Smart Attendance System")
templates = Jinja2Templates(directory="templates")
recognizer = Recognizer()

app.mount("/unknown", StaticFiles(directory="unknown"), name="unknown")

_train_status: dict = {"state": "idle"}


def _read_records() -> list[dict]:
    if not ATTENDANCE_CSV.exists():
        return []
    with open(ATTENDANCE_CSV, newline="") as f:
        return list(csv.DictReader(f))


def _build_stats(records: list[dict]) -> dict:
    today_str = date.today().isoformat()
    today     = [r for r in records if r["timestamp"].startswith(today_str)]
    counts    = Counter(r["name"] for r in today)
    most_seen = counts.most_common(1)[0][0] if counts else None

    all_counts = Counter(r["name"] for r in records)
    return {
        "today":         len(today),
        "total":         len(records),
        "unknown_count": len(list(UNKNOWN_DIR.glob("*.jpg"))),
        "most_seen":     most_seen,
        "chart": {
            "labels": list(all_counts.keys()),
            "values": list(all_counts.values()),
        },
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    records = _read_records()
    stats   = _build_stats(records)
    return templates.TemplateResponse("dashboard.html", {
        "request":        request,
        "stats":          stats,
        "chart_data":     stats["chart"],
        "recent":         list(reversed(records))[:10],
        "records":        list(reversed(records)),
        "unknown_images": sorted([f.name for f in UNKNOWN_DIR.glob("*.jpg")], reverse=True),
    })


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": recognizer.model is not None,
        "classes":      recognizer.classes,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")
    try:
        return recognizer.predict(await file.read())
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/train")
def train():
    if _train_status.get("state") == "training":
        return {"message": "Training already in progress.", "status": _train_status}

    def _run():
        try:
            result = run_train(status=_train_status)
            recognizer.load()
            _train_status.update(result)
        except Exception as e:
            _train_status["state"] = "error"
            _train_status["error"] = str(e)

    _train_status.clear()
    _train_status["state"] = "starting"
    threading.Thread(target=_run, daemon=True).start()
    return {"message": "Training started in background.", "status": _train_status}


@app.get("/train/status")
def train_status():
    return _train_status


@app.get("/reload")
def reload():
    recognizer.load()
    if recognizer.model is None:
        raise HTTPException(503, "Model not found. Train first.")
    return {"message": "Model reloaded.", "classes": recognizer.classes}


@app.get("/attendance")
def get_attendance():
    records = _read_records()
    return {"total": len(records), "records": records}


@app.delete("/attendance")
def clear_attendance():
    if ATTENDANCE_CSV.exists():
        ATTENDANCE_CSV.unlink()
    return {"message": "Attendance log cleared."}
