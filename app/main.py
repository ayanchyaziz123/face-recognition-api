from fastapi import FastAPI, Depends
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.auth import get_current_org
from app.dependencies import get_recognizer
from app.routes import auth, members, recognition, attendance, training, dashboard

Path("unknown").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

app = FastAPI(title="Smart Attendance System", version="2.0.0")

app.mount("/unknown", StaticFiles(directory="unknown"), name="unknown")

app.include_router(auth.router)
app.include_router(members.router)
app.include_router(recognition.router)
app.include_router(attendance.router)
app.include_router(training.router)
app.include_router(dashboard.router)


@app.get("/health", tags=["System"])
def health():
    return {"status": "ok"}
