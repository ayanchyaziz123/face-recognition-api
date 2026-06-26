from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.controllers import (
    attendance_controller,
    auth_controller,
    dashboard_controller,
    member_controller,
    recognition_controller,
    training_controller,
)

Path("unknown").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)
Path("spoofs").mkdir(exist_ok=True)

app = FastAPI(title="Smart Attendance System", version="3.0.0")

app.mount("/unknown", StaticFiles(directory="unknown"), name="unknown")

app.include_router(auth_controller.router)
app.include_router(member_controller.router)
app.include_router(recognition_controller.router)
app.include_router(attendance_controller.router)
app.include_router(training_controller.router)
app.include_router(dashboard_controller.router)


@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "version": "3.0.0"}
