from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from model import UNKNOWN_DIR
from app.routes import dashboard, recognition, attendance, training

UNKNOWN_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Smart Attendance System", version="1.0.0")

app.mount("/unknown", StaticFiles(directory="unknown"), name="unknown")

app.include_router(dashboard.router)
app.include_router(recognition.router)
app.include_router(attendance.router)
app.include_router(training.router)


@app.get("/health", tags=["System"])
def health():
    from app.dependencies import recognizer
    return {
        "status":       "ok",
        "model_loaded": recognizer.model is not None,
        "classes":      recognizer.classes,
    }
