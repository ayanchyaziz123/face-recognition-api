from fastapi import APIRouter, File, HTTPException, UploadFile

from app.dependencies import recognizer

router = APIRouter(tags=["Recognition"])


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")
    try:
        return recognizer.predict(await file.read())
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
