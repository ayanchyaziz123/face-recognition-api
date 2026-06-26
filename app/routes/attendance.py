import csv

from fastapi import APIRouter

from model import ATTENDANCE_CSV

router = APIRouter(tags=["Attendance"])


def read_records() -> list[dict]:
    if not ATTENDANCE_CSV.exists():
        return []
    with open(ATTENDANCE_CSV, newline="") as f:
        return list(csv.DictReader(f))


@router.get("/attendance")
def get_attendance():
    records = read_records()
    return {"total": len(records), "records": records}


@router.delete("/attendance")
def clear_attendance():
    if ATTENDANCE_CSV.exists():
        ATTENDANCE_CSV.unlink()
    return {"message": "Attendance log cleared."}
