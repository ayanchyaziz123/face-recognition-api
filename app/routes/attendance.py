from fastapi import APIRouter, Depends

from app.auth import get_current_org
from app.database import AttQuery, attendance_table

router = APIRouter(tags=["Attendance"])


def read_records(org_id: str) -> list[dict]:
    return attendance_table.search(AttQuery.org_id == org_id)


@router.get("/attendance")
def get_attendance(org_id: str = Depends(get_current_org)):
    records = read_records(org_id)
    return {"total": len(records), "records": sorted(records, key=lambda r: r["timestamp"], reverse=True)}


@router.delete("/attendance")
def clear_attendance(org_id: str = Depends(get_current_org)):
    attendance_table.remove(AttQuery.org_id == org_id)
    return {"message": "Attendance log cleared."}
