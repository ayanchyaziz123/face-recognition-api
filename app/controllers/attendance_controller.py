from fastapi import APIRouter, Depends

from app.core.auth import get_current_org
from app.services import attendance_service

router = APIRouter(tags=["Attendance"])


@router.get("/attendance")
def get_attendance(org_id: str = Depends(get_current_org)):
    return attendance_service.get_attendance(org_id)


@router.delete("/attendance")
def clear_attendance(org_id: str = Depends(get_current_org)):
    return attendance_service.clear_attendance(org_id)
