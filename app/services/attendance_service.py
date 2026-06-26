from app.core.database import AttQuery, attendance_table


def get_attendance(org_id: str) -> dict:
    records = attendance_table.search(AttQuery.org_id == org_id)
    sorted_records = sorted(records, key=lambda r: r["timestamp"], reverse=True)
    return {"total": len(records), "records": sorted_records}


def clear_attendance(org_id: str) -> dict:
    attendance_table.remove(AttQuery.org_id == org_id)
    return {"message": "Attendance log cleared."}
