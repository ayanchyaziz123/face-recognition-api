from pydantic import BaseModel


class RegisterBody(BaseModel):
    org_name: str
    email:    str
    password: str


class LoginBody(BaseModel):
    email:    str
    password: str


class AttendanceRecord(BaseModel):
    org_id:     str
    name:       str
    timestamp:  str
    confidence: float


class MemberRecord(BaseModel):
    id:     str
    org_id: str
    name:   str
