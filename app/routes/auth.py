import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.auth import create_token, get_current_org, hash_password, verify_password
from app.database import OrgQuery, orgs_table

router = APIRouter(prefix="/auth", tags=["Auth"])


class RegisterBody(BaseModel):
    org_name: str
    email:    str
    password: str


class LoginBody(BaseModel):
    email:    str
    password: str


@router.post("/register", status_code=201)
def register(body: RegisterBody):
    if orgs_table.search(OrgQuery.email == body.email):
        raise HTTPException(400, "Email already registered.")

    org_id = str(uuid.uuid4())
    orgs_table.insert({
        "id":       org_id,
        "name":     body.org_name,
        "email":    body.email,
        "password": hash_password(body.password),
    })
    return {"message": "Organization registered.", "org_id": org_id}


@router.post("/login")
def login(body: LoginBody):
    results = orgs_table.search(OrgQuery.email == body.email)
    if not results or not verify_password(body.password, results[0]["password"]):
        raise HTTPException(401, "Invalid email or password.")

    org = results[0]
    return {
        "access_token": create_token(org["id"]),
        "token_type":   "bearer",
        "org_name":     org["name"],
        "org_id":       org["id"],
    }


@router.get("/me")
def me(org_id: str = Depends(get_current_org)):
    org = orgs_table.search(OrgQuery.id == org_id)
    if not org:
        raise HTTPException(404, "Organization not found.")
    return {"org_id": org[0]["id"], "org_name": org[0]["name"], "email": org[0]["email"]}
