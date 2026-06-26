import uuid

from fastapi import HTTPException

from app.core.auth import create_token, hash_password, verify_password
from app.core.database import OrgQuery, orgs_table
from app.models.schemas import LoginBody, RegisterBody


def register_org(body: RegisterBody) -> dict:
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


def login_org(body: LoginBody) -> dict:
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


def get_org_profile(org_id: str) -> dict:
    org = orgs_table.search(OrgQuery.id == org_id)
    if not org:
        raise HTTPException(404, "Organization not found.")
    return {"org_id": org[0]["id"], "org_name": org[0]["name"], "email": org[0]["email"]}
