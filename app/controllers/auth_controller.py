from fastapi import APIRouter, Depends

from app.core.auth import get_current_org
from app.models.schemas import LoginBody, RegisterBody
from app.services import auth_service

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/register", status_code=201)
def register(body: RegisterBody):
    return auth_service.register_org(body)


@router.post("/login")
def login(body: LoginBody):
    return auth_service.login_org(body)


@router.get("/me")
def me(org_id: str = Depends(get_current_org)):
    return auth_service.get_org_profile(org_id)
