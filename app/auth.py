from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

SECRET_KEY         = "change-this-secret-in-production"
ALGORITHM          = "HS256"
TOKEN_EXPIRE_HOURS = 24

_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
_bearer = HTTPBearer()


def hash_password(password: str) -> str:
    return _pwd.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd.verify(plain, hashed)


def create_token(org_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode({"sub": org_id, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def get_current_org(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> str:
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        org_id  = payload.get("sub")
        if not org_id:
            raise HTTPException(401, "Invalid token.")
        return org_id
    except JWTError:
        raise HTTPException(401, "Invalid or expired token.")
