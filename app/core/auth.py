from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

SECRET_KEY         = "change-this-secret-in-production"
ALGORITHM          = "HS256"
TOKEN_EXPIRE_HOURS = 24

_bearer = HTTPBearer()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


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
