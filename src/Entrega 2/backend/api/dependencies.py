import sys
import os
from typing import Generator
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from sqlalchemy.orm import Session

# Add backend/bd to sys.path so we can import models easily
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from bd.models.database import SessionLocal
from bd.models.usuario import Usuario
from api.core.config import settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/login/access-token")

def get_db() -> Generator:
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

def get_current_user(
    db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)
) -> Usuario:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        token_data = payload.get("sub")
        if token_data is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    try:
        user_id = int(token_data)
    except ValueError:
        raise credentials_exception

    user = db.query(Usuario).filter(Usuario.id == user_id).first()
    if not user:
        raise credentials_exception
    return user

def get_current_active_user(
    current_user: Usuario = Depends(get_current_user),
) -> Usuario:
    if not current_user.ativo:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
