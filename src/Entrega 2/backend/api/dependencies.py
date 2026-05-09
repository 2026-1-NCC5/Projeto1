import sys
import os
from typing import Generator
from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from api.core.config import settings
from api.services import auth_service
from bd.models.database import SessionLocal
from bd.models.usuario import Usuario

def get_db() -> Generator:
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()


def get_admin_atual(
    request: Request,
    db: Session = Depends(get_db),
) -> Usuario:
    token = request.cookies.get(settings.AUTH_COOKIE_NAME)
    usuario = auth_service.obter_usuario_por_token(db, token, renovar_acesso=True)
    if usuario is None:
        raise HTTPException(status_code=401, detail="Não autenticado ou sessão expirada")
    return usuario
