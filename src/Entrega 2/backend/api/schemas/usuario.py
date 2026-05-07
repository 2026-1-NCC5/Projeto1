from typing import Optional
from datetime import datetime
from pydantic import BaseModel, EmailStr

class UsuarioBase(BaseModel):
    nome: str
    email: EmailStr
    perfil: str = "operador"
    ativo: bool = True

class UsuarioCreate(UsuarioBase):
    pass

class UsuarioResponse(UsuarioBase):
    id: int
    criado_em: datetime

    class Config:
        from_attributes = True
