from typing import Optional
from datetime import datetime
from pydantic import BaseModel


class GrupoBase(BaseModel):
    nome: str
    descricao: Optional[str] = None
    status: Optional[str] = "pendente"


class GrupoCreate(GrupoBase):
    pass


class GrupoUpdate(BaseModel):
    nome: Optional[str] = None
    descricao: Optional[str] = None
    status: Optional[str] = None


class GrupoResponse(GrupoBase):
    id: int
    criado_em: datetime
    atualizado_em: datetime

    class Config:
        from_attributes = True
