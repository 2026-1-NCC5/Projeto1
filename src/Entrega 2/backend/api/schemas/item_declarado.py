from typing import Optional
from datetime import datetime
from pydantic import BaseModel


class ItemDeclaradoBase(BaseModel):
    grupo_id: int
    nome_alimento: str
    marca: Optional[str] = None
    quantidade: int = 1
    peso_declarado_kg: float = 0
    observacao: Optional[str] = None


class ItemDeclaradoCreate(ItemDeclaradoBase):
    pass


class ItemDeclaradoResponse(ItemDeclaradoBase):
    id: int
    criado_em: datetime

    class Config:
        from_attributes = True
