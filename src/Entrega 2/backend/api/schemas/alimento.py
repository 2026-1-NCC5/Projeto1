from typing import Optional
from datetime import datetime
from pydantic import BaseModel


class AlimentoBase(BaseModel):
    nome: str
    grupo_alimento_id: Optional[int] = None
    peso_padrao_kg: float
    unidade: str = "kg"
    classe_yolo: Optional[str] = None
    descricao: Optional[str] = None
    ativo: bool = True


class AlimentoCreate(AlimentoBase):
    pass


class AlimentoUpdate(BaseModel):
    nome: Optional[str] = None
    grupo_alimento_id: Optional[int] = None
    peso_padrao_kg: Optional[float] = None
    unidade: Optional[str] = None
    classe_yolo: Optional[str] = None
    descricao: Optional[str] = None
    ativo: Optional[bool] = None


class AlimentoResponse(AlimentoBase):
    id: int
    criado_em: datetime

    class Config:
        from_attributes = True
