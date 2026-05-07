from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel
from api.schemas.deteccao import DeteccaoResponse

class SessaoBase(BaseModel):
    grupo_id: int

class SessaoCreate(SessaoBase):
    usuario_id: Optional[int] = None

class SessaoResponse(SessaoBase):
    id: int
    usuario_id: Optional[int] = None
    inicio: datetime
    fim: Optional[datetime] = None
    status: str
    total_kg: float = 0
    total_itens: int = 0

    deteccoes: List[DeteccaoResponse] = []

    class Config:
        from_attributes = True
