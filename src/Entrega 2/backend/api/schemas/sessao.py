from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel
from api.schemas.deteccao import DeteccaoResponse

class SessaoBase(BaseModel):
    grupo_id: int

class SessaoCreate(SessaoBase):
    pass

class SessaoResponse(SessaoBase):
    id: int
    usuario_id: int
    data_inicio: datetime
    data_fim: Optional[datetime] = None
    status: str
    
    deteccoes: List[DeteccaoResponse] = []

    class Config:
        from_attributes = True
