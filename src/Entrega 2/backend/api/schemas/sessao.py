from typing import Optional, List, Literal
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
    fonte_resultado_final: Optional[str] = None
    decisao_final_em: Optional[datetime] = None
    decisao_final_por_usuario_id: Optional[int] = None
    total_kg_final: Optional[float] = None
    total_itens_final: Optional[int] = None

    deteccoes: List[DeteccaoResponse] = []

    class Config:
        from_attributes = True


class SessaoDecisaoFinalRequest(BaseModel):
    fonte_final: Literal["manual", "capturas"]
    usuario_id: Optional[int] = None
