from typing import Optional
from datetime import datetime
from pydantic import BaseModel

class DeteccaoBase(BaseModel):
    alimento_id: Optional[int] = None
    peso_kg: float
    quantidade: int = 1
    confianca: Optional[float] = None
    imagem_path: Optional[str] = None

class DeteccaoCreate(DeteccaoBase):
    sessao_id: int
    alimento_id_original: Optional[int] = None
    fonte: Optional[str] = "YOLO"
    gemini_concorda: Optional[bool] = None
    gemini_classe: Optional[str] = None
    gemini_justificativa: Optional[str] = None

class DeteccaoCorrecao(BaseModel):
    alimento_id: Optional[int] = None
    peso_kg: Optional[float] = None

class DeteccaoResponse(DeteccaoBase):
    id: int
    sessao_id: int
    alimento_id_original: Optional[int] = None
    corrigido_manualmente: bool
    fonte: str
    gemini_concorda: Optional[bool] = None
    gemini_classe: Optional[str] = None
    gemini_justificativa: Optional[str] = None
    revisao_manual_pendente: bool = False
    criado_em: datetime

    class Config:
        from_attributes = True
