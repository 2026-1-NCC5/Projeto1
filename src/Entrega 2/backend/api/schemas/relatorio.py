from typing import List
from pydantic import BaseModel


class RelatorioCategoria(BaseModel):
    alimento_nome: str
    total_quantidade: int
    total_peso_kg: float


class RelatorioGrupo(BaseModel):
    grupo_nome: str
    total_quantidade: int
    total_peso_kg: float


class LinhaAuditoriaSessao(BaseModel):
    """Linha consolidada por alimento: declarado pelo grupo vs detectado pela câmera."""
    alimento_nome: str
    qtd_declarada: int = 0
    peso_declarado_kg: float = 0
    qtd_detectada: int = 0
    peso_detectado_kg: float = 0
    diferenca_qtd: int = 0
    diferenca_peso_kg: float = 0
    status: str  # "validado" | "divergente" | "inesperado" | "nao_detectado"


class RelatorioSessao(BaseModel):
    sessao_id: int
    grupo_id: int
    grupo_nome: str
    status: str
    total_kg_detectado: float = 0
    total_itens_detectados: int = 0
    total_kg_declarado: float = 0
    total_itens_declarados: int = 0
    divergencias: int = 0
    linhas: List[LinhaAuditoriaSessao] = []
