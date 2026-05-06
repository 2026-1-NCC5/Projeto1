from pydantic import BaseModel

class RelatorioCategoria(BaseModel):
    alimento_nome: str
    total_quantidade: int
    total_peso_kg: float

class RelatorioGrupo(BaseModel):
    grupo_nome: str
    total_quantidade: int
    total_peso_kg: float
