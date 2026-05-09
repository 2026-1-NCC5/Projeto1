from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, case
from api.dependencies import get_db, get_admin_atual
from api.schemas.relatorio import RelatorioCategoria, RelatorioGrupo
from bd.models.deteccao import Deteccao
from bd.models.alimento import Alimento
from bd.models.sessao import Sessao
from bd.models.grupo import Grupo
from bd.models.usuario import Usuario

router = APIRouter()

@router.get("/categorias", response_model=List[RelatorioCategoria])
def relatorio_por_categoria(
    db: Session = Depends(get_db),
    _admin: Usuario = Depends(get_admin_atual),
):
    """
    Gera relatório de totais arrecadados por categoria de alimento
    """
    resultados = db.query(
        Alimento.nome.label("alimento_nome"),
        func.sum(Deteccao.quantidade).label("total_quantidade"),
        func.sum(Deteccao.peso_kg).label("total_peso_kg")
    ).join(Deteccao, Alimento.id == Deteccao.alimento_id)\
     .group_by(Alimento.nome).all()
     
    return [{"alimento_nome": r[0], "total_quantidade": r[1] or 0, "total_peso_kg": float(r[2] or 0)} for r in resultados]

@router.get("/grupos", response_model=List[RelatorioGrupo])
def relatorio_por_grupo(
    db: Session = Depends(get_db),
    _admin: Usuario = Depends(get_admin_atual),
):
    """
    Gera relatório de totais arrecadados por grupo
    """
    total_quantidade_expr = case(
        (
            (Sessao.status == "finalizada") & (Sessao.total_itens_final.isnot(None)),
            Sessao.total_itens_final,
        ),
        else_=Sessao.total_itens,
    )
    total_peso_expr = case(
        (
            (Sessao.status == "finalizada") & (Sessao.total_kg_final.isnot(None)),
            Sessao.total_kg_final,
        ),
        else_=Sessao.total_kg,
    )

    resultados = db.query(
        Grupo.nome.label("grupo_nome"),
        func.coalesce(func.sum(total_quantidade_expr), 0).label("total_quantidade"),
        func.coalesce(func.sum(total_peso_expr), 0).label("total_peso_kg")
    ).select_from(Grupo)\
     .join(Sessao, Sessao.grupo_id == Grupo.id)\
     .group_by(Grupo.nome).all()
     
    return [{"grupo_nome": r[0], "total_quantidade": r[1] or 0, "total_peso_kg": float(r[2] or 0)} for r in resultados]
