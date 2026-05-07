from typing import List
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.dependencies import get_db
from api.schemas.grupo import GrupoCreate, GrupoUpdate, GrupoResponse
from bd.models.grupo import Grupo

router = APIRouter()


@router.get("/", response_model=List[GrupoResponse])
def listar_grupos(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """Lista todos os grupos cadastrados (equipes arrecadadoras)."""
    return db.query(Grupo).order_by(Grupo.criado_em.desc()).offset(skip).limit(limit).all()


@router.post("/", response_model=GrupoResponse, status_code=201)
def criar_grupo(
    grupo_in: GrupoCreate,
    db: Session = Depends(get_db),
):
    """Cria um grupo (equipe) que será associado a sessões de auditoria."""
    db_obj = Grupo(
        nome=grupo_in.nome,
        descricao=grupo_in.descricao,
        status=grupo_in.status or "pendente",
    )
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


@router.get("/{id}", response_model=GrupoResponse)
def detalhar_grupo(
    id: int,
    db: Session = Depends(get_db),
):
    grupo = db.query(Grupo).filter(Grupo.id == id).first()
    if not grupo:
        raise HTTPException(status_code=404, detail="Grupo não encontrado")
    return grupo


@router.put("/{id}", response_model=GrupoResponse)
def atualizar_grupo(
    id: int,
    grupo_in: GrupoUpdate,
    db: Session = Depends(get_db),
):
    grupo = db.query(Grupo).filter(Grupo.id == id).first()
    if not grupo:
        raise HTTPException(status_code=404, detail="Grupo não encontrado")

    update_data = grupo_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(grupo, field, value)
    grupo.atualizado_em = datetime.utcnow()

    db.add(grupo)
    db.commit()
    db.refresh(grupo)
    return grupo


@router.delete("/{id}", status_code=204)
def deletar_grupo(
    id: int,
    db: Session = Depends(get_db),
):
    grupo = db.query(Grupo).filter(Grupo.id == id).first()
    if not grupo:
        raise HTTPException(status_code=404, detail="Grupo não encontrado")
    db.delete(grupo)
    db.commit()
    return None
