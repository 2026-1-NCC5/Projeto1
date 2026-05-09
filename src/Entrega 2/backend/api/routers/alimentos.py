from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.dependencies import get_db, get_admin_atual
from api.schemas.alimento import AlimentoCreate, AlimentoUpdate, AlimentoResponse
from bd.models.alimento import Alimento
from bd.models.usuario import Usuario

router = APIRouter()


@router.get("/", response_model=List[AlimentoResponse])
def listar_alimentos(
    apenas_ativos: bool = True,
    db: Session = Depends(get_db),
    _admin: Usuario = Depends(get_admin_atual),
):
    """Lista alimentos cadastrados (requer sessão de admin)."""
    query = db.query(Alimento)
    if apenas_ativos:
        query = query.filter(Alimento.ativo.is_(True))
    return query.order_by(Alimento.nome).all()


@router.post("/", response_model=AlimentoResponse, status_code=201)
def criar_alimento(
    alimento_in: AlimentoCreate,
    db: Session = Depends(get_db),
    _admin: Usuario = Depends(get_admin_atual),
):
    """Cadastro de alimento — admin define classe_yolo e peso médio antes da sessão."""
    existente = db.query(Alimento).filter(Alimento.nome == alimento_in.nome).first()
    if existente:
        raise HTTPException(status_code=400, detail="Já existe alimento com esse nome")

    db_obj = Alimento(**alimento_in.model_dump())
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


@router.get("/{id}", response_model=AlimentoResponse)
def detalhar_alimento(
    id: int,
    db: Session = Depends(get_db),
    _admin: Usuario = Depends(get_admin_atual),
):
    alimento = db.query(Alimento).filter(Alimento.id == id).first()
    if not alimento:
        raise HTTPException(status_code=404, detail="Alimento não encontrado")
    return alimento


@router.put("/{id}", response_model=AlimentoResponse)
def atualizar_alimento(
    id: int,
    alimento_in: AlimentoUpdate,
    db: Session = Depends(get_db),
    _admin: Usuario = Depends(get_admin_atual),
):
    alimento = db.query(Alimento).filter(Alimento.id == id).first()
    if not alimento:
        raise HTTPException(status_code=404, detail="Alimento não encontrado")

    update_data = alimento_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(alimento, field, value)

    db.add(alimento)
    db.commit()
    db.refresh(alimento)
    return alimento


@router.delete("/{id}", status_code=204)
def deletar_alimento(
    id: int,
    db: Session = Depends(get_db),
    _admin: Usuario = Depends(get_admin_atual),
):
    alimento = db.query(Alimento).filter(Alimento.id == id).first()
    if not alimento:
        raise HTTPException(status_code=404, detail="Alimento não encontrado")
    # Soft delete preserva referências históricas em deteccoes
    alimento.ativo = False
    db.add(alimento)
    db.commit()
    return None
