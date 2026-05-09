from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.dependencies import get_db, get_admin_atual
from api.schemas.item_declarado import ItemDeclaradoCreate, ItemDeclaradoResponse
from bd.models.item_declarado import ItemDeclarado
from bd.models.grupo import Grupo
from bd.models.usuario import Usuario

router = APIRouter()


@router.post("/", response_model=ItemDeclaradoResponse, status_code=201)
def criar_item_declarado(
    item_in: ItemDeclaradoCreate,
    db: Session = Depends(get_db),
    _admin: Usuario = Depends(get_admin_atual),
):
    """
    Cadastra item que o admin declara que o grupo trouxe (cadastro manual prévio à sessão).
    Esses dados servem de baseline para o relatório comparativo.
    """
    grupo = db.query(Grupo).filter(Grupo.id == item_in.grupo_id).first()
    if not grupo:
        raise HTTPException(status_code=404, detail="Grupo não encontrado")

    db_obj = ItemDeclarado(**item_in.model_dump())
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


@router.get("/grupo/{grupo_id}", response_model=List[ItemDeclaradoResponse])
def listar_por_grupo(
    grupo_id: int,
    db: Session = Depends(get_db),
    _admin: Usuario = Depends(get_admin_atual),
):
    """Lista todos os itens declarados de um grupo (utilizado pela tela de auditoria)."""
    return (
        db.query(ItemDeclarado)
        .filter(ItemDeclarado.grupo_id == grupo_id)
        .order_by(ItemDeclarado.criado_em.desc())
        .all()
    )


@router.delete("/{id}", status_code=204)
def deletar_item_declarado(
    id: int,
    db: Session = Depends(get_db),
    _admin: Usuario = Depends(get_admin_atual),
):
    item = db.query(ItemDeclarado).filter(ItemDeclarado.id == id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item declarado não encontrado")
    db.delete(item)
    db.commit()
    return None
