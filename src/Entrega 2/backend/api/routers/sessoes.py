from typing import List
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from api.dependencies import get_db, get_current_active_user
from api.schemas.sessao import SessaoCreate, SessaoResponse
from bd.models.sessao import Sessao
from bd.models.usuario import Usuario
from bd.models.grupo import Grupo

router = APIRouter()

@router.post("/", response_model=SessaoResponse)
def create_sessao(
    sessao_in: SessaoCreate,
    db: Session = Depends(get_db),
    current_user: Usuario = Depends(get_current_active_user)
):
    """
    Inicia uma nova sessão de triagem para um grupo
    """
    grupo = db.query(Grupo).filter(Grupo.id == sessao_in.grupo_id).first()
    if not grupo:
        raise HTTPException(status_code=404, detail="Grupo não encontrado")
        
    sessao_ativa = db.query(Sessao).filter(
        Sessao.grupo_id == sessao_in.grupo_id, 
        Sessao.status == "aberta"
    ).first()
    if sessao_ativa:
        raise HTTPException(status_code=400, detail="Grupo já possui uma sessão aberta")

    db_obj = Sessao(
        grupo_id=sessao_in.grupo_id,
        usuario_id=current_user.id,
        status="aberta"
    )
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj

@router.put("/{id}/finalizar", response_model=SessaoResponse)
def finalizar_sessao(
    id: int,
    db: Session = Depends(get_db),
    current_user: Usuario = Depends(get_current_active_user)
):
    """
    Finaliza uma sessão de triagem
    """
    sessao = db.query(Sessao).filter(Sessao.id == id).first()
    if not sessao:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
        
    sessao.status = "concluida"
    sessao.data_fim = datetime.utcnow()
    
    db.add(sessao)
    db.commit()
    db.refresh(sessao)
    return sessao

@router.get("/", response_model=List[SessaoResponse])
def listar_sessoes(
    skip: int = 0, limit: int = 100,
    db: Session = Depends(get_db),
    current_user: Usuario = Depends(get_current_active_user)
):
    """
    Lista sessões
    """
    sessoes = db.query(Sessao).offset(skip).limit(limit).all()
    return sessoes
