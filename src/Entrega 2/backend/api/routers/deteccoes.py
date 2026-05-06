from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from api.dependencies import get_db, get_current_active_user
from api.schemas.deteccao import DeteccaoCreate, DeteccaoResponse, DeteccaoCorrecao
from bd.models.deteccao import Deteccao
from bd.models.usuario import Usuario
from bd.models.sessao import Sessao

router = APIRouter()

@router.post("/", response_model=DeteccaoResponse)
def create_deteccao(
    deteccao_in: DeteccaoCreate,
    db: Session = Depends(get_db)
):
    """
    Cria uma nova detecção (usado pelo sistema de IA)
    Não exige autenticação por padrão para facilitar integração com o script da câmera.
    """
    sessao = db.query(Sessao).filter(Sessao.id == deteccao_in.sessao_id).first()
    if not sessao:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    
    db_obj = Deteccao(
        sessao_id=deteccao_in.sessao_id,
        alimento_id=deteccao_in.alimento_id,
        alimento_id_original=deteccao_in.alimento_id,
        peso_kg=deteccao_in.peso_kg,
        quantidade=deteccao_in.quantidade,
        confianca=deteccao_in.confianca,
        imagem_path=deteccao_in.imagem_path,
        corrigido_manualmente=False
    )
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj

@router.put("/{id}/correcao", response_model=DeteccaoResponse)
def corrigir_deteccao(
    id: int,
    correcao_in: DeteccaoCorrecao,
    db: Session = Depends(get_db),
    current_user: Usuario = Depends(get_current_active_user)
):
    """
    Corrige manualmente uma classificação de alimento
    """
    deteccao = db.query(Deteccao).filter(Deteccao.id == id).first()
    if not deteccao:
        raise HTTPException(status_code=404, detail="Detecção não encontrada")
    
    deteccao.alimento_id = correcao_in.alimento_id
    deteccao.corrigido_manualmente = True
    
    db.add(deteccao)
    db.commit()
    db.refresh(deteccao)
    return deteccao
