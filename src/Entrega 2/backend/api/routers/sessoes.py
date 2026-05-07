from typing import List
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from api.dependencies import get_db
from api.schemas.sessao import SessaoCreate, SessaoResponse
from api.schemas.relatorio import RelatorioSessao, LinhaAuditoriaSessao
from bd.models.sessao import Sessao
from bd.models.usuario import Usuario
from bd.models.grupo import Grupo
from bd.models.deteccao import Deteccao
from bd.models.alimento import Alimento
from bd.models.item_declarado import ItemDeclarado

router = APIRouter()

@router.post("/", response_model=SessaoResponse)
def create_sessao(
    sessao_in: SessaoCreate,
    db: Session = Depends(get_db),
):
    """
    Inicia uma nova sessão de auditoria para um grupo
    """
    grupo = db.query(Grupo).filter(Grupo.id == sessao_in.grupo_id).first()
    if not grupo:
        raise HTTPException(status_code=404, detail="Grupo não encontrado")

    sessao_ativa = db.query(Sessao).filter(
        Sessao.grupo_id == sessao_in.grupo_id,
        Sessao.status == "ativa"
    ).first()
    if sessao_ativa:
        raise HTTPException(status_code=400, detail="Grupo já possui uma sessão ativa")

    usuario_id = sessao_in.usuario_id
    if usuario_id is None:
        usuario = db.query(Usuario).order_by(Usuario.id).first()
        usuario_id = usuario.id if usuario else None

    db_obj = Sessao(
        grupo_id=sessao_in.grupo_id,
        usuario_id=usuario_id,
        status="ativa"
    )
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj

@router.put("/{id}/finalizar", response_model=SessaoResponse)
def finalizar_sessao(
    id: int,
    db: Session = Depends(get_db),
):
    """
    Finaliza uma sessão e recalcula totais (bulk) a partir das detecções já gravadas.
    """
    sessao = db.query(Sessao).filter(Sessao.id == id).first()
    if not sessao:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    if sessao.status != "ativa":
        raise HTTPException(status_code=400, detail="Sessão já está finalizada ou cancelada")

    total_peso, total_qtd = db.query(
        func.coalesce(func.sum(Deteccao.peso_kg), 0),
        func.coalesce(func.sum(Deteccao.quantidade), 0),
    ).filter(Deteccao.sessao_id == id).one()

    sessao.total_kg = float(total_peso or 0)
    sessao.total_itens = int(total_qtd or 0)
    sessao.fim = datetime.utcnow()
    sessao.status = "finalizada"

    db.add(sessao)
    db.commit()
    db.refresh(sessao)
    return sessao

@router.get("/", response_model=List[SessaoResponse])
def listar_sessoes(
    skip: int = 0, limit: int = 100,
    db: Session = Depends(get_db),
):
    """
    Lista sessões
    """
    sessoes = db.query(Sessao).order_by(Sessao.criado_em.desc()).offset(skip).limit(limit).all()
    return sessoes


@router.get("/{id}", response_model=SessaoResponse)
def detalhar_sessao(
    id: int,
    db: Session = Depends(get_db),
):
    """
    Retorna uma sessão com suas detecções (lista completa, conferida na auditoria final).
    """
    sessao = db.query(Sessao).filter(Sessao.id == id).first()
    if not sessao:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    return sessao


@router.get("/{id}/relatorio", response_model=RelatorioSessao)
def relatorio_sessao(
    id: int,
    db: Session = Depends(get_db),
):
    """Comparação entre itens declarados pelo admin e detecções da câmera.

    Para cada alimento aparecendo em qualquer um dos lados, retorna uma linha
    com qtd/peso declarado, qtd/peso detectado e diferença. Status:
    - ``validado``: detectado e declarado batem (ambos > 0)
    - ``divergente``: declarado e detectado, mas valores diferem
    - ``inesperado``: detectado, mas não estava declarado
    - ``nao_detectado``: declarado, mas câmera não viu
    """
    sessao = db.query(Sessao).filter(Sessao.id == id).first()
    if not sessao:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")

    grupo = db.query(Grupo).filter(Grupo.id == sessao.grupo_id).first()

    # 1) Detecções agregadas por nome do alimento (nome final, pós-correção)
    detectados_rows = (
        db.query(
            Alimento.nome.label("nome"),
            func.coalesce(func.sum(Deteccao.quantidade), 0).label("qtd"),
            func.coalesce(func.sum(Deteccao.peso_kg), 0).label("peso"),
        )
        .join(Deteccao, Deteccao.alimento_id == Alimento.id)
        .filter(Deteccao.sessao_id == id)
        .group_by(Alimento.nome)
        .all()
    )
    detectados = {r.nome: (int(r.qtd or 0), float(r.peso or 0)) for r in detectados_rows}

    # 2) Itens declarados (cadastro manual prévio do admin) agregados por nome
    declarados_rows = (
        db.query(
            ItemDeclarado.nome_alimento.label("nome"),
            func.coalesce(func.sum(ItemDeclarado.quantidade), 0).label("qtd"),
            func.coalesce(func.sum(ItemDeclarado.peso_declarado_kg), 0).label("peso"),
        )
        .filter(ItemDeclarado.grupo_id == sessao.grupo_id)
        .group_by(ItemDeclarado.nome_alimento)
        .all()
    )
    declarados = {r.nome: (int(r.qtd or 0), float(r.peso or 0)) for r in declarados_rows}

    nomes = sorted(set(detectados.keys()) | set(declarados.keys()))
    linhas: list[LinhaAuditoriaSessao] = []
    for nome in nomes:
        qd, pd = declarados.get(nome, (0, 0.0))
        qe, pe = detectados.get(nome, (0, 0.0))
        diff_q = qe - qd
        diff_p = round(pe - pd, 2)
        if qd == 0 and qe > 0:
            status = "inesperado"
        elif qe == 0 and qd > 0:
            status = "nao_detectado"
        elif diff_q == 0 and abs(diff_p) < 0.01:
            status = "validado"
        else:
            status = "divergente"
        linhas.append(
            LinhaAuditoriaSessao(
                alimento_nome=nome,
                qtd_declarada=qd,
                peso_declarado_kg=pd,
                qtd_detectada=qe,
                peso_detectado_kg=pe,
                diferenca_qtd=diff_q,
                diferenca_peso_kg=diff_p,
                status=status,
            )
        )

    return RelatorioSessao(
        sessao_id=sessao.id,
        grupo_id=sessao.grupo_id,
        grupo_nome=grupo.nome if grupo else "(sem grupo)",
        status=sessao.status,
        total_kg_detectado=float(sessao.total_kg or 0),
        total_itens_detectados=int(sessao.total_itens or 0),
        linhas=linhas,
    )
