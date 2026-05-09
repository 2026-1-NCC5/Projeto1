from typing import List
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, or_
from api.dependencies import get_db, get_admin_atual
from api.schemas.sessao import SessaoCreate, SessaoResponse, SessaoDecisaoFinalRequest
from api.schemas.relatorio import (
    RelatorioSessao,
    LinhaAuditoriaSessao,
    RevisaoManualItem,
)
from bd.models.sessao import Sessao
from bd.models.usuario import Usuario
from bd.models.grupo import Grupo
from bd.models.deteccao import Deteccao
from bd.models.alimento import Alimento
from bd.models.item_declarado import ItemDeclarado

router = APIRouter()


def _merge_declarado_por_nome(
    declarados: dict[str, tuple[int, float]],
    rows: list,
    nome_attr: str = "nome",
) -> None:
    """Soma quantidade e peso declarado por nome de alimento no mapa `declarados` (in-place)."""
    for r in rows:
        nome = getattr(r, nome_attr, None)
        if not nome:
            continue
        qtd = int(getattr(r, "qtd", 0) or 0)
        peso = float(getattr(r, "peso", 0) or 0)
        dq, dp = declarados.get(nome, (0, 0.0))
        declarados[nome] = (dq + qtd, dp + peso)


def _build_relatorio_sessao(sessao: Sessao, db: Session) -> RelatorioSessao:
    grupo = db.query(Grupo).filter(Grupo.id == sessao.grupo_id).first()

    # Capturas = somente deteccoes da triagem (YOLO/GEMINI etc.), nao MANUAL.
    # MANUAL via tela de insercao ou popup entra no lado "declarado" desta conferencia.
    detectados_rows = (
        db.query(
            Alimento.nome.label("nome"),
            func.coalesce(func.sum(Deteccao.quantidade), 0).label("qtd"),
            func.coalesce(func.sum(Deteccao.peso_kg), 0).label("peso"),
        )
        .join(Deteccao, Deteccao.alimento_id == Alimento.id)
        .filter(
            Deteccao.sessao_id == sessao.id,
            or_(Deteccao.fonte.is_(None), Deteccao.fonte != "MANUAL"),
        )
        .group_by(Alimento.nome)
        .all()
    )
    detectados = {r.nome: (int(r.qtd or 0), float(r.peso or 0)) for r in detectados_rows}

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

    manual_sessao_rows = (
        db.query(
            Alimento.nome.label("nome"),
            func.coalesce(func.sum(Deteccao.quantidade), 0).label("qtd"),
            func.coalesce(func.sum(Deteccao.peso_kg), 0).label("peso"),
        )
        .join(Deteccao, Deteccao.alimento_id == Alimento.id)
        .filter(Deteccao.sessao_id == sessao.id, Deteccao.fonte == "MANUAL")
        .group_by(Alimento.nome)
        .all()
    )
    _merge_declarado_por_nome(declarados, manual_sessao_rows)

    nomes = sorted(set(detectados.keys()) | set(declarados.keys()))
    linhas: list[LinhaAuditoriaSessao] = []
    divergencias = 0
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
        if status != "validado":
            divergencias += 1
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

    total_kg_detectado = float(sum((row[1] for row in detectados.values()), 0.0))
    total_itens_detectados = int(sum((row[0] for row in detectados.values()), 0))
    total_kg_declarado = float(sum((row[1] for row in declarados.values()), 0.0))
    total_itens_declarados = int(sum((row[0] for row in declarados.values()), 0))

    pend_rows = (
        db.query(Deteccao, Alimento.nome)
        .outerjoin(Alimento, Deteccao.alimento_id == Alimento.id)
        .filter(
            Deteccao.sessao_id == sessao.id,
            Deteccao.revisao_manual_pendente.is_(True),
        )
        .order_by(Deteccao.id)
        .all()
    )
    revisao_manual_itens = [
        RevisaoManualItem(
            deteccao_id=d.id,
            alimento_nome=nome,
            imagem_path=d.imagem_path,
            gemini_justificativa=d.gemini_justificativa,
        )
        for d, nome in pend_rows
    ]

    return RelatorioSessao(
        sessao_id=sessao.id,
        grupo_id=sessao.grupo_id,
        grupo_nome=grupo.nome if grupo else "(sem grupo)",
        status=sessao.status,
        total_kg_detectado=total_kg_detectado,
        total_itens_detectados=total_itens_detectados,
        total_kg_declarado=total_kg_declarado,
        total_itens_declarados=total_itens_declarados,
        divergencias=divergencias,
        linhas=linhas,
        revisao_manual_pendente_count=len(revisao_manual_itens),
        revisao_manual_itens=revisao_manual_itens,
    )

@router.post("/", response_model=SessaoResponse)
def create_sessao(
    sessao_in: SessaoCreate,
    db: Session = Depends(get_db),
    admin: Usuario = Depends(get_admin_atual),
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

    db_obj = Sessao(
        grupo_id=sessao_in.grupo_id,
        usuario_id=admin.id,
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
    _admin: Usuario = Depends(get_admin_atual),
):
    """
    Finaliza uma sessão e recalcula totais (bulk) a partir das detecções já gravadas.
    """
    sessao = db.query(Sessao).filter(Sessao.id == id).first()
    if not sessao:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    if sessao.status != "ativa":
        raise HTTPException(status_code=400, detail="Sessão já está finalizada ou cancelada")

    pendentes = (
        db.query(func.count(Deteccao.id))
        .filter(
            Deteccao.sessao_id == id,
            Deteccao.revisao_manual_pendente.is_(True),
        )
        .scalar()
    )
    if pendentes and int(pendentes) > 0:
        raise HTTPException(
            status_code=400,
            detail="Existem itens pendentes de revisão manual de peso ou categoria.",
        )

    if sessao.fonte_resultado_final and sessao.total_kg_final is not None and sessao.total_itens_final is not None:
        sessao.total_kg = float(sessao.total_kg_final or 0)
        sessao.total_itens = int(sessao.total_itens_final or 0)
    else:
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


@router.get("/{id}/conciliacao-previa", response_model=RelatorioSessao)
def conciliacao_previa_sessao(
    id: int,
    db: Session = Depends(get_db),
    _admin: Usuario = Depends(get_admin_atual),
):
    """
    Retorna o comparativo declarado x capturado nesta sessao.

    Lado declarado: `itens_declarados` do grupo + linhas MANUAL gravadas nesta sessao (ex.: ManualScreen).

    Lado capturado: apenas deteccoes com fonte distinta de MANUAL (pipeline da camera).
    """
    sessao = db.query(Sessao).filter(Sessao.id == id).first()
    if not sessao:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    return _build_relatorio_sessao(sessao, db)


@router.put("/{id}/decisao-final", response_model=SessaoResponse)
def decidir_fonte_final_sessao(
    id: int,
    payload: SessaoDecisaoFinalRequest,
    db: Session = Depends(get_db),
    admin: Usuario = Depends(get_admin_atual),
):
    """
    Define a fonte oficial da sessão (manual ou capturas) para totalização final.
    """
    sessao = db.query(Sessao).filter(Sessao.id == id).first()
    if not sessao:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    if sessao.status != "ativa":
        raise HTTPException(status_code=400, detail="Sessão não está ativa")

    pendentes = (
        db.query(func.count(Deteccao.id))
        .filter(
            Deteccao.sessao_id == id,
            Deteccao.revisao_manual_pendente.is_(True),
        )
        .scalar()
    )
    if pendentes and int(pendentes) > 0:
        raise HTTPException(
            status_code=400,
            detail="Existem itens pendentes de revisão manual de peso ou categoria.",
        )

    relatorio = _build_relatorio_sessao(sessao, db)
    if payload.fonte_final == "manual":
        sessao.total_kg_final = float(relatorio.total_kg_declarado or 0)
        sessao.total_itens_final = int(relatorio.total_itens_declarados or 0)
    else:
        sessao.total_kg_final = float(relatorio.total_kg_detectado or 0)
        sessao.total_itens_final = int(relatorio.total_itens_detectados or 0)

    sessao.fonte_resultado_final = payload.fonte_final
    sessao.decisao_final_em = datetime.utcnow()
    sessao.decisao_final_por_usuario_id = admin.id
    db.add(sessao)
    db.commit()
    db.refresh(sessao)
    return sessao

@router.get("/", response_model=List[SessaoResponse])
def listar_sessoes(
    skip: int = 0, limit: int = 100,
    db: Session = Depends(get_db),
    _admin: Usuario = Depends(get_admin_atual),
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
    _admin: Usuario = Depends(get_admin_atual),
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
    _admin: Usuario = Depends(get_admin_atual),
):
    """Comparação entre baseline manual e capturas desta sessao.

    Declarado: cadastro ``itens_declarados`` do grupo + registros MANUAL em ``deteccoes`` desta sessao.

    Detectado pela camera: apenas deteccoes com ``fonte`` diferente de MANUAL.

    Status por linha: ``validado``, ``divergente``, ``inesperado``, ``nao_detectado``.
    """
    sessao = db.query(Sessao).filter(Sessao.id == id).first()
    if not sessao:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    return _build_relatorio_sessao(sessao, db)
