"""WebSocket /ws/auditoria/{sessao_id}: pipeline câmera → YOLO → Gemini.

Mensagens cliente → servidor:
- {"tipo": "frame", "ts": <ms>, "imagem_b64": "data:image/jpeg;base64,...", "usar_gemini": bool}
- {"tipo": "reset"} (libera lock cedo, reinicia estabilidade)

Mensagens servidor → cliente:
- {"tipo": "preview", "ts": <ms>, "yolo": {...}|null}
- {"tipo": "status", "estado": "monitorando"|"estavel"|"analisando"|"lock", "lock_ate_ts": <ms>}
- {"tipo": "deteccao_preliminar", "deteccao_id": int, "yolo": {...}, "resultado_final": {...}, "imagem_path": str, "ts": <ms>}
- {"tipo": "deteccao_atualizada", "deteccao_id": int, "gemini": {...}|null, "alimento_id": int|null, "alimento_nome": str|null, "fonte": "YOLO"|"GEMINI", "ts": <ms>}
- {"tipo": "log", "stage": str, "mensagem": str, "dados": {...}, "ts": <ms>}
- {"tipo": "erro", "stage": "yolo"|"gemini"|"io", "mensagem": "..."}

O fluxo é assíncrono: ao bater o gatilho de estabilidade, persistimos uma
detecção preliminar com a classe YOLO e disparamos o Gemini em background
(`asyncio.create_task`). O frontend reage à `deteccao_preliminar`
imediatamente (mostra o item na lista, segue capturando) e atualiza o item
quando a `deteccao_atualizada` chegar.

Requer cookie de sessão de admin válido; a sessão de triagem (`sessao_id`)
deve ter sido aberta pelo mesmo usuário (`sessoes.usuario_id`).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.core.config import settings
from api.services.auth_service import (
    obter_usuario_por_token as auth_obter_usuario_por_token,
)
from api.services.estabilidade_service import (
    EstadoSessao,
    avaliar_fallback_sem_yolo,
    avaliar_gatilho,
    descrever_estado,
)
from api.services.evidencia_service import salvar_evidencia
from api.services.gemini_service import get_gemini_service
from api.services.yolo_service import get_yolo_service
from bd.models.alimento import Alimento
from bd.models.database import SessionLocal
from bd.models.deteccao import Deteccao
from bd.models.sessao import Sessao

logger = logging.getLogger(__name__)
router = APIRouter()


def _hidratar_classes(db) -> dict[str, Alimento]:
    """Mapa classe_yolo -> Alimento (somente ativos com classe definida)."""
    rows = (
        db.query(Alimento)
        .filter(Alimento.ativo.is_(True), Alimento.classe_yolo.isnot(None))
        .all()
    )
    return {a.classe_yolo: a for a in rows}


def _anexar_alimento_yolo(
    palpite: Optional[dict[str, Any]],
    classes_db: dict[str, Alimento],
) -> Optional[dict[str, Any]]:
    if not palpite:
        return None
    alimento = classes_db.get(palpite.get("classe"))
    return {
        **palpite,
        "alimento_id": alimento.id if alimento else None,
        "alimento_nome": alimento.nome if alimento else None,
    }


def _log_payload(stage: str, mensagem: str, **dados: Any) -> dict[str, Any]:
    return {
        "tipo": "log",
        "stage": stage,
        "mensagem": mensagem,
        "ts": int(time.time() * 1000),
        "dados": dados,
    }


async def _processar_gemini_em_background(
    websocket: WebSocket,
    deteccao_id: int,
    jpeg_bytes: bytes,
    palpite: Optional[dict[str, Any]],
    classes_validas: list[str],
    classes_db: dict[str, Alimento],
    gemini: Any,
) -> None:
    """Roda Gemini em thread separada, atualiza a detecção no banco e
    notifica o cliente via WS (best-effort).

    Aberto seu próprio SessionLocal pois a sessão DB do handler já foi
    fechada antes do gather. Falhas no envio WS (cliente desconectou)
    são silenciosas — o UPDATE no banco já completou.
    """
    try:
        gemini_out = await asyncio.to_thread(
            gemini.validar, jpeg_bytes, palpite, classes_validas
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gemini falhou em background (det=%s): %s", deteccao_id, exc)
        gemini_out = None

    db = SessionLocal()
    novo_alimento_nome: Optional[str] = None
    try:
        deteccao = db.get(Deteccao, deteccao_id)
        if deteccao is None:
            return
        if gemini_out is not None:
            deteccao.gemini_concorda = bool(gemini_out.get("concorda"))
            deteccao.gemini_classe = gemini_out.get("classe")
            deteccao.gemini_justificativa = gemini_out.get("justificativa")
            if not gemini_out.get("concorda") and gemini_out.get("classe"):
                novo_alimento = classes_db.get(gemini_out["classe"])
                if novo_alimento is not None and novo_alimento.id != deteccao.alimento_id:
                    deteccao.alimento_id = novo_alimento.id
                    deteccao.fonte = "GEMINI"
                    novo_alimento_nome = novo_alimento.nome
        db.commit()
        db.refresh(deteccao)
        fonte_atual = deteccao.fonte
        alimento_id_atual = deteccao.alimento_id
    except Exception:
        logger.exception("Falha ao atualizar detecção %s com Gemini", deteccao_id)
        db.rollback()
        return
    finally:
        db.close()

    try:
        await websocket.send_json({
            "tipo": "deteccao_atualizada",
            "deteccao_id": deteccao_id,
            "gemini": gemini_out,
            "alimento_id": alimento_id_atual,
            "alimento_nome": novo_alimento_nome,
            "fonte": fonte_atual,
            "ts": int(time.time() * 1000),
        })
    except Exception:
        # WS pode ter fechado antes do background concluir — ok, o UPDATE persistiu.
        pass


@router.websocket("/auditoria/{sessao_id}")
async def ws_auditoria(websocket: WebSocket, sessao_id: int):
    await websocket.accept()

    db = SessionLocal()
    try:
        token = websocket.cookies.get(settings.AUTH_COOKIE_NAME)
        usuario = auth_obter_usuario_por_token(db, token, renovar_acesso=True)
        if usuario is None:
            await websocket.send_json(
                {
                    "tipo": "erro",
                    "stage": "io",
                    "mensagem": "Não autenticado ou sessão expirada",
                }
            )
            await websocket.close(code=4401)
            return

        sessao = db.query(Sessao).filter(Sessao.id == sessao_id).first()
        if not sessao:
            await websocket.send_json(
                {"tipo": "erro", "stage": "io", "mensagem": "Sessão não encontrada"}
            )
            await websocket.close(code=4404)
            return
        if sessao.status != "ativa":
            await websocket.send_json(
                {"tipo": "erro", "stage": "io", "mensagem": "Sessão não está ativa"}
            )
            await websocket.close(code=4400)
            return
        if sessao.usuario_id is not None and sessao.usuario_id != usuario.id:
            await websocket.send_json(
                {
                    "tipo": "erro",
                    "stage": "io",
                    "mensagem": "Sessão de triagem vinculada a outro administrador",
                }
            )
            await websocket.close(code=4403)
            return
        classes_db = _hidratar_classes(db)
    finally:
        db.close()

    # Carrega serviços (YOLO é pesado — singleton lazy garante uma única carga).
    try:
        yolo = get_yolo_service()
    except Exception as exc:
        logger.exception("Falha ao carregar YOLO")
        await websocket.send_json(
            {"tipo": "erro", "stage": "yolo", "mensagem": str(exc)}
        )
        await websocket.close(code=4500)
        return
    gemini = get_gemini_service()  # pode ser None (modo degradado)

    estado = EstadoSessao()
    classes_validas = sorted(classes_db.keys())
    await websocket.send_json(
        _log_payload(
            "ws",
            "WebSocket conectado",
            sessao_id=sessao_id,
            classes=len(classes_validas),
            gemini_disponivel=gemini is not None,
        )
    )

    try:
        while True:
            msg = await websocket.receive_json()
            tipo = msg.get("tipo")

            if tipo == "reset":
                estado.reset_total()
                await websocket.send_json(
                    {"tipo": "status", "estado": "monitorando", "lock_ate_ts": 0}
                )
                await websocket.send_json(_log_payload("ws", "Estado de captura resetado"))
                continue

            if tipo != "frame":
                await websocket.send_json(
                    _log_payload("ws", "Mensagem ignorada", tipo_recebido=tipo)
                )
                continue

            imagem_b64 = msg.get("imagem_b64")
            usar_gemini = bool(msg.get("usar_gemini", True))
            frame = yolo.decodificar_b64(imagem_b64) if imagem_b64 else None
            if frame is None:
                await websocket.send_json(
                    {"tipo": "erro", "stage": "io", "mensagem": "Frame inválido"}
                )
                continue

            # 1) YOLO sempre roda (também serve de monitoramento de presença)
            try:
                palpite = yolo.detectar_dominante(frame)
            except Exception as exc:
                logger.exception("Erro no YOLO")
                await websocket.send_json(
                    {"tipo": "erro", "stage": "yolo", "mensagem": str(exc)}
                )
                continue

            agora = time.time()
            palpite_payload = _anexar_alimento_yolo(palpite, classes_db)
            await websocket.send_json({
                "tipo": "preview",
                "ts": int(agora * 1000),
                "yolo": palpite_payload,
            })

            bbox = tuple(palpite["bbox"]) if palpite else None
            if bbox is None and gemini is not None and usar_gemini:
                disparar = avaliar_fallback_sem_yolo(
                    estado,
                    agora,
                    settings.STABILITY_SECONDS,
                )
            else:
                disparar = avaliar_gatilho(
                    estado,
                    bbox,
                    agora,
                    settings.STABILITY_SECONDS,
                    settings.STABILITY_IOU_MIN,
                )

            if not disparar:
                await websocket.send_json({
                    "tipo": "status",
                    "estado": descrever_estado(estado),
                    "lock_ate_ts": int(estado.lock_ate * 1000),
                })
                continue

            # 2) Lock imediato anti-duplicidade (mesma janela visual de antes)
            estado.liberar_lock_em(settings.LOCK_SECONDS)
            await websocket.send_json({
                "tipo": "status",
                "estado": "analisando",
                "lock_ate_ts": int(estado.lock_ate * 1000),
            })
            await websocket.send_json(
                _log_payload(
                    "yolo",
                    "Objeto estável, registrando preliminar",
                    classe=palpite.get("classe") if palpite else None,
                    confianca=palpite.get("confianca") if palpite else None,
                    gemini_ativado=usar_gemini and gemini is not None,
                )
            )

            # 3) Codifica JPEG (para evidência + futura análise Gemini)
            try:
                jpeg_bytes = yolo.codificar_jpeg(frame, qualidade=85)
            except Exception as exc:
                logger.exception("Falha ao codificar JPEG")
                await websocket.send_json(
                    {"tipo": "erro", "stage": "io", "mensagem": str(exc)}
                )
                estado.reset()
                continue

            # 4) Persistência da evidência visual
            imagem_path = salvar_evidencia(sessao_id, jpeg_bytes)

            # 5) INSERT preliminar com YOLO (gemini_* fica null até background task)
            alimento_yolo = (
                classes_db.get(palpite["classe"]) if palpite and palpite.get("classe") else None
            )
            if alimento_yolo is None:
                # Sem classe YOLO mapeada: não dá pra registrar preliminar; pula.
                # Ainda assim mantemos o lock para evitar que o mesmo frame
                # dispare imediatamente outra rodada.
                await websocket.send_json(
                    _log_payload(
                        "yolo",
                        "Sem alimento mapeado para a classe — preliminar descartada",
                        classe=palpite.get("classe") if palpite else None,
                    )
                )
                estado.reset()
                continue

            db_persist = SessionLocal()
            try:
                deteccao = Deteccao(
                    sessao_id=sessao_id,
                    alimento_id=alimento_yolo.id,
                    alimento_id_original=alimento_yolo.id,
                    peso_kg=float(alimento_yolo.peso_padrao_kg or 0.0),
                    quantidade=1,
                    confianca=(palpite.get("confianca") if palpite else None),
                    imagem_path=imagem_path,
                    corrigido_manualmente=False,
                    fonte="YOLO",
                    gemini_concorda=None,
                    gemini_classe=None,
                    gemini_justificativa=None,
                )
                db_persist.add(deteccao)
                db_persist.commit()
                db_persist.refresh(deteccao)
                deteccao_id = deteccao.id
                resultado_final = {
                    "deteccao_id": deteccao_id,
                    "alimento_id": alimento_yolo.id,
                    "alimento_nome": alimento_yolo.nome,
                    "classe_yolo": palpite.get("classe") if palpite else None,
                    "fonte": "YOLO",
                    "peso_padrao_kg": float(alimento_yolo.peso_padrao_kg or 0.0),
                }
            except Exception as exc:
                logger.exception("Falha ao inserir detecção preliminar")
                db_persist.rollback()
                await websocket.send_json(
                    {"tipo": "erro", "stage": "io", "mensagem": str(exc)}
                )
                estado.reset()
                continue
            finally:
                db_persist.close()

            # 6) Notifica frontend imediatamente — captura segue ininterrupta
            await websocket.send_json({
                "tipo": "deteccao_preliminar",
                "ts": int(agora * 1000),
                "deteccao_id": deteccao_id,
                "yolo": palpite_payload,
                "resultado_final": resultado_final,
                "imagem_path": imagem_path,
            })
            await websocket.send_json(
                _log_payload(
                    "resultado",
                    "Detecção preliminar registrada",
                    deteccao_id=deteccao_id,
                    alimento=resultado_final["alimento_nome"],
                    imagem_path=imagem_path,
                )
            )

            # 7) Gemini em background (fire-and-forget) ou notificação imediata se OFF
            if gemini is not None and usar_gemini:
                asyncio.create_task(
                    _processar_gemini_em_background(
                        websocket=websocket,
                        deteccao_id=deteccao_id,
                        jpeg_bytes=jpeg_bytes,
                        palpite=palpite,
                        classes_validas=classes_validas,
                        classes_db=classes_db,
                        gemini=gemini,
                    )
                )
            else:
                # Gemini desligado/indisponível: fecha o ciclo visual já como "sem_gemini"
                if usar_gemini and gemini is None:
                    await websocket.send_json(
                        _log_payload("gemini", "Gemini indisponível: GEMINI_API_KEY ausente")
                    )
                await websocket.send_json({
                    "tipo": "deteccao_atualizada",
                    "deteccao_id": deteccao_id,
                    "gemini": None,
                    "alimento_id": alimento_yolo.id,
                    "alimento_nome": None,
                    "fonte": "YOLO",
                    "ts": int(time.time() * 1000),
                })

            # Reseta estabilidade — lock continua até expirar
            estado.reset()

    except WebSocketDisconnect:
        logger.info("Cliente desconectou da sessão %s", sessao_id)
        return
    except Exception:
        logger.exception("Erro inesperado no WS /auditoria/%s", sessao_id)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
