"""WebSocket /ws/auditoria/{sessao_id}: pipeline câmera → YOLO → Gemini.

Mensagens cliente → servidor:
- {"tipo": "frame", "ts": <ms>, "imagem_b64": "data:image/jpeg;base64,..."}
- {"tipo": "reset"} (libera lock cedo, reinicia estabilidade)

Mensagens servidor → cliente:
- {"tipo": "status", "estado": "monitorando"|"estavel"|"analisando"|"lock", "lock_ate_ts": <ms>}
- {"tipo": "deteccao", ... payload completo (ver plano §3.2) ...}
- {"tipo": "erro", "stage": "yolo"|"gemini"|"io", "mensagem": "..."}

Não exige autenticação (compatível com `POST /api/v1/deteccoes/`) para
simplificar a integração da câmera no PoC.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.core.config import settings
from api.dependencies import get_db
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
from bd.models.sessao import Sessao

logger = logging.getLogger(__name__)
router = APIRouter()


def _resolver_classe_final(
    palpite_yolo: Optional[dict[str, Any]],
    gemini_out: Optional[dict[str, Any]],
) -> tuple[Optional[str], str]:
    """Decide a classe final + a fonte (YOLO|GEMINI|DESCONHECIDO).

    Regras (ordem importa):
    1. YOLO + Gemini concordando -> classe do YOLO, fonte=YOLO.
    2. Gemini propôs uma classe (concorda=False ou YOLO sem palpite) -> classe=Gemini, fonte=GEMINI.
    3. Apenas YOLO -> classe=YOLO, fonte=YOLO.
    4. Nada -> (None, DESCONHECIDO).
    """
    if palpite_yolo and gemini_out and gemini_out.get("concorda"):
        return palpite_yolo["classe"], "YOLO"
    if gemini_out and gemini_out.get("classe"):
        return gemini_out["classe"], "GEMINI"
    if palpite_yolo:
        return palpite_yolo["classe"], "YOLO"
    return None, "DESCONHECIDO"


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


@router.websocket("/auditoria/{sessao_id}")
async def ws_auditoria(websocket: WebSocket, sessao_id: int):
    await websocket.accept()

    # Valida sessão antes de aceitar tráfego pesado
    db_gen = get_db()
    db = next(db_gen)
    try:
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
        classes_db = _hidratar_classes(db)
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass

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

            # 2) Lock imediato anti-duplicidade
            estado.liberar_lock_em(settings.LOCK_SECONDS)
            await websocket.send_json({
                "tipo": "status",
                "estado": "analisando",
                "lock_ate_ts": int(estado.lock_ate * 1000),
            })
            await websocket.send_json(
                _log_payload(
                    "yolo",
                    "Objeto estável, iniciando análise",
                    classe=palpite.get("classe") if palpite else None,
                    confianca=palpite.get("confianca") if palpite else None,
                    gemini_ativado=usar_gemini and gemini is not None,
                )
            )

            # 3) Codifica JPEG para Gemini + evidência
            try:
                jpeg_bytes = yolo.codificar_jpeg(frame, qualidade=85)
            except Exception as exc:
                logger.exception("Falha ao codificar JPEG")
                await websocket.send_json(
                    {"tipo": "erro", "stage": "io", "mensagem": str(exc)}
                )
                estado.reset()
                continue

            # 4) Gemini (em thread separada para não bloquear o event loop)
            gemini_out: Optional[dict[str, Any]] = None
            if gemini is not None and usar_gemini:
                try:
                    gemini_out = await asyncio.to_thread(
                        gemini.validar, jpeg_bytes, palpite, classes_validas
                    )
                    await websocket.send_json(
                        _log_payload(
                            "gemini",
                            "Gemini respondeu",
                            classe=gemini_out.get("classe") if gemini_out else None,
                            concorda=gemini_out.get("concorda") if gemini_out else None,
                        )
                    )
                except Exception as exc:
                    logger.warning("Gemini falhou: %s", exc)
                    gemini_out = None
                    await websocket.send_json(
                        {"tipo": "erro", "stage": "gemini", "mensagem": str(exc)}
                    )
            elif usar_gemini and gemini is None:
                await websocket.send_json(
                    _log_payload("gemini", "Gemini indisponível: GEMINI_API_KEY ausente")
                )

            # 5) Persistência da evidência visual
            imagem_path = salvar_evidencia(sessao_id, jpeg_bytes)

            # 6) Resolução final + lookup de alimento
            classe_final, fonte = _resolver_classe_final(palpite, gemini_out)
            alimento = classes_db.get(classe_final) if classe_final else None
            resultado_final = {
                "alimento_id": alimento.id if alimento else None,
                "alimento_nome": alimento.nome if alimento else "(desconhecido)",
                "classe_yolo": classe_final,
                "fonte": fonte,
                "peso_padrao_kg": float(alimento.peso_padrao_kg) if alimento else 0.0,
            }

            await websocket.send_json({
                "tipo": "deteccao",
                "ts": int(agora * 1000),
                "yolo": palpite_payload,
                "gemini": gemini_out,
                "resultado_final": resultado_final,
                "imagem_path": imagem_path,
            })
            await websocket.send_json(
                _log_payload(
                    "resultado",
                    "Detecção consolidada enviada",
                    alimento=resultado_final["alimento_nome"],
                    fonte=fonte,
                    imagem_path=imagem_path,
                )
            )

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
