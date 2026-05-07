"""Gatilho por estabilidade + lock anti-duplicidade.

Cada conexão WebSocket mantém um EstadoSessao em memória. A bbox dominante do
YOLO é comparada com a anterior por IoU; quando a sobreposição passa de
``iou_min`` por ``estabilidade_seg`` segundos consecutivos, dispara inferência
e ativa lock por ``LOCK_SECONDS`` para evitar duplicidades.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

Bbox = Tuple[int, int, int, int]


@dataclass
class EstadoSessao:
    ultima_bbox: Optional[Bbox] = None
    ts_inicio_estavel: Optional[float] = None
    ts_inicio_sem_yolo: Optional[float] = None
    lock_ate: float = 0.0

    def em_lock(self) -> bool:
        return time.time() < self.lock_ate

    def liberar_lock_em(self, segundos: float) -> None:
        self.lock_ate = time.time() + segundos

    def reset(self) -> None:
        """Reseta apenas o estado de estabilidade; lock continua até expirar."""
        self.ultima_bbox = None
        self.ts_inicio_estavel = None
        self.ts_inicio_sem_yolo = None

    def reset_total(self) -> None:
        """Reset completo (utilizado pelo evento `reset` vindo do cliente)."""
        self.reset()
        self.lock_ate = 0.0


def iou(a: Optional[Bbox], b: Optional[Bbox]) -> float:
    """Intersection over Union para duas bboxes [x1, y1, x2, y2]."""
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
    b_area = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
    uniao = a_area + b_area - inter
    if uniao <= 0:
        return 0.0
    return inter / uniao


def avaliar_gatilho(
    estado: EstadoSessao,
    bbox_atual: Optional[Bbox],
    agora: float,
    estabilidade_seg: float,
    iou_min: float,
) -> bool:
    """Atualiza o estado da sessão e devolve True quando deve disparar análise.

    Mecânica:
    1. Se está em lock OU não há objeto detectado, ignora.
    2. Se não há bbox anterior OU a sobreposição caiu abaixo do limiar, reinicia o
       cronômetro de estabilidade.
    3. Se objeto está consistente há `estabilidade_seg` segundos, dispara.
    """
    if estado.em_lock():
        return False

    if bbox_atual is None:
        # Objeto desapareceu -> reinicia mas mantém qualquer lock vigente
        estado.ultima_bbox = None
        estado.ts_inicio_estavel = None
        return False

    estado.ts_inicio_sem_yolo = None

    if estado.ultima_bbox is None or iou(estado.ultima_bbox, bbox_atual) < iou_min:
        estado.ultima_bbox = bbox_atual
        estado.ts_inicio_estavel = agora
        return False

    # Mesma região há tempo suficiente -> dispara
    if (agora - (estado.ts_inicio_estavel or agora)) >= estabilidade_seg:
        return True

    return False


def avaliar_fallback_sem_yolo(
    estado: EstadoSessao,
    agora: float,
    estabilidade_seg: float,
) -> bool:
    """Dispara Gemini como fallback após alguns frames consecutivos sem palpite YOLO."""
    if estado.em_lock():
        return False

    estado.ultima_bbox = None
    estado.ts_inicio_estavel = None
    if estado.ts_inicio_sem_yolo is None:
        estado.ts_inicio_sem_yolo = agora
        return False

    return (agora - estado.ts_inicio_sem_yolo) >= estabilidade_seg


def descrever_estado(estado: EstadoSessao) -> str:
    """Helper textual usado nas mensagens de status do WebSocket."""
    if estado.em_lock():
        return "lock"
    if estado.ts_inicio_estavel is not None and estado.ultima_bbox is not None:
        return "estavel"
    if estado.ts_inicio_sem_yolo is not None:
        return "estavel"
    return "monitorando"
