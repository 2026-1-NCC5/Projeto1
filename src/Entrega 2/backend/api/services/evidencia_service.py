"""Persistência de imagens de evidência (JPEG) por sessão.

Para o PoC gravamos no filesystem local em ``backend/evidencias/{sessao_id}/``.
A função retorna um caminho relativo legível para gravar em ``deteccoes.imagem_path``
e que casa com o mount StaticFiles em ``api/main.py`` (rota ``/evidencias/...``).
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from api.core.config import settings


def _slug_timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S_%f")[:-3]


def salvar_evidencia(sessao_id: int, jpeg_bytes: bytes) -> Optional[str]:
    """Grava bytes JPEG em disco e devolve o caminho relativo (servido via StaticFiles).

    Retorna ``None`` em caso de falha — o pipeline segue sem evidência mas a
    detecção continua sendo registrada.
    """
    if not jpeg_bytes:
        return None
    try:
        pasta = os.path.join(settings.EVIDENCIA_DIR, str(sessao_id))
        os.makedirs(pasta, exist_ok=True)
        nome_arquivo = f"{_slug_timestamp()}.jpg"
        caminho_abs = os.path.join(pasta, nome_arquivo)
        with open(caminho_abs, "wb") as f:
            f.write(jpeg_bytes)
        # Caminho relativo a settings.EVIDENCIA_DIR — usado para montar a URL
        # pública: f"/evidencias/{sessao_id}/{nome_arquivo}".
        return f"evidencias/{sessao_id}/{nome_arquivo}"
    except OSError:
        return None
