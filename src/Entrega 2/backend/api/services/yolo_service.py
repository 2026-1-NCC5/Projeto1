"""Serviço de detecção YOLO local usado pelo WebSocket /ws/auditoria.

Mantém o modelo carregado em memória (singleton) e expõe utilitários para:
- decodificar frames base64/JPEG vindos do navegador;
- rodar inferência e devolver a bbox dominante (maior confiança >= threshold).

Não tem dependência do FastAPI: pode ser invocado em script standalone para
debug (ver bloco __main__ no fim do arquivo).
"""
from __future__ import annotations

import base64
import os
from typing import Any, Optional

import numpy as np


# Lazy import de cv2/ultralytics para que módulos que apenas leem tipos não
# precisem do PyTorch/OpenCV instalado (ex.: tooling de schema).
def _lazy_imports():
    import cv2  # noqa: WPS433  (import dentro de função é intencional)
    from ultralytics import YOLO  # noqa: WPS433
    return cv2, YOLO


class YoloService:
    """Wrapper fino sobre Ultralytics YOLO."""

    def __init__(self, model_path: str, conf_threshold: float = 0.35):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo YOLO não encontrado em {model_path}")

        cv2, YOLO = _lazy_imports()
        self._cv2 = cv2
        self.model = YOLO(model_path, task="detect")
        self.classes = self.model.names  # dict[int, str]
        self.conf_threshold = conf_threshold

    # ------------------------- helpers de imagem ------------------------- #
    def decodificar_b64(self, imagem_b64: str) -> Optional[np.ndarray]:
        """Aceita data URL (`data:image/jpeg;base64,...`) ou base64 puro."""
        if not imagem_b64:
            return None
        if "," in imagem_b64:
            imagem_b64 = imagem_b64.split(",", 1)[1]
        try:
            buf = np.frombuffer(base64.b64decode(imagem_b64), dtype=np.uint8)
        except Exception:
            return None
        return self._cv2.imdecode(buf, self._cv2.IMREAD_COLOR)

    def codificar_jpeg(self, frame: np.ndarray, qualidade: int = 85) -> bytes:
        """Codifica frame BGR em JPEG bytes (utilizado para enviar ao Gemini e gravar evidência)."""
        ok, buf = self._cv2.imencode(
            ".jpg", frame, [self._cv2.IMWRITE_JPEG_QUALITY, qualidade]
        )
        if not ok:
            raise RuntimeError("Falha ao codificar JPEG")
        return buf.tobytes()

    # ----------------------------- detecção ----------------------------- #
    def detectar_dominante(self, frame: np.ndarray) -> Optional[dict[str, Any]]:
        """Retorna a melhor detecção (maior confiança) acima do threshold ou None."""
        if frame is None:
            return None

        result = self.model(frame, verbose=False)[0]
        boxes = result.boxes
        melhor: Optional[dict[str, Any]] = None

        for box in boxes:
            conf = float(box.conf.item())
            if conf < self.conf_threshold:
                continue
            cls_id = int(box.cls.item())
            xyxy = box.xyxy.cpu().numpy().squeeze().astype(int).tolist()
            candidato = {
                "classe": self.classes[cls_id],
                "confianca": conf,
                "bbox": [int(v) for v in xyxy],  # [x1, y1, x2, y2]
            }
            if melhor is None or candidato["confianca"] > melhor["confianca"]:
                melhor = candidato

        return melhor


# ---- Singleton lazy (evita carregar PyTorch quando outro módulo só importa tipos) ----
_singleton: Optional[YoloService] = None


def get_yolo_service() -> YoloService:
    """Retorna o YoloService configurado (carregando o modelo na primeira chamada)."""
    global _singleton
    if _singleton is None:
        # Import local para quebrar ciclo (config -> services -> ...)
        from api.core.config import settings
        _singleton = YoloService(settings.YOLO_MODEL_PATH, settings.YOLO_CONF_THRESHOLD)
    return _singleton


# ----------------- CLI de debug: python -m api.services.yolo_service caminho.jpg --
if __name__ == "__main__":  # pragma: no cover
    import sys
    import json
    if len(sys.argv) < 2:
        print("Uso: python -m api.services.yolo_service caminho_da_imagem.jpg")
        sys.exit(1)

    cv2, _ = _lazy_imports()
    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Falha ao ler imagem: {sys.argv[1]}")
        sys.exit(1)

    svc = get_yolo_service()
    print(json.dumps(svc.detectar_dominante(img), indent=2, ensure_ascii=False))
