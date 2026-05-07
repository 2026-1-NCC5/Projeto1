"""Camada de validação/fallback usando a API Gemini Flash.

O serviço é chamado em todo gatilho de detecção: recebe o frame JPEG do
momento em que o objeto ficou estável (ROI já isolada pelo YOLO) e devolve uma
resposta JSON estruturada com a classificação confirmada/corrigida.

Doc oficial: https://ai.google.dev/gemini-api/docs/image-understanding
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = """Voce e um auditor de alimentos arrecadados. Analise a imagem e identifique
o pacote alimenticio em primeiro plano (no centro do quadro).

CLASSES VALIDAS (use exatamente estes nomes nos campos `classe`):
{lista_classes}

PALPITE PRELIMINAR DO MODELO LOCAL (YOLO):
- classe: {yolo_classe}
- confianca: {yolo_confianca}

INSTRUCOES:
1. Confirme ou corrija o palpite do YOLO observando a imagem.
2. Se nao houver pacote claramente visivel, retorne classe=null e
   concorda=false.
3. Se o palpite do YOLO estiver correto, concorda=true e classe=palpite.
4. Se o palpite estiver errado mas voce identificar outro item da lista
   de classes validas, concorda=false e classe=item correto.
5. Se nao for nenhuma das classes validas, classe=null, concorda=false e
   justifique.
6. Confianca qualitativa: "alta" (item nitido e inequivoco), "media"
   (parcialmente visivel ou marca duvidosa), "baixa" (palpite especulativo).

RESPONDA SOMENTE COM JSON VALIDO no formato:
{{
  "classe": "<nome da classe valida ou null>",
  "concorda": <true|false>,
  "confianca_qualitativa": "alta" | "media" | "baixa",
  "justificativa": "<frase curta em portugues, max 200 caracteres>"
}}
"""


def _lazy_import_genai():
    # Import dentro de função para não quebrar quando o pacote ainda não estiver
    # instalado em ambientes sem Gemini (ex.: smoke check sem GEMINI_API_KEY).
    from google import genai
    from google.genai import types
    return genai, types


class GeminiService:
    """Camada fina sobre google-genai para validar a classe predita pelo YOLO."""

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY não configurada — o serviço Gemini está desabilitado"
            )
        genai, types = _lazy_import_genai()
        self._types = types
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def _montar_prompt(
        self, palpite: Optional[dict[str, Any]], classes_validas: list[str]
    ) -> str:
        if palpite:
            yolo_classe = palpite.get("classe", "nenhum")
            yolo_conf = f"{palpite.get('confianca', 0):.2f}"
        else:
            yolo_classe = "nenhum"
            yolo_conf = "n/a"
        return PROMPT_TEMPLATE.format(
            lista_classes=", ".join(classes_validas) or "(nenhuma cadastrada)",
            yolo_classe=yolo_classe,
            yolo_confianca=yolo_conf,
        )

    def validar(
        self,
        frame_jpeg_bytes: bytes,
        palpite_yolo: Optional[dict[str, Any]],
        classes_validas: list[str],
    ) -> Optional[dict[str, Any]]:
        """Valida o palpite do YOLO; em caso de falha, retorna None.

        Estrutura esperada do retorno (compatível com o WebSocket):
        ``{ "classe": str|None, "concorda": bool, "confianca_qualitativa": str,
            "justificativa": str }``
        """
        prompt = self._montar_prompt(palpite_yolo, classes_validas)
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    self._types.Part.from_bytes(
                        data=frame_jpeg_bytes, mime_type="image/jpeg"
                    ),
                    prompt,
                ],
                config=self._types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )
        except Exception as exc:  # noqa: BLE001 — qualquer erro vira None
            logger.warning("Gemini falhou: %s", exc)
            return None

        texto = (response.text or "").strip()
        if not texto:
            return None
        try:
            data = json.loads(texto)
        except json.JSONDecodeError:
            logger.warning("Gemini retornou JSON inválido: %r", texto[:200])
            return None

        # Sanitiza: garante chaves esperadas
        return {
            "classe": data.get("classe"),
            "concorda": bool(data.get("concorda", False)),
            "confianca_qualitativa": data.get("confianca_qualitativa", "baixa"),
            "justificativa": str(data.get("justificativa") or "")[:500],
        }


# ---------------- Singleton lazy ----------------
_singleton: Optional[GeminiService] = None


def get_gemini_service() -> Optional[GeminiService]:
    """Retorna o serviço Gemini, ou None se a chave não estiver configurada.

    Permite que o pipeline funcione em modo degradado (YOLO puro) quando o
    Gemini não está disponível — útil para PoC sem créditos.
    """
    global _singleton
    if _singleton is not None:
        return _singleton
    from api.core.config import settings
    if not settings.GEMINI_API_KEY:
        return None
    try:
        _singleton = GeminiService(settings.GEMINI_API_KEY, settings.GEMINI_MODEL)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Não foi possível inicializar Gemini: %s", exc)
        _singleton = None
    return _singleton
