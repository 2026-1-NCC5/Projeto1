"""Camada de validação/fallback usando a API Gemini Flash.

O serviço é chamado em todo gatilho de detecção: recebe o frame JPEG do
momento em que o objeto ficou estável (ROI já isolada pelo YOLO) e devolve uma
resposta JSON estruturada com a classificação confirmada/corrigida e o peso
líquido inferido da embalagem quando visível.

Doc oficial: https://ai.google.dev/gemini-api/docs/image-understanding
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = """Voce e um auditor de alimentos arrecadados. Analise a imagem e identifique
o pacote alimenticio em primeiro plano (no centro do quadro).

CLASSES VALIDAS (use exatamente estes nomes no campo `classe` quando aplicavel):
{lista_classes}

PALPITE PRELIMINAR DO MODELO LOCAL (YOLO):
- classe: {yolo_classe}
- confianca: {yolo_confianca}

PESO DE REFERENCIA ATUAL (cadastro / modelo, em kg — pode estar errado frente a embalagem):
{peso_referencia_kg}

INSTRUCOES:
1. Confirme ou corrija o palpite do YOLO observando a imagem (`concorda_classe`).
2. Leia na embalagem a massa liquida / peso neto do produto (ex.: "500g", "1 kg").
   - Converta para quilogramas no campo `peso_kg` (ex.: 500 g -> 0.5).
   - Se nao conseguir ler, use `peso_kg`: null e explique em `justificativa`.
3. Compare `peso_kg` com o PESO DE REFERENCIA acima. Se diferenca relevante (> ~20 g),
   `concorda_peso` = false; senao true.
4. Se nao houver pacote claramente visivel, retorne classe=null, concorda_classe=false,
   peso_kg=null, concorda_peso=false.
5. Se o palpite de classe estiver correto, concorda_classe=true e classe=palpite.
6. Se estiver errado mas voce identificar outro item da lista, concorda_classe=false e classe=item correto.
7. Confianca qualitativa: "alta" | "media" | "baixa".
8. Se nao conseguir ler a massa liquida na embalagem (borrado, cortado, fora do quadro,
   reflexo, etc.), use `peso_kg`: null, `concorda_peso`: false e `alerta_revisao_manual`: true.
   Se leu o peso com confianca, `alerta_revisao_manual`: false.

O campo legado `concorda` deve ser true somente se concorda_classe E concorda_peso forem true.

RESPONDA SOMENTE COM JSON VALIDO no formato:
{{
  "classe": "<nome da classe valida ou null>",
  "concorda_classe": <true|false>,
  "concorda_peso": <true|false>,
  "peso_kg": <number ou null>,
  "alerta_revisao_manual": <true|false>,
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
        self,
        palpite: Optional[dict[str, Any]],
        classes_validas: list[str],
        peso_referencia_kg: Optional[float],
    ) -> str:
        if palpite:
            yolo_classe = palpite.get("classe", "nenhum")
            yolo_conf = f"{palpite.get('confianca', 0):.2f}"
        else:
            yolo_classe = "nenhum"
            yolo_conf = "n/a"
        pr = peso_referencia_kg
        peso_ref_str = f"{pr:.3f} kg" if pr is not None else "(desconhecido)"
        return PROMPT_TEMPLATE.format(
            lista_classes=", ".join(classes_validas) or "(nenhuma cadastrada)",
            yolo_classe=yolo_classe,
            yolo_confianca=yolo_conf,
            peso_referencia_kg=peso_ref_str,
        )

    def validar(
        self,
        frame_jpeg_bytes: bytes,
        palpite_yolo: Optional[dict[str, Any]],
        classes_validas: list[str],
        peso_referencia_kg: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        """Valida o palpite do YOLO e infere peso na embalagem; em caso de falha, None.

        Retorno compatível com o WebSocket; inclui chaves novas e legado `concorda`.
        """
        prompt = self._montar_prompt(palpite_yolo, classes_validas, peso_referencia_kg)
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

        concorda_classe = bool(data.get("concorda_classe", data.get("concorda", False)))

        raw_peso = data.get("peso_kg")
        peso_kg: Optional[float]
        if raw_peso is None or raw_peso == "":
            peso_kg = None
        else:
            try:
                peso_kg = float(raw_peso)
            except (TypeError, ValueError):
                peso_kg = None

        alerta_revisao_manual = bool(data.get("alerta_revisao_manual", False))

        concorda_peso = data.get("concorda_peso")
        if concorda_peso is None:
            concorda_peso = peso_kg is not None and not alerta_revisao_manual
        else:
            concorda_peso = bool(concorda_peso)
        if peso_kg is None or alerta_revisao_manual:
            concorda_peso = False

        concorda = bool(data.get("concorda", concorda_classe and concorda_peso))

        return {
            "classe": data.get("classe"),
            "concorda_classe": concorda_classe,
            "concorda_peso": concorda_peso,
            "peso_kg": peso_kg,
            "alerta_revisao_manual": alerta_revisao_manual,
            "concorda": concorda,
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
