import { API_BASE } from '../constants';

// Persiste uma detecção confirmada pelo operador na câmera. Aceita o objeto
// `ultimaDeteccao` retornado pelo hook useAuditoriaWS — inclui YOLO + Gemini.
export async function criarDeteccao({ sessaoIdNumero, deteccao, currWeight, currFood }) {
  if (!deteccao || !deteccao.resultado_final || !deteccao.resultado_final.alimento_id) {
    return { ok: false, status: 0, motivo: 'sem_alimento_id' };
  }
  const payload = {
    sessao_id: sessaoIdNumero,
    alimento_id: deteccao.resultado_final.alimento_id,
    alimento_id_original: deteccao.yolo?.alimento_id || deteccao.resultado_final.alimento_id,
    peso_kg: deteccao.resultado_final.peso_padrao_kg || currWeight,
    quantidade: 1,
    confianca: deteccao.yolo ? deteccao.yolo.confianca : null,
    imagem_path: deteccao.imagem_path || null,
    fonte: deteccao.resultado_final.fonte || 'YOLO',
    gemini_concorda: deteccao.gemini ? deteccao.gemini.concorda : null,
    gemini_classe: deteccao.gemini ? deteccao.gemini.classe : null,
    gemini_justificativa: deteccao.gemini ? deteccao.gemini.justificativa : null,
  };
  const resp = await fetch(`${API_BASE}/api/v1/deteccoes/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return { ok: resp.ok, status: resp.status, currWeight, currFood };
}

// Finaliza uma sessão no backend (status -> finalizada).
export async function finalizarSessao(sessaoIdNumero) {
  const resp = await fetch(`${API_BASE}/api/v1/sessoes/${sessaoIdNumero}/finalizar`, {
    method: 'PUT',
  });
  return { ok: resp.ok, status: resp.status };
}
