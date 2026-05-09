import { API_BASE } from '../constants';

let onUnauthorized = () => {};

export function setApiUnauthorizedHandler(fn) {
  onUnauthorized = typeof fn === 'function' ? fn : () => {};
}

async function readBody(resp) {
  const text = await resp.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text };
  }
}

function pathDisparaRedirecionario401(path) {
  if (!path.startsWith('/api/v1/auth/')) return true;
  if (path.startsWith('/api/v1/auth/me')) return false;
  if (path.startsWith('/api/v1/auth/login')) return false;
  if (path.startsWith('/api/v1/auth/cadastro-admin')) return false;
  if (path.startsWith('/api/v1/auth/logout')) return false;
  return true;
}

async function fetchJson(path, options = {}) {
  const resp = await fetch(`${API_BASE}${path}`, {
    credentials: 'include',
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  if (resp.status === 401 && pathDisparaRedirecionario401(path)) {
    onUnauthorized();
  }
  const data = await readBody(resp);
  return { ok: resp.ok, status: resp.status, data };
}

/**
 * Requisição genérica à API — retorna { ok, status, data }.
 */
export async function apiRequest(path, options = {}) {
  return fetchJson(path, options);
}

export async function authCadastroAdmin({ nome, email }) {
  return fetchJson('/api/v1/auth/cadastro-admin', {
    method: 'POST',
    body: JSON.stringify({ nome, email }),
  });
}

export async function authLogin({ nome, email }) {
  return fetchJson('/api/v1/auth/login', {
    method: 'POST',
    body: JSON.stringify({ nome, email }),
  });
}

export async function authMe() {
  return fetchJson('/api/v1/auth/me');
}

export async function authLogout() {
  return fetchJson('/api/v1/auth/logout', { method: 'POST' });
}

export async function listarGrupos() {
  return apiRequest('/api/v1/grupos/');
}

export async function criarGrupo(body) {
  return apiRequest('/api/v1/grupos/', { method: 'POST', body: JSON.stringify(body) });
}

export async function atualizarGrupo(id, body) {
  return apiRequest(`/api/v1/grupos/${id}`, { method: 'PUT', body: JSON.stringify(body) });
}

export async function excluirGrupo(id) {
  return apiRequest(`/api/v1/grupos/${id}`, { method: 'DELETE' });
}

export async function listarSessoes() {
  return apiRequest('/api/v1/sessoes/');
}

export async function criarSessao({ grupo_id }) {
  return apiRequest('/api/v1/sessoes/', {
    method: 'POST',
    body: JSON.stringify({ grupo_id }),
  });
}

/**
 * Cria sessão ou reaproveita a ativa do grupo (quando o backend retorna 400).
 * @returns {{ sessaoId: number, reutilizada?: boolean, erro?: string }}
 */
export async function obterOuCriarSessaoAtiva(grupoId) {
  const criado = await criarSessao({ grupo_id: grupoId });
  if (criado.ok && criado.data?.id != null) {
    return { sessaoId: criado.data.id, reutilizada: false };
  }
  if (criado.status === 400) {
    const lista = await listarSessoes();
    if (lista.ok && Array.isArray(lista.data)) {
      const ativa = lista.data.find(
        (s) => s.grupo_id === grupoId && s.status === 'ativa',
      );
      if (ativa) {
        return { sessaoId: ativa.id, reutilizada: true };
      }
    }
  }
  const det = criado.data?.detail;
  const msg = typeof det === 'string' ? det : Array.isArray(det) ? JSON.stringify(det) : 'Não foi possível abrir sessão';
  return { erro: msg };
}

export async function listarAlimentos() {
  return apiRequest('/api/v1/alimentos/');
}

export async function relatorioGrupos() {
  return apiRequest('/api/v1/relatorios/grupos');
}

export async function relatorioCategorias() {
  return apiRequest('/api/v1/relatorios/categorias');
}

// Persiste uma detecção confirmada pelo operador na câmera. Aceita o objeto
// `ultimaDeteccao` retornado pelo hook useAuditoriaWS — inclui YOLO + Gemini.
//
// Não é mais usado no fluxo automático da câmera (a partir do refactor para
// pipeline assíncrono, o backend persiste sozinho via WS — INSERT preliminar
// + UPDATE pós-Gemini). Mantido por enquanto para futuros usos manuais ou
// testes; avaliar remoção após estabilizar o fluxo WS-only.
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
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (resp.status === 401) onUnauthorized();
  return { ok: resp.ok, status: resp.status, currWeight, currFood };
}

/**
 * Registro manual de detecção (inserção manual na triagem).
 */
export async function criarDeteccaoManual({
  sessaoIdNumero,
  alimentoId,
  peso_kg,
  alimentoIdOriginal,
  fonte = 'MANUAL',
}) {
  if (!sessaoIdNumero || !alimentoId) {
    return { ok: false, status: 0, motivo: 'sessao_ou_alimento' };
  }
  const payload = {
    sessao_id: sessaoIdNumero,
    alimento_id: alimentoId,
    alimento_id_original: alimentoIdOriginal ?? alimentoId,
    peso_kg,
    quantidade: 1,
    confianca: null,
    imagem_path: null,
    fonte,
    gemini_concorda: null,
    gemini_classe: null,
    gemini_justificativa: null,
  };
  const resp = await fetch(`${API_BASE}/api/v1/deteccoes/`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (resp.status === 401) onUnauthorized();
  return { ok: resp.ok, status: resp.status };
}

/** Correção manual de alimento e/ou peso de uma detecção existente. */
export async function corrigirDeteccao(deteccaoId, { alimentoId, pesoKg } = {}) {
  const body = {};
  if (alimentoId != null) body.alimento_id = alimentoId;
  if (pesoKg != null) body.peso_kg = pesoKg;
  return fetchJson(`/api/v1/deteccoes/${deteccaoId}/correcao`, {
    method: 'PUT',
    body: JSON.stringify(body),
  });
}

export async function excluirDeteccao(deteccaoId) {
  const resp = await fetch(`${API_BASE}/api/v1/deteccoes/${deteccaoId}`, {
    method: 'DELETE',
    credentials: 'include',
  });
  if (resp.status === 401) onUnauthorized();
  return { ok: resp.ok, status: resp.status };
}

// Finaliza uma sessão no backend (status -> finalizada).
export async function finalizarSessao(sessaoIdNumero) {
  const resp = await fetch(`${API_BASE}/api/v1/sessoes/${sessaoIdNumero}/finalizar`, {
    method: 'PUT',
    credentials: 'include',
  });
  if (resp.status === 401) onUnauthorized();
  return { ok: resp.ok, status: resp.status };
}

export async function obterConciliacaoPreviaSessao(sessaoIdNumero) {
  return apiRequest(`/api/v1/sessoes/${sessaoIdNumero}/conciliacao-previa`);
}

export async function decidirFonteFinalSessao({ sessaoIdNumero, fonteFinal }) {
  return apiRequest(`/api/v1/sessoes/${sessaoIdNumero}/decisao-final`, {
    method: 'PUT',
    body: JSON.stringify({
      fonte_final: fonteFinal,
    }),
  });
}
