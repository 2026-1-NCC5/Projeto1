import { useEffect, useState } from 'react';

const STORAGE_KEY = 'abraceai_appState';
const SESSAO_KEY = 'abraceai_sessao_id';

const DEFAULT_APP_STATE = [];

function normalizeLoaded(arr) {
  if (!Array.isArray(arr)) return DEFAULT_APP_STATE;
  return arr.map((g) => {
    if (g.grupoIdBackend != null) {
      return { ...g, id: String(g.id) };
    }
    const n = Number(g.id);
    if (!Number.isNaN(n) && String(n) === String(g.id).trim()) {
      return { ...g, id: String(n), grupoIdBackend: n };
    }
    return { ...g, id: String(g.id) };
  });
}

// Mantém o array de grupos (`appState`) sincronizado com localStorage.
// Shape: [{ id, title, members, totalKg, items, grupoIdBackend? }].
// id é sempre string; grupoIdBackend amarra ao GET /api/v1/grupos.
// Usa lazy initializer para evitar setState dentro de useEffect na hidratação.
export function usePersistedAppState() {
  const [appState, setAppState] = useState(() => {
    try {
      const saved = typeof window !== 'undefined' ? localStorage.getItem(STORAGE_KEY) : null;
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) return normalizeLoaded(parsed);
      }
    } catch (e) { console.warn('Erro ao carregar dados locais:', e); }
    return DEFAULT_APP_STATE;
  });

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(appState));
    } catch (e) { console.warn('Erro ao salvar dados locais:', e); }
  }, [appState]);

  return [appState, setAppState];
}

// ID da Sessao no backend (aberta via POST /api/v1/sessoes ao entrar na câmera/manual).
// Persistido em localStorage para PoC.
export function usePersistedSessaoId() {
  const [auditSessaoId, setAuditSessaoId] = useState(() => {
    const saved = typeof window !== 'undefined' ? localStorage.getItem(SESSAO_KEY) : null;
    return saved ?? '';
  });

  useEffect(() => {
    try {
      if (auditSessaoId === '' || auditSessaoId == null) {
        localStorage.removeItem(SESSAO_KEY);
      } else {
        localStorage.setItem(SESSAO_KEY, String(auditSessaoId));
      }
    } catch { /* noop */ }
  }, [auditSessaoId]);

  return [auditSessaoId, setAuditSessaoId];
}
