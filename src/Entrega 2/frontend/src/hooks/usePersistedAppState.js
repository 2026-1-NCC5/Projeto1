import { useEffect, useState } from 'react';

const STORAGE_KEY = 'abraceai_appState';
const SESSAO_KEY = 'abraceai_sessao_id';

const DEFAULT_APP_STATE = [
  { id: 'A', title: 'Grupo A', members: [{ name: 'Maria Silva', ra: '12345' }, { name: 'João P.', ra: '54321' }], totalKg: 0, items: [] },
  { id: 'B', title: 'Grupo B', members: [], totalKg: 0, items: [] }
];

// Mantém o array de grupos (`appState`) sincronizado com localStorage.
// Shape preservado de versões anteriores: [{ id, title, members, totalKg, items }].
// Usa lazy initializer para evitar setState dentro de useEffect na hidratação.
export function usePersistedAppState() {
  const [appState, setAppState] = useState(() => {
    try {
      const saved = typeof window !== 'undefined' ? localStorage.getItem(STORAGE_KEY) : null;
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) return parsed;
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

// ID da Sessao no backend (criada via /api/v1/sessoes). Pode ser ajustado no
// header da tela de câmera. Persistido em localStorage para PoC.
export function usePersistedSessaoId() {
  const [auditSessaoId, setAuditSessaoId] = useState(() => {
    const saved = typeof window !== 'undefined' ? localStorage.getItem(SESSAO_KEY) : null;
    return saved || '1';
  });

  useEffect(() => {
    try {
      localStorage.setItem(SESSAO_KEY, String(auditSessaoId));
    } catch { /* noop */ }
  }, [auditSessaoId]);

  return [auditSessaoId, setAuditSessaoId];
}
