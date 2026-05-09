import React, { useMemo, useState, useEffect, useCallback } from 'react';
import useToasts from '../hooks/useToasts';
import { usePersistedAppState, usePersistedSessaoId } from '../hooks/usePersistedAppState';
import { AppStateContext } from './appStateContextValue';
import { authMe, authLogout, setApiUnauthorizedHandler } from '../services/api';

function adminParaUserData(admin) {
  if (!admin) {
    return { nome: '', sobrenome: '', email: '', ra: '' };
  }
  const partes = String(admin.nome || '').trim().split(/\s+/).filter(Boolean);
  const nome = partes[0] || '';
  const sobrenome = partes.slice(1).join(' ');
  return {
    nome,
    sobrenome,
    email: admin.email || '',
    ra: admin.id != null ? String(admin.id) : '',
  };
}

// Provider único compartilhado por todas as telas. Cada tela deve consumir só
// o que precisa via `const { ... } = useAppState();` (em ./appStateContextValue).
export function AppStateProvider({ children }) {
  const [currentScreen, setCurrentScreen] = useState('home');
  const [appState, setAppState] = usePersistedAppState();
  const [auditSessaoId, setAuditSessaoId] = usePersistedSessaoId();
  const [activeGroupId, setActiveGroupId] = useState(null);
  const [manualSomenteAcrescentar, setManualSomenteAcrescentar] = useState(false);
  const [authUsuario, setAuthUsuario] = useState(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [userData, setUserData] = useState(() => adminParaUserData(null));
  const { toasts, addToast } = useToasts();

  const refreshMe = useCallback(async () => {
    const r = await authMe();
    if (r.ok && r.data?.id != null) {
      setAuthUsuario(r.data);
      setUserData(adminParaUserData(r.data));
      return true;
    }
    setAuthUsuario(null);
    return false;
  }, []);

  const logout = useCallback(async () => {
    try {
      await authLogout();
    } catch { /* noop */ }
    setAuthUsuario(null);
    setUserData(adminParaUserData(null));
    setCurrentScreen('login');
  }, []);

  useEffect(() => {
    setApiUnauthorizedHandler(() => {
      setAuthUsuario(null);
      setUserData(adminParaUserData(null));
      setCurrentScreen('login');
      addToast('Sessão expirada. Faça login novamente.', 'warning');
    });
  }, [addToast]);

  useEffect(() => {
    let cancel = false;
    (async () => {
      setAuthLoading(true);
      await refreshMe();
      if (!cancel) setAuthLoading(false);
    })();
    return () => { cancel = true; };
  }, [refreshMe]);

  const auditSessaoIdParsed = Number(auditSessaoId);
  const auditSessaoIdNumero = Number.isFinite(auditSessaoIdParsed) && auditSessaoIdParsed > 0
    ? auditSessaoIdParsed
    : 0;

  const value = useMemo(() => ({
    currentScreen, setCurrentScreen,
    appState, setAppState,
    activeGroupId, setActiveGroupId,
    manualSomenteAcrescentar, setManualSomenteAcrescentar,
    auditSessaoId, setAuditSessaoId, auditSessaoIdNumero,
    userData, setUserData,
    authUsuario, authLoading, refreshMe, logout,
    toasts, addToast,
  }), [
    currentScreen, appState, activeGroupId, manualSomenteAcrescentar, auditSessaoId, auditSessaoIdNumero,
    userData, authUsuario, authLoading, toasts, addToast,
    refreshMe, logout, setAppState, setAuditSessaoId,
  ]);

  return (
    <AppStateContext.Provider value={value}>
      {children}
    </AppStateContext.Provider>
  );
}
