import React, { useMemo, useState } from 'react';
import useToasts from '../hooks/useToasts';
import { usePersistedAppState, usePersistedSessaoId } from '../hooks/usePersistedAppState';
import useRealtimeSocket from '../hooks/useRealtimeSocket';
import { AppStateContext } from './appStateContextValue';

// Provider único compartilhado por todas as telas. Cada tela deve consumir só
// o que precisa via `const { ... } = useAppState();` (em ./appStateContextValue).
export function AppStateProvider({ children }) {
  const [currentScreen, setCurrentScreen] = useState('home');
  const [appState, setAppState] = usePersistedAppState();
  const [auditSessaoId, setAuditSessaoId] = usePersistedSessaoId();
  // Grupo selecionado pelo Kanban antes de abrir Câmera ou Inserção Manual.
  const [activeGroupId, setActiveGroupId] = useState(null);
  const [userData, setUserData] = useState({
    nome: 'Admin',
    sobrenome: 'Central',
    email: 'admin@abraceai.com.br',
    ra: '00000000'
  });
  const { toasts, addToast } = useToasts();

  // Realtime só conecta quando a tela atual é 'realtime'.
  const realtimeStatus = useRealtimeSocket({
    ativo: currentScreen === 'realtime',
    setAppState,
    addToast,
  });

  const auditSessaoIdNumero = Number(auditSessaoId) || 1;

  const value = useMemo(() => ({
    currentScreen, setCurrentScreen,
    appState, setAppState,
    activeGroupId, setActiveGroupId,
    auditSessaoId, setAuditSessaoId, auditSessaoIdNumero,
    userData, setUserData,
    toasts, addToast,
    realtimeStatus,
  }), [
    currentScreen, appState, activeGroupId, auditSessaoId, auditSessaoIdNumero,
    userData, toasts, addToast, realtimeStatus, setAppState, setAuditSessaoId,
  ]);

  return (
    <AppStateContext.Provider value={value}>
      {children}
    </AppStateContext.Provider>
  );
}
