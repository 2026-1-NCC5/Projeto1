import { createContext, useContext } from 'react';

// Context separado do provider para que o módulo do provider mantenha
// fast-refresh limpo (regra react-refresh/only-export-components).
export const AppStateContext = createContext(null);

export function useAppState() {
  const ctx = useContext(AppStateContext);
  if (!ctx) throw new Error('useAppState deve ser usado dentro de <AppStateProvider>');
  return ctx;
}
