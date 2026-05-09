import React, { useEffect } from 'react';
import { AppStateProvider } from './context/AppStateContext';
import { useAppState } from './context/appStateContextValue';
import HomeScreen from './screens/HomeScreen';
import LoginScreen from './screens/LoginScreen';
import CadastroAdminScreen from './screens/CadastroAdminScreen';
import ConfigScreen from './screens/ConfigScreen';
import DashboardScreen from './screens/DashboardScreen';
import CameraScreen from './screens/CameraScreen';
import ManualScreen from './screens/ManualScreen';
import RealtimeScreen from './screens/RealtimeScreen';
import ToastContainer from './components/ToastContainer';

const PROTECTED_SCREENS = new Set(['dashboard', 'camera', 'manual', 'realtime', 'config']);

// Roteador simples baseado em `currentScreen` no contexto. Cada tela
// é totalmente isolada e consome o que precisa via useAppState().
function ScreenRouter() {
  const { currentScreen, setCurrentScreen, authUsuario, authLoading } = useAppState();

  useEffect(() => {
    if (authLoading) return;
    if (PROTECTED_SCREENS.has(currentScreen) && !authUsuario) {
      setCurrentScreen('login');
    }
  }, [authLoading, currentScreen, authUsuario, setCurrentScreen]);

  if (authLoading) {
    return (
      <div className="flex flex-col flex-1 min-h-0 bg-dark align-center justify-center">
        <p className="text-white font-bold tracking-wide">Carregando…</p>
      </div>
    );
  }

  switch (currentScreen) {
    case 'home': return <HomeScreen />;
    case 'login': return <LoginScreen />;
    case 'cadastro': return <CadastroAdminScreen />;
    case 'config': return <ConfigScreen />;
    case 'camera': return <CameraScreen />;
    case 'manual': return <ManualScreen />;
    case 'realtime': return <RealtimeScreen />;
    default: return <DashboardScreen />;
  }
}

// Main Application
export default function App() {
  return (
    <AppStateProvider>
      <div className="h-full w-full flex flex-col app-root-shell">
        <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
          <ScreenRouter />
        </div>
        <ToastContainer />
      </div>
    </AppStateProvider>
  );
}
