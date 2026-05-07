import React from 'react';
import { AppStateProvider } from './context/AppStateContext';
import { useAppState } from './context/appStateContextValue';
import HomeScreen from './screens/HomeScreen';
import CadastroScreen from './screens/CadastroScreen';
import ConfigScreen from './screens/ConfigScreen';
import DashboardScreen from './screens/DashboardScreen';
import CameraScreen from './screens/CameraScreen';
import ManualScreen from './screens/ManualScreen';
import RealtimeScreen from './screens/RealtimeScreen';
import ToastContainer from './components/ToastContainer';

// Roteador simples baseado em `currentScreen` no contexto. Cada tela
// é totalmente isolada e consome o que precisa via useAppState().
function ScreenRouter() {
  const { currentScreen } = useAppState();
  switch (currentScreen) {
    case 'home':      return <HomeScreen />;
    case 'cadastro':  return <CadastroScreen />;
    case 'config':    return <ConfigScreen />;
    case 'camera':    return <CameraScreen />;
    case 'manual':    return <ManualScreen />;
    case 'realtime':  return <RealtimeScreen />;
    default:          return <DashboardScreen />;
  }
}

// Main Application
export default function App() {
  return (
    <AppStateProvider>
      <ScreenRouter />
      <ToastContainer />
    </AppStateProvider>
  );
}
