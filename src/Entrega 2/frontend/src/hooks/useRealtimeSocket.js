import { useEffect, useRef, useState } from 'react';
import { REALTIME_BASE } from '../constants';

// Conecta ao servidor Socket.IO de tempo real (porta 5000) somente enquanto
// `ativo` é true. Atualiza `appState` via setter recebido por parâmetro quando
// chegam eventos `grupo_atualizado`, `item_adicionado` ou `peso_atualizado`.
// Carrega `window.io` (script CDN no index.html) para evitar nova dep.
export default function useRealtimeSocket({ ativo, setAppState, addToast }) {
  const [realtimeStatus, setRealtimeStatus] = useState('offline');
  const realtimeSocketRef = useRef(null);

  useEffect(() => {
    if (!ativo) {
      if (realtimeSocketRef.current) {
        realtimeSocketRef.current.disconnect();
        realtimeSocketRef.current = null;
        // Sincroniza estado externo (socket fechado) com estado React.
        // eslint-disable-next-line react-hooks/set-state-in-effect
        setRealtimeStatus('offline');
      }
      return;
    }

    if (realtimeSocketRef.current) return;

    setRealtimeStatus('connecting');
    addToast('Conectando ao servidor...', 'info');

    try {
      const rt = window.io(REALTIME_BASE, {
        transports: ['websocket'],
        reconnectionAttempts: Infinity,
        reconnectionDelay: 2000,
        reconnectionDelayMax: 10000,
        timeout: 8000
      });
      realtimeSocketRef.current = rt;

      rt.on('connect', () => {
        setRealtimeStatus('online');
        addToast('Conectado ao servidor em tempo real!', 'success');
      });

      rt.on('disconnect', () => {
        setRealtimeStatus('offline');
        addToast('Desconectado do servidor. Usando dados locais.', 'warning');
      });

      rt.on('connect_error', () => {
        setRealtimeStatus('offline');
      });

      rt.on('reconnecting', () => {
        setRealtimeStatus('connecting');
      });

      rt.on('reconnect', () => {
        setRealtimeStatus('online');
        addToast('Reconectado com sucesso!', 'success');
      });

      // Real-time data events
      rt.on('grupo_atualizado', (data) => {
        if (data && data.id) {
          setAppState(prev => prev.map(g => g.id === data.id ? { ...g, ...data } : g));
          addToast(`Grupo ${data.title || data.id} atualizado`, 'info');
        }
      });

      rt.on('item_adicionado', (data) => {
        if (data && data.groupId && data.item) {
          setAppState(prev => prev.map(g => {
            if (g.id === data.groupId) {
              const newItems = [...g.items, data.item];
              return { ...g, items: newItems, totalKg: parseFloat((g.totalKg + (data.item.weight || 0)).toFixed(2)) };
            }
            return g;
          }));
          addToast(`+${data.item.weight}kg ${data.item.name} adicionado`, 'success');
        }
      });

      rt.on('peso_atualizado', (data) => {
        if (data && data.groupId !== undefined) {
          setAppState(prev => prev.map(g => g.id === data.groupId ? { ...g, totalKg: data.totalKg } : g));
        }
      });

      // After 5s, if still connecting, show offline toast
      setTimeout(() => {
        if (realtimeSocketRef.current && !realtimeSocketRef.current.connected) {
          setRealtimeStatus('offline');
          addToast('Servidor indisponível. Modo offline ativado.', 'warning');
        }
      }, 5000);

    } catch {
      setRealtimeStatus('offline');
      addToast('Erro ao conectar. Modo offline ativado.', 'error');
    }

    return () => {
      if (realtimeSocketRef.current) {
        realtimeSocketRef.current.disconnect();
        realtimeSocketRef.current = null;
      }
    };
  }, [ativo, addToast, setAppState]);

  return realtimeStatus;
}
