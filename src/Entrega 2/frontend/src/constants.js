// Endpoint da API FastAPI (mesma origem do WebSocket /ws/auditoria/{sessao_id})
const RAW_API = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
export const API_BASE = RAW_API.replace(/\/$/, '');
export const WS_BASE = import.meta.env.VITE_WS_BASE
  || (API_BASE.startsWith('https')
    ? API_BASE.replace(/^https/, 'wss')
    : API_BASE.replace(/^http/, 'ws'));

// Servidor Socket.IO de tempo real (eventos grupo_atualizado / item_adicionado /
// peso_atualizado). Roda em porta separada do FastAPI; ainda não há servidor
// implementado — o cliente entra em modo offline quando indisponível.
export const REALTIME_BASE = 'http://localhost:5000';
