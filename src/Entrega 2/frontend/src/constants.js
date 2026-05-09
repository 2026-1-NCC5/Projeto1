// Endpoint da API FastAPI (mesma origem do WebSocket /ws/auditoria/{sessao_id})
const RAW_API = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
export const API_BASE = RAW_API.replace(/\/$/, '');
export const WS_BASE = import.meta.env.VITE_WS_BASE
  || (API_BASE.startsWith('https')
    ? API_BASE.replace(/^https/, 'wss')
    : API_BASE.replace(/^http/, 'ws'));
