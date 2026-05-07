// Endpoint da API FastAPI (mesma origem do WebSocket /ws/auditoria/{sessao_id})
export const API_BASE = 'http://localhost:8000';
export const WS_BASE = 'ws://localhost:8000';

// Servidor Socket.IO de tempo real (eventos grupo_atualizado / item_adicionado /
// peso_atualizado). Roda em porta separada do FastAPI; ainda não há servidor
// implementado — o cliente entra em modo offline quando indisponível.
export const REALTIME_BASE = 'http://localhost:5000';
