import { useEffect, useRef, useState, useCallback } from 'react';

/**
 * Conecta-se ao WebSocket nativo do FastAPI (`ws://<host>/ws/auditoria/{sessaoId}`),
 * envia frames base64 a 2 FPS a partir do <video> referenciado e expõe a última
 * detecção/status recebida do backend.
 *
 * Mensagens vindas do servidor (ver api/routers/ws_auditoria.py):
 *  - { tipo: 'status',   estado: 'monitorando'|'estavel'|'analisando'|'lock', lock_ate_ts }
 *  - { tipo: 'deteccao', yolo, gemini, resultado_final, imagem_path, ts }
 *  - { tipo: 'erro',     stage, mensagem }
 *
 * @param {{ sessaoId: number|string|null, videoRef: React.RefObject<HTMLVideoElement>,
 *           wsBaseUrl?: string, fps?: number, ativo?: boolean }} params
 */
export default function useAuditoriaWS({
  sessaoId,
  videoRef,
  wsBaseUrl = 'ws://localhost:8000',
  fps = 2,
  ativo = true,
  usarGemini = true,
}) {
  const wsRef = useRef(null);
  const intervalRef = useRef(null);
  const canvasRef = useRef(null);
  const usarGeminiRef = useRef(usarGemini);

  const [status, setStatus] = useState('offline');
  const [ultimaDeteccao, setUltimaDeteccao] = useState(null);
  const [ultimoErro, setUltimoErro] = useState(null);
  const [ultimoPreview, setUltimoPreview] = useState(null);
  const [logs, setLogs] = useState([]);
  const [capturando, setCapturando] = useState(false);

  const adicionarLog = useCallback((entrada) => {
    const item = {
      id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      ts: entrada.ts || Date.now(),
      stage: entrada.stage || 'front',
      mensagem: entrada.mensagem || '',
      dados: entrada.dados || null,
    };
    setLogs(prev => [...prev.slice(-79), item]);
  }, []);

  useEffect(() => {
    usarGeminiRef.current = usarGemini;
    const t = setTimeout(() => {
      adicionarLog({
        stage: 'front',
        mensagem: usarGemini ? 'Consulta Gemini ativada' : 'Consulta Gemini desativada',
      });
    }, 0);
    return () => clearTimeout(t);
  }, [usarGemini, adicionarLog]);

  useEffect(() => {
    if (!ativo || !sessaoId) {
      return;
    }
    if (canvasRef.current === null && typeof document !== 'undefined') {
      canvasRef.current = document.createElement('canvas');
    }
    const url = `${wsBaseUrl}/ws/auditoria/${sessaoId}`;
    const logTimer = setTimeout(() => {
      adicionarLog({ stage: 'ws', mensagem: `Conectando em ${url}` });
    }, 0);
    let ws;
    try {
      ws = new WebSocket(url);
    } catch {
      // Erro ao construir WebSocket: agenda mudança de status para o próximo tick
      // (evita setState síncrono no corpo do effect — exigência do React 19/eslint).
      const t = setTimeout(() => setStatus('erro'), 0);
      return () => clearTimeout(t);
    }
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus('monitorando');
      adicionarLog({ stage: 'ws', mensagem: 'WebSocket aberto' });
    };
    ws.onclose = () => {
      setStatus('offline');
      setCapturando(false);
      adicionarLog({ stage: 'ws', mensagem: 'WebSocket fechado' });
    };
    ws.onerror = () => {
      setStatus('erro');
      adicionarLog({ stage: 'ws', mensagem: 'Erro no WebSocket' });
    };
    ws.onmessage = (evt) => {
      let msg;
      try {
        msg = JSON.parse(evt.data);
      } catch {
        return;
      }
      if (msg.tipo === 'deteccao') {
        setUltimaDeteccao(msg);
        adicionarLog({
          stage: 'resultado',
          mensagem: `Detecção: ${msg.resultado_final?.alimento_nome || 'desconhecido'}`,
          dados: msg,
        });
      } else if (msg.tipo === 'preview') {
        setUltimoPreview(msg);
        if (msg.yolo) {
          adicionarLog({
            stage: 'yolo',
            mensagem: `${msg.yolo.classe} (${Math.round((msg.yolo.confianca || 0) * 100)}%)`,
            dados: msg.yolo,
          });
        }
      } else if (msg.tipo === 'log') {
        adicionarLog(msg);
      } else if (msg.tipo === 'status') {
        setStatus(msg.estado || 'monitorando');
      } else if (msg.tipo === 'erro') {
        setUltimoErro(msg);
        adicionarLog({
          stage: msg.stage || 'erro',
          mensagem: msg.mensagem || 'Erro recebido do backend',
          dados: msg,
        });
      }
    };

    return () => {
      clearTimeout(logTimer);
      try { ws.close(); } catch { /* noop */ }
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      setCapturando(false);
    };
  }, [sessaoId, wsBaseUrl, ativo, adicionarLog]);

  const iniciarCaptura = useCallback(() => {
    if (intervalRef.current) {
      adicionarLog({ stage: 'front', mensagem: 'Captura já está ativa' });
      return;
    }
    const intervaloMs = Math.max(100, Math.round(1000 / fps));
    const canvas = canvasRef.current || document.createElement('canvas');
    canvasRef.current = canvas;
    adicionarLog({ stage: 'front', mensagem: `Captura iniciada (${fps} FPS)` });
    setCapturando(true);

    intervalRef.current = setInterval(() => {
      const video = videoRef.current;
      const ws = wsRef.current;
      if (!video || !video.videoWidth || !ws || ws.readyState !== WebSocket.OPEN) return;

      // Reduz para 640x480 para limitar payload (qualidade adequada para YOLO)
      canvas.width = 640;
      canvas.height = 480;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const b64 = canvas.toDataURL('image/jpeg', 0.6);
      try {
        ws.send(JSON.stringify({
          tipo: 'frame',
          ts: Date.now(),
          imagem_b64: b64,
          usar_gemini: usarGeminiRef.current,
        }));
      } catch { /* socket pode ter caído entre o check e o send */ }
    }, intervaloMs);
  }, [videoRef, fps, adicionarLog]);

  const pararCaptura = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
      setCapturando(false);
      adicionarLog({ stage: 'front', mensagem: 'Captura pausada' });
    }
  }, [adicionarLog]);

  const reset = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ tipo: 'reset' }));
    }
    setUltimaDeteccao(null);
    adicionarLog({ stage: 'front', mensagem: 'Reset enviado ao backend' });
  }, [adicionarLog]);

  const limparLogs = useCallback(() => {
    setLogs([]);
  }, []);

  return {
    status,
    ultimaDeteccao,
    ultimoErro,
    ultimoPreview,
    logs,
    capturando,
    iniciarCaptura,
    pararCaptura,
    reset,
    limparLogs,
    registrarLog: adicionarLog,
  };
}
