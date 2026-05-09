import React from 'react';

// Painel lateral de logs ao vivo (WebSocket / YOLO / API) na CameraScreen.
export default function ScannerLogPanel({ logs, onLimpar, onOcultar }) {
  return (
    <aside className="scanner-log-panel">
      <div className="scanner-log-header">
        <div>
          <span className="text-primary text-[10px] font-bold uppercase tracking-widest">Logs ao vivo</span>
          <h3 className="text-white text-sm font-bold">WebSocket / YOLO / API</h3>
        </div>
        <div className="scanner-log-header-actions">
          {typeof onOcultar === 'function' && (
            <button
              type="button"
              className="btn-icon circle-bg-dark border border-gray-medium"
              onClick={onOcultar}
              title="Ocultar painel de logs"
            >
              <i className="ph ph-eye-slash text-white text-base" aria-hidden></i>
            </button>
          )}
          <button className="btn btn-outline btn-small" onClick={onLimpar}>Limpar</button>
        </div>
      </div>
      <div className="scanner-log-list">
        {logs.length === 0 ? (
          <div className="scanner-log-empty">Clique em iniciar captura para ver os eventos.</div>
        ) : logs.slice(-28).reverse().map(log => (
          <div className="scanner-log-item" key={log.id}>
            <span className="scanner-log-stage">{log.stage}</span>
            <span className="scanner-log-message">{log.mensagem}</span>
          </div>
        ))}
      </div>
    </aside>
  );
}
