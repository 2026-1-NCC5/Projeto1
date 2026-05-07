import React from 'react';
import { useAppState } from '../context/appStateContextValue';

// Overlay global de toasts. Renderizado uma única vez por App.jsx —
// fica fixo via CSS (`.toast-container`).
export default function ToastContainer() {
  const { toasts } = useAppState();
  if (toasts.length === 0) return null;
  return (
    <div className="toast-container">
      {toasts.map(t => (
        <div key={t.id} className={`toast toast-${t.type} ${t.exiting ? 'toast-exit' : ''}`}>
          <div className="toast-icon">
            <i className={`ph-fill ${t.type === 'success' ? 'ph-check-circle' :
              t.type === 'error' ? 'ph-x-circle' :
                t.type === 'warning' ? 'ph-warning' :
                  'ph-info'
              }`}></i>
          </div>
          <span>{t.message}</span>
        </div>
      ))}
    </div>
  );
}
