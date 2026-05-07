import React, { useEffect, useState } from 'react';

// Overlay animado que aparece após o operador confirmar uma detecção.
// Mostra nome/categoria/peso, total da sessão e uma progress bar de
// `duracaoMs` ms até retomar a captura. Permite pular via botão.
export default function DetectionConfirmedOverlay({
  visivel,
  alimento,
  peso,
  categoria,
  totalCapturados,
  pesoTotalSessao,
  duracaoMs = 3000,
  geminiConcorda,
  onConcluir,
  onPular,
}) {
  const [progresso, setProgresso] = useState(0);

  // Progresso 0 → 100 em duracaoMs. Reinicia quando o overlay reabre.
  useEffect(() => {
    if (!visivel) return undefined;
    const inicio = performance.now();
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setProgresso(0);
    let raf;
    const tick = (now) => {
      const p = Math.min(100, ((now - inicio) / duracaoMs) * 100);
      setProgresso(p);
      if (p < 100) {
        raf = requestAnimationFrame(tick);
      } else {
        onConcluir?.();
      }
    };
    raf = requestAnimationFrame(tick);
    return () => { if (raf) cancelAnimationFrame(raf); };
  }, [visivel, duracaoMs, onConcluir]);

  if (!visivel) return null;

  return (
    <div className="detection-confirmed-overlay">
      <div className="detection-confirmed-card">
        <div className="detection-confirmed-check">
          <i className="ph-fill ph-check"></i>
        </div>

        <span className="detection-confirmed-tag">ALIMENTO REGISTRADO</span>
        <h2 className="detection-confirmed-name">{alimento || 'Alimento'}</h2>

        {(categoria || geminiConcorda != null) && (
          <div className="detection-confirmed-meta">
            {categoria && (
              <span className="detection-confirmed-chip">
                <i className="ph ph-tag"></i> {categoria}
              </span>
            )}
            {geminiConcorda != null && (
              <span className={`detection-confirmed-chip ${geminiConcorda ? 'chip-success' : 'chip-warn'}`}>
                <i className={`ph-fill ${geminiConcorda ? 'ph-sparkle' : 'ph-warning'}`}></i>
                Gemini {geminiConcorda ? 'concorda' : 'discordou'}
              </span>
            )}
          </div>
        )}

        <div className="detection-confirmed-weight">
          <span className="weight-value">+{Number(peso || 0).toFixed(1)}</span>
          <span className="weight-unit">kg</span>
        </div>

        <div className="detection-confirmed-stats">
          <div className="confirmed-stat">
            <span className="confirmed-stat-value">{totalCapturados}</span>
            <span className="confirmed-stat-label">{totalCapturados === 1 ? 'item capturado' : 'itens capturados'}</span>
          </div>
          <div className="confirmed-stat-divider"></div>
          <div className="confirmed-stat">
            <span className="confirmed-stat-value">{Number(pesoTotalSessao || 0).toFixed(1)}<small>kg</small></span>
            <span className="confirmed-stat-label">peso da sessão</span>
          </div>
        </div>

        <div className="detection-confirmed-progress">
          <div
            className="detection-confirmed-progress-fill"
            style={{ width: `${progresso}%` }}
          />
        </div>
        <div className="detection-confirmed-hint">
          <i className="ph ph-camera-rotate"></i>
          Próxima captura em {Math.max(0, Math.ceil((duracaoMs / 1000) * (1 - progresso / 100)))}s
        </div>

        <button className="detection-confirmed-skip" onClick={onPular} type="button">
          Pular e continuar agora
        </button>
      </div>
    </div>
  );
}
