import React, { useMemo, useState } from 'react';
import SessionItemStatusChip from './SessionItemStatusChip';

// Card lateral fixo na CameraScreen exibindo a lista de itens capturados na
// sessão atual. Não substitui o `DetectionPopup` (card na base e foca no
// próximo passo) — funciona como "scoreboard" persistente da contagem.
// Pode ser recolhido horizontalmente mantendo o resumo (itens + peso).
export default function SessionItemsPanel({ items = [], pesoTotal, onRemoverItem }) {
  const [colapsado, setColapsado] = useState(false);

  const { totalItens, totalKg, ultimoNome } = useMemo(() => {
    const totalKg = pesoTotal != null
      ? pesoTotal
      : items.reduce((acc, it) => acc + (it.weight || 0), 0);
    return {
      totalItens: items.length,
      totalKg,
      ultimoNome: items.length ? items[items.length - 1].name : null,
    };
  }, [items, pesoTotal]);

  return (
    <aside
      className={`session-items-panel${colapsado ? ' session-items-panel--collapsed' : ''}`}
      aria-label={colapsado ? 'Resumo da captura (painel recolhido)' : 'Itens capturados na sessão'}
    >
      <div className="session-items-panel-surface">
        <div
          className="session-items-panel-view session-items-panel-view--strip"
          aria-hidden={!colapsado}
          inert={!colapsado ? true : undefined}
        >
          <button
            type="button"
            className="session-items-strip-expand"
            onClick={() => setColapsado(false)}
            title="Expandir painel de itens"
            aria-expanded={false}
            aria-label="Expandir painel de itens da sessão"
          >
            <i className="ph ph-caret-right text-lg" aria-hidden></i>
          </button>
          <div className="session-items-collapsed-metrics" role="status">
            <span className="session-items-collapsed-value">{totalItens}</span>
            <span className="session-items-collapsed-label">{totalItens === 1 ? 'item' : 'itens'}</span>
            <span className="session-items-collapsed-rule" aria-hidden></span>
            <span className="session-items-collapsed-value">{totalKg.toFixed(1)}</span>
            <span className="session-items-collapsed-label">kg</span>
          </div>
          <div className={`session-items-collapsed-pulse ${totalItens > 0 ? 'is-active' : ''}`}>
            <span></span>
          </div>
        </div>

        <div
          className="session-items-panel-view session-items-panel-view--full"
          aria-hidden={colapsado}
          inert={colapsado ? true : undefined}
        >
          <div className="session-items-header">
            <div className="session-items-header-text">
              <span className="session-items-eyebrow">
                <i className="ph-fill ph-shopping-bag"></i> CAPTURA ATUAL
              </span>
              <h3 className="session-items-title">Itens da sessão</h3>
              <p className="session-items-subtitle">
                {ultimoNome ? `Último: ${ultimoNome}` : 'Aguardando primeiro alimento'}
              </p>
            </div>
            <div className="session-items-header-actions">
              <div className={`session-items-pulse ${totalItens > 0 ? 'is-active' : ''}`}>
                <span></span>
              </div>
              <button
                type="button"
                className="session-items-strip-collapse"
                onClick={() => setColapsado(true)}
                title="Recolher painel lateral"
                aria-expanded={true}
                aria-label="Recolher painel lateral de itens"
              >
                <i className="ph ph-caret-left text-lg" aria-hidden></i>
              </button>
            </div>
          </div>

          <div className="session-items-metrics">
            <div className="session-items-metric">
              <span className="session-items-metric-value">{totalItens}</span>
              <span className="session-items-metric-label">{totalItens === 1 ? 'item' : 'itens'}</span>
            </div>
            <div className="session-items-metric-divider" />
            <div className="session-items-metric">
              <span className="session-items-metric-value">
                {totalKg.toFixed(1)}
                <small>kg</small>
              </span>
              <span className="session-items-metric-label">peso total</span>
            </div>
          </div>

          <div className="session-items-list">
            {items.length === 0 ? (
              <div className="session-items-empty">
                <i className="ph ph-package"></i>
                <p>Nenhum alimento capturado ainda nesta contagem.</p>
                <span>Inicie a captura e posicione o alimento no quadro.</span>
              </div>
            ) : (
              items.map((item, index) => (
                <div
                  className="session-items-row session-items-row--enter"
                  key={`${item.deteccaoId ?? 'manual'}-${index}`}
                >
                  <div className="session-items-row-icon">
                    <i className="ph-fill ph-package"></i>
                  </div>
                  <div className="session-items-row-text">
                    <span className="session-items-row-index">#{index + 1}</span>
                    <span className="session-items-row-name">{item.name}</span>
                    {item.status && (
                      <SessionItemStatusChip
                        status={item.status}
                        nomeOriginal={item.nomeOriginal}
                        nomeAtual={item.name}
                      />
                    )}
                  </div>
                  <span className="session-items-row-weight">+{Number(item.weight || 0).toFixed(1)}<small>kg</small></span>
                  {onRemoverItem && (
                    <button
                      type="button"
                      className="session-items-row-remove"
                      onClick={() => onRemoverItem(index)}
                      title="Remover item"
                    >
                      <i className="ph ph-x"></i>
                    </button>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </aside>
  );
}
