import React from 'react';

export default function FinalizacaoConciliacaoModal({
  aberto,
  carregando,
  erro,
  relatorio,
  salvando,
  onFechar,
  onConfirmarManual,
  onConfirmarCapturas,
}) {
  if (!aberto) return null;

  const divergencias = relatorio?.divergencias ?? 0;
  const revisaoCount = relatorio?.revisao_manual_pendente_count ?? 0;
  const revisaoItens = relatorio?.revisao_manual_itens ?? [];
  const bloqueadoRevisao = revisaoCount > 0;
  const sucesso = !carregando && !erro && divergencias === 0 && !bloqueadoRevisao;
  const atencao = !carregando && !erro && divergencias > 0 && !bloqueadoRevisao;

  return (
    <div className="modal-overlay active">
      <div className="modal-content finalizacao-conciliacao-modal">
        <div className="finalizacao-conciliacao-header">
          <div>
            <span className="finalizacao-conciliacao-eyebrow">Conferência Final</span>
            <h3 className="finalizacao-conciliacao-title">Manual x Capturas</h3>
          </div>
          <button type="button" className="btn-icon" onClick={onFechar} disabled={salvando}>
            <i className="ph ph-x" />
          </button>
        </div>

        <div className="finalizacao-conciliacao-status">
          {carregando ? (
            <div className="finalizacao-status-spinner-wrap">
              <span className="finalizacao-status-spinner" />
              <p>Comparando os dados mais recentes...</p>
            </div>
          ) : null}
          {!carregando && erro ? (
            <div className="finalizacao-status-erro">
              <i className="ph ph-warning-circle" />
              <p>{erro}</p>
            </div>
          ) : null}
          {sucesso ? (
            <div className="finalizacao-status-ok">
              <i className="ph-fill ph-check-circle" />
              <p>Conferência sem divergências.</p>
            </div>
          ) : null}
          {atencao ? (
            <div className="finalizacao-status-alerta">
              <i className="ph-fill ph-warning-circle" />
              <p>{divergencias} divergência(s) encontrada(s).</p>
            </div>
          ) : null}
          {bloqueadoRevisao ? (
            <div className="finalizacao-status-alerta">
              <i className="ph-fill ph-warning-octagon" />
              <div>
                <p>
                  {revisaoCount === 1
                    ? 'Há 1 item que exige revisão manual de peso ou categoria antes de finalizar.'
                    : `Há ${revisaoCount} itens que exigem revisão manual de peso ou categoria antes de finalizar.`}
                </p>
                <p className="text-gray text-xs mt-1">
                  Use o ícone de lápis na lista de itens da sessão, confira a evidência e salve a correção.
                </p>
                {revisaoItens.length > 0 ? (
                  <ul className="text-xs text-gray mt-2">
                    {revisaoItens.slice(0, 8).map((ri) => (
                      <li key={ri.deteccao_id}>
                        #{ri.deteccao_id}
                        {ri.alimento_nome ? ` — ${ri.alimento_nome}` : ''}
                      </li>
                    ))}
                    {revisaoItens.length > 8 ? (
                      <li>… e mais {revisaoItens.length - 8}</li>
                    ) : null}
                  </ul>
                ) : null}
              </div>
            </div>
          ) : null}
        </div>

        {relatorio && !carregando ? (
          <div className="finalizacao-conciliacao-resumo">
            <div className="finalizacao-resumo-col">
              <span>Declarado</span>
              <strong>{Number(relatorio.total_kg_declarado || 0).toFixed(1)}kg</strong>
              <small>{relatorio.total_itens_declarados || 0} item(ns)</small>
            </div>
            <div className="finalizacao-resumo-col">
              <span>Capturado</span>
              <strong>{Number(relatorio.total_kg_detectado || 0).toFixed(1)}kg</strong>
              <small>{relatorio.total_itens_detectados || 0} item(ns)</small>
            </div>
          </div>
        ) : null}

        <div className="finalizacao-conciliacao-actions">
          <button
            type="button"
            className="btn btn-outline"
            onClick={onConfirmarManual}
            disabled={carregando || !!erro || salvando || bloqueadoRevisao}
          >
            <i className="ph ph-hand-palm" />
            Confirmar manualmente
          </button>
          <button
            type="button"
            className="btn btn-primary"
            onClick={onConfirmarCapturas}
            disabled={carregando || !!erro || salvando || bloqueadoRevisao}
          >
            <i className="ph ph-camera" />
            Confirmar capturas
          </button>
        </div>
      </div>
    </div>
  );
}
