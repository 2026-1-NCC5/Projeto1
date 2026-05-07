import React from 'react';

// Pareados: peso por grupo (esq.) e itens por grupo (dir.) em barras horizontais.
export default function GroupBarCharts({ appState, maxGroupKg, maxGroupItems }) {
  return (
    <div className="realtime-grid fade-in" style={{ animationDelay: '0.1s', animationFillMode: 'both', marginBottom: '1.5rem' }}>
      {/* Weight per group */}
      <div className="chart-container">
        <div className="dash-section-header"><i className="ph-fill ph-chart-bar text-primary"></i> Peso por Grupo (kg)</div>
        {appState.length === 0 ? (
          <div className="text-center text-gray py-8 text-sm opacity-50">Nenhum grupo cadastrado.</div>
        ) : (
          appState.map(group => (
            <div key={group.id} className="hbar-row">
              <div className="hbar-label">{group.title}</div>
              <div className="hbar-track">
                <div className="hbar-fill" style={{ width: `${Math.max((group.totalKg / maxGroupKg) * 100, 2)}%` }}>
                  {group.totalKg > 0 && <span className="hbar-value">{group.totalKg}kg</span>}
                </div>
              </div>
              <div className="hbar-side-value">{group.totalKg}kg</div>
            </div>
          ))
        )}
      </div>

      {/* Items per group */}
      <div className="chart-container">
        <div className="dash-section-header"><i className="ph-fill ph-list-numbers text-primary"></i> Itens por Grupo</div>
        {appState.length === 0 ? (
          <div className="text-center text-gray py-8 text-sm opacity-50">Nenhum grupo cadastrado.</div>
        ) : (
          appState.map(group => (
            <div key={group.id} className="hbar-row">
              <div className="hbar-label">{group.title}</div>
              <div className="hbar-track">
                <div className="hbar-fill hbar-fill-alt" style={{ width: `${Math.max((group.items.length / maxGroupItems) * 100, 2)}%` }}>
                  {group.items.length > 0 && <span className="hbar-value">{group.items.length}</span>}
                </div>
              </div>
              <div className="hbar-side-value" style={{ color: '#3b82f6' }}>{group.items.length} itens</div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
