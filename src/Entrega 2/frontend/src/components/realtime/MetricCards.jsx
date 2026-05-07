import React from 'react';

// Cartões de métrica do topo da Visualização Gráfica.
export default function MetricCards({ totalGroups, uniqueProducts, totalItems, totalKg, avgKgPerGroup, topProduct }) {
  return (
    <div className="realtime-metrics fade-in" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))' }}>
      <div className="realtime-card">
        <div className="realtime-card-icon" style={{ background: 'rgba(159, 24, 24, 0.15)', color: 'var(--primary)' }}><i className="ph-fill ph-users-three"></i></div>
        <div className="realtime-card-value">{totalGroups}</div>
        <div className="realtime-card-label">Grupos Ativos</div>
      </div>
      <div className="realtime-card">
        <div className="realtime-card-icon" style={{ background: 'rgba(139, 92, 246, 0.15)', color: '#8b5cf6' }}><i className="ph-fill ph-carrot"></i></div>
        <div className="realtime-card-value">{uniqueProducts}</div>
        <div className="realtime-card-label">Produtos Únicos</div>
      </div>
      <div className="realtime-card">
        <div className="realtime-card-icon" style={{ background: 'rgba(59, 130, 246, 0.15)', color: '#3b82f6' }}><i className="ph-fill ph-package"></i></div>
        <div className="realtime-card-value">{totalItems}</div>
        <div className="realtime-card-label">Itens Registrados</div>
      </div>
      <div className="realtime-card">
        <div className="realtime-card-icon" style={{ background: 'rgba(34, 197, 94, 0.15)', color: '#22c55e' }}><i className="ph-fill ph-scales"></i></div>
        <div className="realtime-card-value">{totalKg.toFixed(1)}<span className="text-sm text-gray ml-1">kg</span></div>
        <div className="realtime-card-label">Peso Total</div>
      </div>
      <div className="realtime-card">
        <div className="realtime-card-icon" style={{ background: 'rgba(245, 166, 35, 0.15)', color: '#f5a623' }}><i className="ph-fill ph-chart-line-up"></i></div>
        <div className="realtime-card-value">{avgKgPerGroup}<span className="text-sm text-gray ml-1">kg</span></div>
        <div className="realtime-card-label">Média / Grupo</div>
      </div>
      <div className="realtime-card">
        <div className="realtime-card-icon" style={{ background: 'rgba(236, 72, 153, 0.15)', color: '#ec4899' }}><i className="ph-fill ph-trophy"></i></div>
        <div className="realtime-card-value" style={{ fontSize: topProduct ? '1.1rem' : '1.25rem' }}>{topProduct ? topProduct.name : '—'}</div>
        <div className="realtime-card-label">Top Produto</div>
      </div>
    </div>
  );
}
