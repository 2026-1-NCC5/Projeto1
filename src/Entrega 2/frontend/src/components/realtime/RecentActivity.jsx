import React from 'react';

// Feed lateral "Atividade Recente": últimos 10 itens (qualquer grupo).
export default function RecentActivity({ recentItems }) {
  return (
    <div className="recent-items-container">
      <div className="dash-section-header"><i className="ph-fill ph-clock-countdown text-primary"></i> Atividade Recente</div>
      {recentItems.length === 0 ? (
        <div className="text-center text-gray py-8 text-sm opacity-50">Nenhum item registrado ainda.</div>
      ) : (
        recentItems.map((item, idx) => (
          <div key={idx} className="recent-item">
            <div className="flex align-center gap-3">
              <div className="circle-icon" style={{ background: 'var(--primary-light)', width: '32px', height: '32px' }}>
                <i className="ph-fill ph-check-circle text-primary text-sm"></i>
              </div>
              <div>
                <div className="text-sm font-bold text-white">{item.name}</div>
                <div className="text-xs text-gray">{item.group}</div>
              </div>
            </div>
            <span className="badge-red px-2 py-1 rounded-full text-xs font-black">+{item.weight}kg</span>
          </div>
        ))
      )}
    </div>
  );
}
