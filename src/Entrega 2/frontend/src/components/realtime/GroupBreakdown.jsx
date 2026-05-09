import React from 'react';

// Detalhamento por grupo: peso, itens, membros + top 4 produtos do grupo.
export default function GroupBreakdown({ appState, totalKg }) {
  return (
    <div className="fade-in" style={{ animationDelay: '0.3s', animationFillMode: 'both', marginBottom: '1.5rem' }}>
      <div className="dash-section-header"><i className="ph-fill ph-squares-four text-primary"></i> Detalhamento por Grupo</div>
      {appState.length === 0 ? (
        <div className="text-center text-gray py-8 text-sm opacity-50">Nenhum grupo cadastrado.</div>
      ) : (
        <div className="group-breakdown">
          {appState.map(group => {
            const groupProducts = {};
            (group.items || []).forEach(item => {
              const k = item.name.toLowerCase().trim();
              if (!groupProducts[k]) groupProducts[k] = { name: item.name, kg: 0, count: 0 };
              groupProducts[k].kg += item.weight;
              groupProducts[k].count += 1;
            });
            const gProducts = Object.values(groupProducts).sort((a, b) => b.kg - a.kg);
            const sharePercent = totalKg > 0 ? ((group.totalKg / totalKg) * 100).toFixed(0) : 0;

            return (
              <div key={group.id} className="group-breakdown-card">
                <div className="flex justify-between align-center mb-3">
                  <div className="flex align-center gap-2">
                    <div className="circle-icon" style={{ background: 'var(--primary-light)', width: '28px', height: '28px' }}>
                      <i className="ph-fill ph-flag text-primary text-xs"></i>
                    </div>
                    <span className="font-bold text-white">{group.title}</span>
                  </div>
                  <span className="text-xs font-bold px-2 py-1 rounded-full" style={{ background: 'var(--primary-light)', color: 'var(--primary)' }}>{sharePercent}%</span>
                </div>

                <div className="flex gap-4 mb-3">
                  <div>
                    <div className="text-lg font-black text-white" style={{ lineHeight: 1 }}>{group.totalKg}kg</div>
                    <div className="text-xs text-gray mt-1">Peso</div>
                  </div>
                  <div>
                    <div className="text-lg font-black text-white" style={{ lineHeight: 1 }}>{group.quantidadeItens ?? group.items.length}</div>
                    <div className="text-xs text-gray mt-1">Itens</div>
                  </div>
                  <div>
                    <div className="text-lg font-black text-white" style={{ lineHeight: 1 }}>{group.members?.length ?? 0}</div>
                    <div className="text-xs text-gray mt-1">Membros</div>
                  </div>
                </div>

                <div className="progress-mini mb-3">
                  <div className="progress-mini-fill" style={{ width: `${Math.max(parseFloat(sharePercent), 2)}%` }}></div>
                </div>

                {gProducts.length > 0 ? (
                  <div className="flex flex-col gap-1">
                    {gProducts.slice(0, 4).map((p, i) => (
                      <div key={i} className="flex justify-between align-center text-xs py-1">
                        <span className="text-gray font-medium">{p.name} <span className="text-white font-bold">x{p.count}</span></span>
                        <span className="font-bold text-primary">{p.kg.toFixed(1)}kg</span>
                      </div>
                    ))}
                    {gProducts.length > 4 && <div className="text-xs text-gray opacity-50 mt-1">+{gProducts.length - 4} outros produtos</div>}
                  </div>
                ) : (
                  <div className="text-xs text-gray opacity-50">Sem itens registrados</div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
