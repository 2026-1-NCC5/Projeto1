import React from 'react';

// Tabela "Distribuição de Produtos" — todos os produtos detectados com
// quantidade, peso total, % e barrinha proporcional.
export default function ProductTable({ products, totalKg }) {
  return (
    <div className="chart-container">
      <div className="dash-section-header"><i className="ph-fill ph-list-bullets text-primary"></i> Distribuição de Produtos</div>
      {products.length === 0 ? (
        <div className="text-center text-gray py-8 text-sm opacity-50">Nenhum produto cadastrado ainda.</div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table className="product-table">
            <thead>
              <tr>
                <th>Produto</th>
                <th>Qtd</th>
                <th>Peso Total</th>
                <th>% do Total</th>
                <th>Distribuição</th>
              </tr>
            </thead>
            <tbody>
              {products.map((p, idx) => {
                const pct = totalKg > 0 ? ((p.totalKg / totalKg) * 100).toFixed(1) : '0';
                return (
                  <tr key={idx}>
                    <td>
                      <div className="product-name">
                        <span className="product-badge" style={{ background: 'var(--primary-light)', color: 'var(--primary)' }}>{idx + 1}</span>
                        {p.name}
                      </div>
                    </td>
                    <td><span className="font-bold">{p.count}x</span></td>
                    <td><span className="font-bold">{p.totalKg.toFixed(1)}kg</span></td>
                    <td><span className="font-bold" style={{ color: 'var(--primary)' }}>{pct}%</span></td>
                    <td style={{ width: '120px' }}>
                      <div className="progress-mini">
                        <div className="progress-mini-fill" style={{ width: `${Math.max(parseFloat(pct), 2)}%` }}></div>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
