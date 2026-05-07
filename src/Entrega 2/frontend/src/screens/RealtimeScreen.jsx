import React, { useMemo } from 'react';
import { useAppState } from '../context/appStateContextValue';
import MetricCards from '../components/realtime/MetricCards';
import GroupBarCharts from '../components/realtime/GroupBarCharts';
import ProductTable from '../components/realtime/ProductTable';
import RecentActivity from '../components/realtime/RecentActivity';
import GroupBreakdown from '../components/realtime/GroupBreakdown';

// Visualização Gráfica: agrega dados do appState em métricas, gráficos
// horizontais, tabela de produtos, feed e detalhamento por grupo.
export default function RealtimeScreen() {
  const { appState, realtimeStatus, setCurrentScreen } = useAppState();

  const computed = useMemo(() => {
    const totalGroups = appState.length;
    const allItems = appState.flatMap(g => g.items.map(item => ({ ...item, group: g.title, groupId: g.id })));
    const totalItems = allItems.length;
    const totalKg = appState.reduce((acc, g) => acc + g.totalKg, 0);
    const avgKgPerGroup = totalGroups > 0 ? (totalKg / totalGroups).toFixed(1) : '0';
    const maxGroupKg = Math.max(...appState.map(g => g.totalKg), 1);
    const maxGroupItems = Math.max(...appState.map(g => g.items.length), 1);

    // Product aggregation
    const productMap = {};
    allItems.forEach(item => {
      const key = item.name.toLowerCase().trim();
      if (!productMap[key]) productMap[key] = { name: item.name, totalKg: 0, count: 0, groups: new Set() };
      productMap[key].totalKg += item.weight;
      productMap[key].count += 1;
      productMap[key].groups.add(item.group);
    });
    const products = Object.values(productMap).sort((a, b) => b.totalKg - a.totalKg);
    const uniqueProducts = products.length;
    const topProduct = products.length > 0 ? products[0] : null;

    // Recent items
    const recentItems = allItems.slice(-10).reverse();

    return { totalGroups, totalItems, totalKg, avgKgPerGroup, maxGroupKg, maxGroupItems, products, uniqueProducts, topProduct, recentItems };
  }, [appState]);

  const statusLabel = realtimeStatus === 'online' ? 'Conectado' : realtimeStatus === 'connecting' ? 'Conectando...' : 'Offline';
  const statusClass = realtimeStatus === 'online' ? 'online' : realtimeStatus === 'connecting' ? 'connecting' : 'offline';

  return (
    <div className="flex flex-col h-full bg-dark">
      <header className="flex justify-between align-center p-5 border-b border-gray-medium bg-header shadow-md relative z-10">
        <div className="flex align-center gap-3">
          <button className="btn-icon circle-bg-gray border border-gray-dark shadow transition hover-bg-gray-light mr-4" onClick={() => setCurrentScreen('dashboard')}>
            <i className="ph ph-arrow-left text-xl text-white"></i>
          </button>
          <i className="ph ph-chart-bar text-2xl text-primary"></i>
          <span className="font-bold text-xl text-white tracking-tight">Visualização Gráfica</span>
        </div>
        <div className="flex align-center gap-3">
          <div className={`flex align-center gap-2 px-4 py-2 rounded-full border border-gray-medium`} style={{ background: 'rgba(0,0,0,0.3)' }}>
            <div className={`status-indicator status-${statusClass}`}></div>
            <span className="text-xs font-bold text-white tracking-wide">{statusLabel}</span>
          </div>
        </div>
      </header>

      <div className="flex-grow p-6 overflow-y-auto">

        {/* ── Metric Cards ── */}
        <MetricCards
          totalGroups={computed.totalGroups}
          uniqueProducts={computed.uniqueProducts}
          totalItems={computed.totalItems}
          totalKg={computed.totalKg}
          avgKgPerGroup={computed.avgKgPerGroup}
          topProduct={computed.topProduct}
        />

        {/* ── Horizontal Bar Charts: Weight & Items per Group ── */}
        <GroupBarCharts
          appState={appState}
          maxGroupKg={computed.maxGroupKg}
          maxGroupItems={computed.maxGroupItems}
        />

        {/* ── Product Distribution Table + Recent Items ── */}
        <div className="realtime-grid fade-in" style={{ animationDelay: '0.2s', animationFillMode: 'both', marginBottom: '1.5rem' }}>
          <ProductTable products={computed.products} totalKg={computed.totalKg} />
          <RecentActivity recentItems={computed.recentItems} />
        </div>

        {/* ── Group Breakdown Cards ── */}
        <GroupBreakdown appState={appState} totalKg={computed.totalKg} />

        {/* Offline Info Banner */}
        {realtimeStatus === 'offline' && (
          <div className="flex align-center gap-3 p-4 rounded-xl fade-in" style={{ background: 'rgba(245, 166, 35, 0.08)', border: '1px solid rgba(245, 166, 35, 0.2)' }}>
            <i className="ph ph-cloud-slash text-xl" style={{ color: '#f5a623' }}></i>
            <div>
              <div className="text-sm font-bold text-white">Modo Offline</div>
              <div className="text-xs text-gray">Exibindo dados salvos localmente. Reconexão automática ativada.</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
