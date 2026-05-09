import React, { useEffect, useMemo, useState } from 'react';
import { useAppState } from '../context/appStateContextValue';
import MetricCards from '../components/realtime/MetricCards';
import GroupBarCharts from '../components/realtime/GroupBarCharts';
import ProductTable from '../components/realtime/ProductTable';
import RecentActivity from '../components/realtime/RecentActivity';
import GroupBreakdown from '../components/realtime/GroupBreakdown';
import { relatorioGrupos, relatorioCategorias } from '../services/api';

// Visualização Gráfica: prioriza agregados do FastAPI (relatórios); se vazio,
// usa appState local. Socket.IO :5000 segue opcional via realtimeStatus.
export default function RealtimeScreen() {
  const { appState, realtimeStatus, setCurrentScreen } = useAppState();
  const [relGrupos, setRelGrupos] = useState(null);
  const [relCategorias, setRelCategorias] = useState(null);

  const carregar = async () => {
    const [rg, rc] = await Promise.all([relatorioGrupos(), relatorioCategorias()]);
    if (rg.ok && Array.isArray(rg.data)) setRelGrupos(rg.data);
    if (rc.ok && Array.isArray(rc.data)) setRelCategorias(rc.data);
  };

  useEffect(() => {
    let intervalId;
    const t = setTimeout(() => {
      void carregar();
      intervalId = setInterval(() => { void carregar(); }, 20000);
    }, 0);
    return () => {
      clearTimeout(t);
      if (intervalId) clearInterval(intervalId);
    };
  }, []);

  const usarRelatorios = !!(relGrupos?.length || relCategorias?.length);

  const chartAppState = useMemo(() => {
    if (relGrupos?.length) {
      return relGrupos.map((r) => {
        const local = appState.find((g) => g.title === r.grupo_nome);
        return {
          id: `rel-${r.grupo_nome}`,
          title: r.grupo_nome,
          totalKg: r.total_peso_kg,
          items: local?.items || [],
          members: local?.members || [],
          quantidadeItens: r.total_quantidade,
        };
      });
    }
    return appState;
  }, [relGrupos, appState]);

  const computed = useMemo(() => {
    const fromG = relGrupos?.length > 0;
    const fromC = relCategorias?.length > 0;

    if (fromG) {
      const totalKg = relGrupos.reduce((a, g) => a + (g.total_peso_kg || 0), 0);
      const totalItems = relGrupos.reduce((a, g) => a + (g.total_quantidade || 0), 0);
      const totalGroups = relGrupos.length;
      const avgKgPerGroup = totalGroups > 0 ? (totalKg / totalGroups).toFixed(1) : '0';
      const maxGroupKg = Math.max(...relGrupos.map((g) => g.total_peso_kg || 0), 1);
      const maxGroupItems = Math.max(...relGrupos.map((g) => g.total_quantidade || 0), 1);

      let products;
      let uniqueProducts;
      let topProduct;
      if (fromC) {
        const productMap = {};
        relCategorias.forEach((c) => {
          const key = c.alimento_nome.toLowerCase().trim();
          productMap[key] = {
            name: c.alimento_nome,
            totalKg: c.total_peso_kg || 0,
            count: c.total_quantidade || 0,
            groups: new Set(),
          };
        });
        products = Object.values(productMap).sort((a, b) => b.totalKg - a.totalKg);
        uniqueProducts = products.length;
        topProduct = products.length > 0 ? products[0] : null;
      } else {
        const allItems = appState.flatMap((g) => g.items.map((item) => ({ ...item, group: g.title, groupId: g.id })));
        const productMap = {};
        allItems.forEach((item) => {
          const key = item.name.toLowerCase().trim();
          if (!productMap[key]) productMap[key] = { name: item.name, totalKg: 0, count: 0, groups: new Set() };
          productMap[key].totalKg += item.weight;
          productMap[key].count += 1;
          productMap[key].groups.add(item.group);
        });
        products = Object.values(productMap).sort((a, b) => b.totalKg - a.totalKg);
        uniqueProducts = products.length;
        topProduct = products.length > 0 ? products[0] : null;
      }

      const allItems = appState.flatMap((g) => g.items.map((item) => ({ ...item, group: g.title, groupId: g.id })));
      const recentItems = allItems.slice(-10).reverse();
      return {
        totalGroups, totalItems, totalKg, avgKgPerGroup, maxGroupKg, maxGroupItems,
        products, uniqueProducts, topProduct, recentItems,
      };
    }

    const totalGroups = appState.length;
    const allItems = appState.flatMap((g) => g.items.map((item) => ({ ...item, group: g.title, groupId: g.id })));
    const totalItems = allItems.length;
    const totalKg = appState.reduce((acc, g) => acc + g.totalKg, 0);
    const avgKgPerGroup = totalGroups > 0 ? (totalKg / totalGroups).toFixed(1) : '0';
    const maxGroupKg = Math.max(...appState.map((g) => g.totalKg), 1);
    const maxGroupItems = Math.max(...appState.map((g) => g.items.length), 1);

    const productMap = {};
    allItems.forEach((item) => {
      const key = item.name.toLowerCase().trim();
      if (!productMap[key]) productMap[key] = { name: item.name, totalKg: 0, count: 0, groups: new Set() };
      productMap[key].totalKg += item.weight;
      productMap[key].count += 1;
      productMap[key].groups.add(item.group);
    });
    const products = Object.values(productMap).sort((a, b) => b.totalKg - a.totalKg);
    const uniqueProducts = products.length;
    const topProduct = products.length > 0 ? products[0] : null;
    const recentItems = allItems.slice(-10).reverse();

    return {
      totalGroups, totalItems, totalKg, avgKgPerGroup, maxGroupKg, maxGroupItems,
      products, uniqueProducts, topProduct, recentItems,
    };
  }, [appState, relGrupos, relCategorias]);

  const statusLabel = realtimeStatus === 'online' ? 'Socket.IO' : realtimeStatus === 'connecting' ? 'Conectando…' : 'API + local';
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
        <div className="flex align-center gap-3 flex-wrap justify-end">
          {usarRelatorios && (
            <span className="text-xs font-bold text-primary border border-primary/30 px-2 py-1 rounded-full">Dados do servidor</span>
          )}
          <div className={`flex align-center gap-2 px-4 py-2 rounded-full border border-gray-medium`} style={{ background: 'rgba(0,0,0,0.3)' }}>
            <div className={`status-indicator status-${statusClass}`}></div>
            <span className="text-xs font-bold text-white tracking-wide">{statusLabel}</span>
          </div>
        </div>
      </header>

      <div className="flex-grow p-6 overflow-y-auto">

        <MetricCards
          totalGroups={computed.totalGroups}
          uniqueProducts={computed.uniqueProducts}
          totalItems={computed.totalItems}
          totalKg={computed.totalKg}
          avgKgPerGroup={computed.avgKgPerGroup}
          topProduct={computed.topProduct}
        />

        <GroupBarCharts
          appState={chartAppState}
          maxGroupKg={computed.maxGroupKg}
          maxGroupItems={computed.maxGroupItems}
        />

        <div className="realtime-grid fade-in" style={{ animationDelay: '0.2s', animationFillMode: 'both', marginBottom: '1.5rem' }}>
          <ProductTable products={computed.products} totalKg={computed.totalKg} />
          <RecentActivity recentItems={computed.recentItems} />
        </div>

        <GroupBreakdown appState={chartAppState} totalKg={computed.totalKg} />

        {realtimeStatus === 'offline' && (
          <div className="flex align-center gap-3 p-4 rounded-xl fade-in" style={{ background: 'rgba(245, 166, 35, 0.08)', border: '1px solid rgba(245, 166, 35, 0.2)' }}>
            <i className="ph ph-cloud-slash text-xl" style={{ color: '#f5a623' }}></i>
            <div>
              <div className="text-sm font-bold text-white">Servidor Socket.IO (:5000) indisponível</div>
              <div className="text-xs text-gray">Os gráficos usam GET /relatorios quando há detecções no banco; atividade recente reflete o Kanban local.</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
