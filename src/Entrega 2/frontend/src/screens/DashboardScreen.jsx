import React, { useEffect, useRef, useState } from 'react';
import LogoAbrace from '../assets/logo_final_nova.png';
import { useAppState } from '../context/appStateContextValue';
import KanbanCard from '../components/KanbanCard';
import GroupModal from '../components/GroupModal';
import UserPopup from '../components/UserPopup';
import {
  listarGrupos,
  criarGrupo,
  atualizarGrupo,
  excluirGrupo,
} from '../services/api';
import { aggregateKanbanItems, totalKgFromItems } from '../utils/kanbanItems';

// Tela principal: Kanban de grupos + header com perfil e atalho para
// visualização gráfica. Disparam câmera/manual e edição de grupos.
export default function DashboardScreen() {
  const {
    appState,
    setAppState,
    userData,
    setCurrentScreen,
    setActiveGroupId,
    setAuditSessaoId,
    setManualSomenteAcrescentar,
    addToast,
    logout,
  } = useAppState();

  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isUserPopupOpen, setIsUserPopupOpen] = useState(false);
  const [isCreatingNew, setIsCreatingNew] = useState(false);
  const [editingGroupId, setEditingGroupId] = useState(null);
  const [modalInitial, setModalInitial] = useState({ title: '', members: [] });
  const syncDoneRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const r = await listarGrupos();
      if (cancelled) return;
      if (!r.ok || !Array.isArray(r.data)) {
        if (r.status !== 401) {
          addToast('Não foi possível carregar grupos do servidor.', 'warning');
        }
        return;
      }
      setAppState((prev) => {
        const merged = r.data.map((apiG) => {
          const local =
            prev.find((p) => p.grupoIdBackend === apiG.id)
            || prev.find((p) => !p.grupoIdBackend && p.title === apiG.nome);
          return {
            id: String(apiG.id),
            title: apiG.nome,
            members: Array.isArray(local?.members) ? local.members : [],
            totalKg: local?.totalKg ?? 0,
            items: Array.isArray(local?.items) ? local.items : [],
            grupoIdBackend: apiG.id,
            etapaTriagem: local?.etapaTriagem ?? 'inicio',
            triagemSessaoId: local?.triagemSessaoId ?? null,
          };
        });
        if (!syncDoneRef.current && prev.some((p) => !p.grupoIdBackend)) {
          syncDoneRef.current = true;
          addToast('Grupos sincronizados com o servidor.', 'info');
        }
        return merged;
      });
    })();
    return () => { cancelled = true; };
  // Sincronização inicial com o backend ao abrir o painel.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleEditGroup = (group) => {
    setIsCreatingNew(false);
    setEditingGroupId(String(group.id));
    setModalInitial({ title: group.title, members: [...(group.members || [])] });
    setIsModalOpen(true);
  };

  const handleAddGroup = () => {
    setIsCreatingNew(true);
    setEditingGroupId(null);
    setModalInitial({ title: '', members: [] });
    setIsModalOpen(true);
  };

  const handleSaveModal = async ({ title, members }) => {
    if (isCreatingNew) {
      const r = await criarGrupo({
        nome: (title || 'Novo grupo').trim(),
        descricao: null,
        status: 'pendente',
      });
      if (!r.ok) {
        addToast(
          typeof r.data?.detail === 'string' ? r.data.detail : `Erro ao criar grupo (${r.status})`,
          'error',
        );
        return;
      }
      const g = r.data;
      setAppState((prev) => [...prev, {
        id: String(g.id),
        title: g.nome,
        members: [...members],
        totalKg: 0,
        items: [],
        grupoIdBackend: g.id,
        etapaTriagem: 'inicio',
        triagemSessaoId: null,
      }]);
      addToast('Grupo criado no servidor.', 'success');
    } else {
      const alvo = appState.find((x) => String(x.id) === String(editingGroupId));
      if (alvo?.grupoIdBackend == null) {
        addToast('Grupo sem vínculo ao servidor. Recarregue o painel.', 'error');
        return;
      }
      const r = await atualizarGrupo(alvo.grupoIdBackend, {
        nome: (title || alvo.title).trim(),
        descricao: null,
      });
      if (!r.ok) {
        addToast(`Erro ao atualizar grupo (${r.status})`, 'error');
        return;
      }
      const g = r.data;
      const idAlvo = String(editingGroupId);
      setAppState((prev) => prev.map((gr) => (String(gr.id) === idAlvo ? {
        ...gr,
        title: (g?.nome ?? title ?? gr.title),
        members: Array.isArray(members) ? [...members] : (gr.members || []),
      } : gr)));
      addToast('Grupo atualizado.', 'success');
    }
    setIsModalOpen(false);
  };

  const handleDeleteGroup = async (id) => {
    const alvo = appState.find((x) => String(x.id) === String(id));
    if (alvo?.grupoIdBackend != null) {
      const r = await excluirGrupo(alvo.grupoIdBackend);
      if (!r.ok && r.status !== 404) {
        addToast(`Erro ao excluir grupo (${r.status})`, 'error');
        return;
      }
    }
    setAppState((prev) => prev.filter((g) => String(g.id) !== String(id)));
    setIsModalOpen(false);
    addToast('Grupo removido.', 'success');
  };

  const handleIniciarTriagem = (groupId) => {
    const alvo = appState.find((x) => String(x.id) === String(groupId));
    if (!alvo) return;
    setManualSomenteAcrescentar(false);
    if (alvo.etapaTriagem === 'manual_ok' && alvo.triagemSessaoId != null) {
      setAuditSessaoId(String(alvo.triagemSessaoId));
      setActiveGroupId(groupId);
      setCurrentScreen('camera');
      return;
    }
    setActiveGroupId(groupId);
    setCurrentScreen('manual');
  };

  const handleAcrescentarItens = (groupId) => {
    setManualSomenteAcrescentar(true);
    setActiveGroupId(groupId);
    setCurrentScreen('manual');
  };

  const handleAtualizarItemKanban = (groupId, idx, { quantity, weight }) => {
    const q = Math.max(1, parseInt(quantity, 10) || 1);
    const p = parseFloat(String(weight).replace(',', '.'));
    if (!p || p <= 0) return;
    setAppState((prev) => prev.map((g) => {
      if (String(g.id) !== String(groupId)) return g;
      const items = [...(g.items || [])];
      if (!items[idx]) return g;
      items[idx] = { ...items[idx], quantity: q, weight: p };
      const agg = aggregateKanbanItems(items);
      const tk = totalKgFromItems(agg);
      return { ...g, items: agg, totalKg: parseFloat(tk.toFixed(2)) };
    }));
  };

  const handleRemoverUnidadeKanban = (groupId, idx) => {
    setAppState((prev) => prev.map((g) => {
      if (String(g.id) !== String(groupId)) return g;
      const items = [...(g.items || [])];
      const it = items[idx];
      if (!it) return g;
      const q = Math.max(0, (it.quantity || 1) - 1);
      if (q <= 0) items.splice(idx, 1);
      else items[idx] = { ...it, quantity: q };
      const tk = totalKgFromItems(items);
      return { ...g, items, totalKg: parseFloat(tk.toFixed(2)) };
    }));
  };

  const handleGoConfig = () => {
    setIsUserPopupOpen(false);
    setCurrentScreen('config');
  };

  return (
    <div className="flex flex-col h-full bg-dark">
      <header className="flex justify-between align-center p-5 border-b border-gray-medium bg-header shadow-sm relative z-10">
        <div className="flex align-center gap-3 mt-1">
          <div style={{ width: '50px', height: '50px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <img src={LogoAbrace} alt="Logo" style={{ width: '130%', height: '130%', objectFit: 'contain', clipPath: 'inset(0 0 18% 0)', marginTop: '8px' }} />
          </div>
          <span className="font-black text-2xl tracking-tight text-white drop-shadow-md">ABRACE<span className="text-primary">AI</span></span>
        </div>
        <div className="flex align-center gap-5">
          <button className="btn btn-primary shadow-lg transition hover:scale-105" style={{ padding: '0.6rem 1.2rem', height: 'fit-content' }} onClick={() => setCurrentScreen('realtime')}>
            <i className="ph ph-chart-bar text-lg"></i>
            <span className="font-bold text-xs tracking-wide">VISUALIZAÇÃO GRÁFICA</span>
          </button>

          <div className="relative">
            <div className="flex align-center gap-3 cursor-pointer border py-2 px-4 rounded-full border-gray-medium bg-gray-light hover-opacity-100 opacity-80 transition hover-bg-gray-medium" onClick={() => setIsUserPopupOpen(!isUserPopupOpen)}>
              <i className="ph-fill ph-user-circle text-3xl text-primary"></i>
              <span className="text-sm font-bold text-white mr-1">{userData.nome || 'Admin'}</span>
              <i className="ph ph-caret-down text-gray text-sm"></i>
            </div>

            {/* User Popup Modal */}
            <UserPopup
              aberto={isUserPopupOpen}
              userData={userData}
              onIrConfig={handleGoConfig}
              onFechar={() => setIsUserPopupOpen(false)}
              onSair={logout}
            />
          </div>
        </div>
      </header>

      <div className="flex-grow p-5 overflow-y-auto">
        <div className="flex justify-between align-center mb-6">
          <h2 className="font-bold text-2xl text-white tracking-tight">Esteiras em Aberto</h2>
          <span className="text-sm text-gray font-medium"><span className="text-white font-bold">{appState.length}</span> Grupos Ativos</span>
        </div>

        <div className="kanban-board">
          {appState.map(group => (
            <KanbanCard
              key={group.id}
              group={group}
              onEdit={handleEditGroup}
              onIniciarTriagem={handleIniciarTriagem}
              onAcrescentarItens={handleAcrescentarItens}
              onAtualizarItemKanban={handleAtualizarItemKanban}
              onRemoverUnidadeKanban={handleRemoverUnidadeKanban}
            />
          ))}

          <div className="kanban-card flex align-center justify-center p-5 text-center cursor-pointer transition border border-dashed border-gray-medium rounded-xl scale-hover hover-opacity-100 opacity-60"
            style={{ minHeight: '400px', background: 'transparent' }} onClick={handleAddGroup}>
            <div className="flex flex-col align-center justify-center items-center gap-3 h-full">
              <div className="circle-btn-red shadow-red mx-auto"><i className="ph ph-plus text-white text-2xl"></i></div>
              <span className="font-bold text-white text-md mt-2 tracking-tight block">Novo Grupo</span>
              <span className="text-xs text-gray font-medium max-w-[150px] inline-block">Adicionar novo Grupo de coleta</span>
            </div>
          </div>
        </div>
      </div>

      <GroupModal
        aberto={isModalOpen}
        isCreatingNew={isCreatingNew}
        initialTitle={modalInitial.title}
        initialMembers={modalInitial.members}
        editingGroupId={editingGroupId}
        onSave={handleSaveModal}
        onDelete={handleDeleteGroup}
        onClose={() => setIsModalOpen(false)}
      />
    </div>
  );
}
