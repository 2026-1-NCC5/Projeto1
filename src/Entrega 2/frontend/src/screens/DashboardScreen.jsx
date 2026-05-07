import React, { useState } from 'react';
import LogoAbrace from '../assets/logo_final_nova.png';
import { useAppState } from '../context/appStateContextValue';
import KanbanCard from '../components/KanbanCard';
import GroupModal from '../components/GroupModal';
import UserPopup from '../components/UserPopup';

// Tela principal: Kanban de grupos + header com perfil e atalho para
// visualização gráfica. Disparam câmera/manual e edição de grupos.
export default function DashboardScreen() {
  const { appState, setAppState, userData, setCurrentScreen, setActiveGroupId } = useAppState();

  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isUserPopupOpen, setIsUserPopupOpen] = useState(false);
  const [isCreatingNew, setIsCreatingNew] = useState(false);
  const [editingGroupId, setEditingGroupId] = useState(null);
  const [modalInitial, setModalInitial] = useState({ title: '', members: [] });

  const handleEditGroup = (group) => {
    setIsCreatingNew(false);
    setEditingGroupId(group.id);
    setModalInitial({ title: group.title, members: [...group.members] });
    setIsModalOpen(true);
  };

  const handleAddGroup = () => {
    setIsCreatingNew(true);
    setEditingGroupId(null);
    setModalInitial({ title: '', members: [] });
    setIsModalOpen(true);
  };

  const handleSaveModal = ({ title, members }) => {
    if (isCreatingNew) {
      const maxCode = appState.length > 0 ? Math.max(...appState.map(g => g.id.charCodeAt(0))) : 64;
      const newId = String.fromCharCode(maxCode + 1) || 'X';
      setAppState([...appState, {
        id: newId,
        title: title || `Grupo ${newId}`,
        members: [...members],
        totalKg: 0,
        items: []
      }]);
    } else {
      setAppState(appState.map(g => g.id === editingGroupId ? {
        ...g,
        title: title || `Grupo ${editingGroupId}`,
        members: [...members]
      } : g));
    }
    setIsModalOpen(false);
  };

  const handleDeleteGroup = (id) => {
    setAppState(appState.filter(g => g.id !== id));
    setIsModalOpen(false);
  };

  const handleStartCount = (groupId) => {
    setActiveGroupId(groupId);
    setCurrentScreen('camera');
  };

  const handleStartManual = (groupId) => {
    setActiveGroupId(groupId);
    setCurrentScreen('manual');
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
              <span className="text-sm font-bold text-white mr-1">{userData.nome}</span>
              <i className="ph ph-caret-down text-gray text-sm"></i>
            </div>

            {/* User Popup Modal */}
            <UserPopup
              aberto={isUserPopupOpen}
              userData={userData}
              onIrConfig={handleGoConfig}
              onFechar={() => setIsUserPopupOpen(false)}
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
              onIniciarCamera={handleStartCount}
              onIniciarManual={handleStartManual}
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
