import React, { useState } from 'react';
import { useAppState } from '../context/appStateContextValue';

// Configuração de dados do usuário (perfil do mantenedor).
export default function ConfigScreen() {
  const { userData, setUserData, setCurrentScreen } = useAppState();
  const [tempUserData, setTempUserData] = useState({ ...userData });

  const handleConfigSave = () => {
    setUserData({ ...tempUserData });
    setCurrentScreen('dashboard');
  };

  return (
    <div className="flex flex-col h-full bg-dark">
      <header className="flex align-center p-6 border-b border-gray-medium bg-header shadow-md relative z-10">
        <button className="btn-icon circle-bg-gray border border-gray-dark shadow transition hover-bg-gray-light mr-4" onClick={() => setCurrentScreen('dashboard')}>
          <i className="ph ph-arrow-left text-xl text-white"></i>
        </button>
        <span className="font-bold text-xl text-white tracking-tight">Configuração de Dados</span>
      </header>

      <div className="flex-grow flex flex-col p-6" style={{ overflow: 'auto' }}>
        <div className="w-full max-w-2xl bg-card border border-gray-medium shadow-2xl slide-up-anim flex flex-col" style={{ padding: '2.5rem 3rem', margin: '0 auto', flex: '1 1 auto', borderRadius: '2rem' }}>
          <div className="flex align-center gap-4 mb-8">
            <div className="circle-bg-gray flex-center" style={{ width: '64px', height: '64px', borderRadius: '50%' }}>
              <i className="ph-fill ph-user-circle text-3xl text-primary"></i>
            </div>
            <div>
              <h2 className="font-bold text-3xl text-white tracking-tight">Editar Perfil</h2>
              <p className="text-gray text-md font-medium mt-2">Atualize seus dados de mantenedor.</p>
            </div>
          </div>

          <div className="flex flex-col gap-8 flex-grow" style={{ justifyContent: 'space-evenly' }}>
            <div className="flex gap-6">
              <div className="flex flex-col gap-3 w-full">
                <label className="text-sm text-gray font-bold uppercase tracking-widest ml-1">Nome</label>
                <input type="text" className="input-dark w-full text-lg" style={{ padding: '1.1rem 1.25rem' }} value={tempUserData.nome || ''} onChange={e => setTempUserData({ ...tempUserData, nome: e.target.value })} placeholder="Ex: João" />
              </div>
              <div className="flex flex-col gap-3 w-full">
                <label className="text-sm text-gray font-bold uppercase tracking-widest ml-1">Sobrenome</label>
                <input type="text" className="input-dark w-full text-lg" style={{ padding: '1.1rem 1.25rem' }} value={tempUserData.sobrenome || ''} onChange={e => setTempUserData({ ...tempUserData, sobrenome: e.target.value })} placeholder="Ex: Silva" />
              </div>
            </div>
            <div className="flex flex-col gap-3 w-full">
              <label className="text-sm text-gray font-bold uppercase tracking-widest ml-1">E-mail Institucional</label>
              <input type="email" className="input-dark w-full text-lg" style={{ padding: '1.1rem 1.25rem' }} value={tempUserData.email || ''} onChange={e => setTempUserData({ ...tempUserData, email: e.target.value })} placeholder="joao.silva@usp.br" />
            </div>
            <div className="flex flex-col gap-3 w-full">
              <label className="text-sm text-gray font-bold uppercase tracking-widest ml-1">Matrícula (RA)</label>
              <input type="number" className="input-dark w-full text-lg" style={{ padding: '1.1rem 1.25rem' }} value={tempUserData.ra || ''} onChange={e => setTempUserData({ ...tempUserData, ra: e.target.value })} placeholder="Ex: 12345678" />
            </div>
          </div>

          <button className="btn btn-primary w-full py-5 rounded-2xl shadow-red transition hover:scale-[1.02] mt-8" onClick={handleConfigSave}>
            <span className="font-bold text-lg">Salvar Alterações</span> <i className="ph ph-check-circle text-2xl"></i>
          </button>
        </div>
      </div>
    </div>
  );
}
