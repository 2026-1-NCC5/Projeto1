import React from 'react';
import { useAppState } from '../context/appStateContextValue';

// Tela de cadastro inicial de mantenedor. Hoje os inputs ainda não persistem
// — apenas dispara um alert e volta para Home (placeholder).
export default function CadastroScreen() {
  const { setCurrentScreen } = useAppState();

  return (
    <div className="flex flex-col h-full bg-dark">
      <header className="flex align-center p-6 border-b border-gray-medium bg-header shadow-md relative z-10">
        <button className="btn-icon circle-bg-gray border border-gray-dark shadow transition hover-bg-gray-light mr-4" onClick={() => setCurrentScreen('home')}>
          <i className="ph ph-arrow-left text-xl text-white"></i>
        </button>
        <span className="font-bold text-xl text-white tracking-tight">Cadastro de Novo Usuário</span>
      </header>

      <div className="flex-grow flex flex-col p-6" style={{ overflow: 'auto' }}>
        <div className="w-full max-w-2xl bg-card border border-gray-medium shadow-2xl slide-up-anim flex flex-col" style={{ padding: '2.5rem 3rem', margin: '0 auto', flex: '1 1 auto', borderRadius: '2rem' }}>
          <div className="flex align-center gap-4 mb-8">
            <div className="circle-bg-gray flex-center" style={{ width: '64px', height: '64px', borderRadius: '50%' }}>
              <i className="ph ph-user-plus text-3xl text-primary"></i>
            </div>
            <div>
              <h2 className="font-bold text-3xl text-white tracking-tight">Criar Conta</h2>
              <p className="text-gray text-md font-medium mt-2">Insira as credenciais do novo mantenedor.</p>
            </div>
          </div>

          <div className="flex flex-col gap-8 flex-grow" style={{ justifyContent: 'space-evenly' }}>
            <div className="flex gap-6">
              <div className="flex flex-col gap-3 w-full">
                <label className="text-sm text-gray font-bold uppercase tracking-widest ml-1">Nome</label>
                <input type="text" className="input-dark w-full text-lg" style={{ padding: '1.1rem 1.25rem' }} placeholder="Ex: João" />
              </div>
              <div className="flex flex-col gap-3 w-full">
                <label className="text-sm text-gray font-bold uppercase tracking-widest ml-1">Sobrenome</label>
                <input type="text" className="input-dark w-full text-lg" style={{ padding: '1.1rem 1.25rem' }} placeholder="Ex: Silva" />
              </div>
            </div>
            <div className="flex flex-col gap-3 w-full">
              <label className="text-sm text-gray font-bold uppercase tracking-widest ml-1">E-mail Institucional</label>
              <input type="email" className="input-dark w-full text-lg" style={{ padding: '1.1rem 1.25rem' }} placeholder="joao.silva@usp.br" />
            </div>
            <div className="flex flex-col gap-3 w-full">
              <label className="text-sm text-gray font-bold uppercase tracking-widest ml-1">Matrícula (RA)</label>
              <input type="number" className="input-dark w-full text-lg" style={{ padding: '1.1rem 1.25rem' }} placeholder="Ex: 12345678" />
            </div>
          </div>

          <button className="btn btn-primary w-full py-5 rounded-2xl shadow-red transition hover:scale-[1.02] mt-8" onClick={() => { alert('Cadastro Feito!'); setCurrentScreen('home'); }}>
            <span className="font-bold text-lg">Finalizar Cadastro</span> <i className="ph ph-check-circle text-2xl"></i>
          </button>
        </div>
      </div>
    </div>
  );
}
