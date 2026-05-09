import React from 'react';

// Dropdown do header da Dashboard. Mostra dados do usuário, Configuração e Sair.
export default function UserPopup({ aberto, userData, onIrConfig, onFechar, onSair }) {
  return (
    <div
      className={`absolute mt-3 bg-card border border-gray-medium rounded-2xl shadow-2xl overflow-hidden transition ${aberto ? 'pop-up-show' : 'pop-up-hide'}`}
      style={{ zIndex: 50, cursor: 'default', right: '0', width: '260px' }}
    >
      <div className="p-4 border-b border-gray-medium bg-black/40">
        <div className="flex align-center gap-3">
          <div className="circle-bg-gray flex-center" style={{ width: '45px', height: '45px', borderRadius: '50%' }}>
            <i className="ph-fill ph-user-circle text-2xl text-primary"></i>
          </div>
          <div>
            <h4 className="font-bold text-sm text-white">{userData.nome} {userData.sobrenome}</h4>
            <p className="text-xs text-gray" style={{ userSelect: 'all' }}>{userData.email}</p>
          </div>
        </div>
      </div>
      <div className="flex flex-col p-2">
        <button className="flex align-center gap-3 w-full p-3 text-sm text-gray hover:text-white rounded-xl transition text-left popup-btn-gray" onClick={onIrConfig}>
          <i className="ph ph-gear text-xl text-primary"></i>
          <span className="font-medium">Configuração de Dados</span>
        </button>
        <div className="my-1 border-t border-gray-medium"></div>
        <button
          type="button"
          className="flex align-center gap-3 w-full p-3 text-sm text-gray rounded-xl transition text-left popup-btn-gray hover-text-white"
          onClick={() => {
            onFechar();
            if (typeof onSair === 'function') onSair();
          }}
        >
          <i className="ph ph-sign-out text-xl text-primary"></i>
          <span className="font-bold tracking-wide">Sair</span>
        </button>
      </div>
    </div>
  );
}
