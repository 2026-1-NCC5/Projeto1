import React from 'react';

// Card de grupo do Kanban da Dashboard. Mostra integrantes, lista de itens
// já contados, total em kg e CTAs para câmera / inserção manual.
export default function KanbanCard({ group, onEdit, onIniciarCamera, onIniciarManual }) {
  return (
    <div className="kanban-card bg-card shadow-lg rounded-xl p-5 flex flex-col justify-between transition border-gray-medium" style={{ minHeight: '500px' }}>
      <div>
        <div className="flex justify-between align-center mb-2">
          <h3 className="font-bold text-xl text-white tracking-tight">{group.title}</h3>
          <button className="btn-icon text-gray transition hover-text-white hover-bg-gray-medium" onClick={() => onEdit(group)}>
            <i className="ph ph-pencil-simple text-lg"></i>
          </button>
        </div>
        <div className="text-xs text-gray mb-4 flex gap-2 font-medium flex-wrap">
          {(group.members || []).length > 0 ? (group.members || []).map((m, i) => (
            <span key={i} className="text-white">{m.name} <span className="text-primary font-bold">({m.ra})</span></span>
          )) : <span className="opacity-50">Sem integrantes</span>}
        </div>

        <div className="flex flex-col mt-4">
          {(group.items || []).length === 0 ? (
            <div className="text-center text-gray py-8 text-sm opacity-50 border border-dashed border-gray-medium rounded-lg font-medium tracking-wide">
              Aguardando contagem...
            </div>
          ) : (
            (group.items || []).map((item, idx) => (
              <div key={idx} className="flex justify-between align-center p-3 bg-red-light rounded-lg text-sm transition hover:shadow-md border border-red-500 border-opacity-20 mb-3" style={{ border: '1px solid rgba(239, 68, 68, 0.2)' }}>
                <span className="flex align-center gap-2 font-bold text-white"><i className="ph-fill ph-check-circle text-primary text-xl"></i> {item.name}</span>
                <span className="text-xs font-black text-primary badge-red px-2 py-1 rounded-full text-white">+{item.weight}kg</span>
              </div>
            ))
          )}
        </div>
      </div>
      <div className="mt-auto pt-4 relative">
        <div className="mb-4 mt-8 flex flex-col px-1">
          <div className="font-bold text-white flex align-end gap-1">
            <span className="text-primary text-3xl" style={{ lineHeight: 1, letterSpacing: '-1px' }}>{group.totalKg}kg</span>
            <span className="text-sm pb-1 font-bold text-gray uppercase tracking-widest">Total</span>
          </div>
        </div>
        <button className="btn btn-primary w-full shadow-red mt-4 py-3 rounded-xl font-bold text-md" onClick={() => onIniciarCamera(group.id)}>
          <i className="ph ph-camera text-lg"></i> Revisar com Câmera
        </button>
        <button className="btn btn-outline w-full mt-3 py-3 rounded-xl font-bold text-md border-gray-medium text-white hover-bg-gray-light" onClick={() => onIniciarManual(group.id)}>
          <i className="ph ph-pencil-simple-line text-lg text-primary"></i> Inserção Manual
        </button>
      </div>
    </div>
  );
}
