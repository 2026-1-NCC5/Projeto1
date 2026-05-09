import React, { useState } from 'react';

function formatPesoKg(w) {
  const n = Number(w);
  if (!Number.isFinite(n)) return '0';
  const s = n.toFixed(3).replace(/\.?0+$/, '');
  return s || '0';
}

// Card de grupo do Kanban da Dashboard. Fluxo único: inserção manual → revisão
// na câmera (CTA único). Lista agregada por alimento + peso unitário.
export default function KanbanCard({
  group,
  onEdit,
  onIniciarTriagem,
  onAcrescentarItens,
  onAtualizarItemKanban,
  onRemoverUnidadeKanban,
}) {
  const etapa = group.etapaTriagem || 'inicio';
  const manualOk = etapa === 'manual_ok';
  const [editIdx, setEditIdx] = useState(null);
  const [editQtd, setEditQtd] = useState('1');
  const [editPeso, setEditPeso] = useState('');

  const items = Array.isArray(group.items) ? group.items : [];

  const iniciarEdicao = (idx, it) => {
    setEditIdx(idx);
    setEditQtd(String(Math.max(1, it.quantity || 1)));
    setEditPeso(String(it.weight ?? ''));
  };

  const cancelarEdicao = () => {
    setEditIdx(null);
  };

  return (
    <div className="kanban-card bg-card shadow-lg rounded-xl p-5 flex flex-col justify-between transition border-gray-medium" style={{ minHeight: '500px' }}>
      <div>
        <div className="flex justify-between align-center mb-2">
          <h3 className="font-bold text-xl text-white tracking-tight">{group.title}</h3>
          <button type="button" className="btn-icon text-gray transition hover-text-white hover-bg-gray-medium" onClick={() => onEdit(group)}>
            <i className="ph ph-pencil-simple text-lg"></i>
          </button>
        </div>
        <div className="text-xs text-gray mb-4 flex gap-2 font-medium flex-wrap">
          {(group.members || []).length > 0 ? (group.members || []).map((m, i) => (
            <span key={i} className="text-white">{m.name} <span className="text-primary font-bold">({m.ra})</span></span>
          )) : <span className="opacity-50">Sem integrantes</span>}
        </div>

        <div className="flex flex-col mt-4">
          {items.length === 0 ? (
            <div className="text-center text-gray py-8 text-sm opacity-50 border border-dashed border-gray-medium rounded-lg font-medium tracking-wide">
              {manualOk ? 'Itens já declarados — siga para a câmera para auditoria.' : 'Aguardando contagem...'}
            </div>
          ) : (
            items.map((item, idx) => (
              <div
                key={`${item.name}-${item.weight}-${idx}`}
                className="flex flex-col gap-2 p-3 bg-red-light rounded-lg text-sm transition hover:shadow-md border border-red-500 border-opacity-20 mb-3"
                style={{ border: '1px solid rgba(239, 68, 68, 0.2)' }}
              >
                {editIdx === idx ? (
                  <div className="flex flex-col gap-2 w-full">
                    <span className="font-bold text-white text-xs truncate">{item.name}</span>
                    <div className="flex flex-wrap gap-2 align-center">
                      <label className="text-[10px] text-gray font-bold uppercase">Qtd</label>
                      <input
                        type="number"
                        min="1"
                        step="1"
                        className="input-dark rounded px-2 py-1 text-sm"
                        style={{ width: '72px' }}
                        value={editQtd}
                        onChange={(e) => setEditQtd(e.target.value)}
                      />
                      <label className="text-[10px] text-gray font-bold uppercase">Peso (kg)</label>
                      <input
                        type="number"
                        min="0"
                        step="0.01"
                        className="input-dark rounded px-2 py-1 text-sm flex-1 min-w-[80px]"
                        value={editPeso}
                        onChange={(e) => setEditPeso(e.target.value)}
                      />
                    </div>
                    <div className="flex gap-2 justify-end">
                      <button type="button" className="btn btn-outline text-xs py-1 px-2 border-gray-medium text-white" onClick={cancelarEdicao}>Cancelar</button>
                      <button
                        type="button"
                        className="btn btn-primary text-xs py-1 px-2"
                        onClick={() => {
                          const q = Math.max(1, parseInt(editQtd, 10) || 1);
                          const p = parseFloat(String(editPeso).replace(',', '.'));
                          if (!p || p <= 0) return;
                          onAtualizarItemKanban?.(group.id, idx, { quantity: q, weight: p });
                          cancelarEdicao();
                        }}
                      >
                        Salvar
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="flex justify-between align-center gap-2 w-full">
                    <span className="flex align-center gap-2 font-bold text-white min-w-0">
                      <i className="ph-fill ph-check-circle text-primary text-xl flex-shrink-0"></i>
                      <span className="truncate">
                        {(item.quantity || 1) > 1 ? `${item.quantity} × ` : ''}{item.name}
                      </span>
                    </span>
                    <div className="flex align-center gap-1 flex-shrink-0">
                      <span className="text-xs font-black text-primary badge-red px-2 py-1 rounded-full text-white whitespace-nowrap">
                        +{formatPesoKg(item.weight)} kg/un.
                      </span>
                      <button
                        type="button"
                        className="btn-icon text-gray transition hover:text-white"
                        title="Editar quantidade e peso unitário"
                        onClick={() => iniciarEdicao(idx, item)}
                      >
                        <i className="ph ph-pencil-simple"></i>
                      </button>
                      <button
                        type="button"
                        className="btn-icon text-gray transition hover:text-red-400"
                        title="Remover uma unidade desta linha"
                        onClick={() => onRemoverUnidadeKanban?.(group.id, idx)}
                      >
                        <i className="ph ph-minus-circle"></i>
                      </button>
                    </div>
                  </div>
                )}
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
        <button
          type="button"
          className="btn btn-primary w-full shadow-red mt-4 py-3 rounded-xl font-bold text-md"
          onClick={() => onIniciarTriagem(group.id)}
        >
          <i className={`ph ${manualOk ? 'ph-camera' : 'ph-list-checks'} text-lg`}></i>
          {' '}
          {manualOk ? 'Revisar com Câmera' : 'Iniciar triagem'}
        </button>
        {items.length > 0 && (
          <button
            type="button"
            className="btn btn-outline w-full mt-3 py-3 rounded-xl font-bold text-md border-gray-medium text-white hover-bg-gray-light"
            onClick={() => onAcrescentarItens(group.id)}
          >
            <i className="ph ph-plus-circle text-lg text-primary"></i>
            {' '}
            Acrescentar itens
          </button>
        )}
      </div>
    </div>
  );
}
