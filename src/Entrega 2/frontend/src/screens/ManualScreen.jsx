import React, { useState } from 'react';
import { useAppState } from '../context/appStateContextValue';

// Inserção manual: o operador digita nome + peso, monta uma lista temporária
// e ao finalizar empurra tudo de uma vez para o grupo selecionado.
export default function ManualScreen() {
  const { appState, setAppState, addToast, setCurrentScreen, activeGroupId: groupId } = useAppState();
  const manualGroup = appState.find(g => g.id === groupId);

  const [manualItems, setManualItems] = useState([]);
  const [manualFoodName, setManualFoodName] = useState('');
  const [manualFoodWeight, setManualFoodWeight] = useState('');

  const handleAddManualItem = () => {
    const name = manualFoodName.trim();
    const weight = parseFloat(manualFoodWeight);
    if (!name) { addToast('Insira o nome do alimento.', 'error'); return; }
    if (!weight || weight <= 0) { addToast('Insira um peso válido.', 'error'); return; }
    setManualItems(prev => [...prev, { name, weight }]);
    setManualFoodName('');
    setManualFoodWeight('');
    addToast(`${name} (${weight}kg) adicionado!`, 'success');
  };

  const handleFinishManual = () => {
    if (groupId && manualItems.length > 0) {
      setAppState(prev => prev.map(group => {
        if (group.id === groupId) {
          const newItems = [...group.items, ...manualItems];
          const addedKg = manualItems.reduce((acc, obj) => acc + obj.weight, 0);
          return {
            ...group,
            items: newItems,
            totalKg: parseFloat((group.totalKg + addedKg).toFixed(2))
          };
        }
        return group;
      }));
      addToast(`${manualItems.length} itens adicionados ao grupo!`, 'success');
    }
    setCurrentScreen('dashboard');
  };

  return (
    <div className="flex flex-col h-full bg-dark">
      <header className="flex justify-between align-center p-5 border-b border-gray-medium bg-header shadow-md relative z-10">
        <div className="flex align-center gap-3">
          <button className="btn-icon circle-bg-gray border border-gray-dark shadow transition hover-bg-gray-light mr-4" onClick={() => setCurrentScreen('dashboard')}>
            <i className="ph ph-arrow-left text-xl text-white"></i>
          </button>
          <i className="ph ph-pencil-simple-line text-2xl text-primary"></i>
          <span className="font-bold text-xl text-white tracking-tight">Inserção Manual</span>
        </div>
        <div className="text-xs font-bold text-white bg-black/40 backdrop-blur px-3 py-1.5 rounded-full border border-gray-medium shadow flex align-center gap-2">
          <div className="pulse-dot" style={{ background: 'var(--primary)' }}></div>
          GRUPO {manualGroup?.title || groupId}
        </div>
      </header>

      <div className="flex-grow p-6 overflow-y-auto flex flex-col gap-6">
        {/* Input Card */}
        <div className="w-full max-w-lg" style={{ margin: '0 auto' }}>
          <div className="bg-card border border-gray-medium rounded-2xl p-6 shadow-2xl slide-up-anim">
            <div className="flex align-center gap-3 mb-6">
              <div className="circle-bg-gray flex-center" style={{ width: '48px', height: '48px', borderRadius: '50%' }}>
                <i className="ph-fill ph-bowl-food text-2xl text-primary"></i>
              </div>
              <div>
                <h2 className="font-bold text-xl text-white tracking-tight">Adicionar Alimento</h2>
                <p className="text-gray text-xs font-medium mt-1">Insira o nome e o peso do alimento.</p>
              </div>
            </div>

            <div className="flex flex-col gap-4 mb-5">
              <div className="flex flex-col gap-1.5">
                <label className="text-[10px] text-gray font-bold uppercase tracking-widest ml-1">Nome do Alimento</label>
                <input
                  type="text"
                  className="input-dark w-full"
                  placeholder="Ex: Macarrão, Arroz, Feijão..."
                  value={manualFoodName}
                  onChange={e => setManualFoodName(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleAddManualItem()}
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <label className="text-[10px] text-gray font-bold uppercase tracking-widest ml-1">Peso (kg)</label>
                <input
                  type="number"
                  className="input-dark w-full"
                  placeholder="Ex: 180"
                  min="0"
                  step="0.1"
                  value={manualFoodWeight}
                  onChange={e => setManualFoodWeight(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleAddManualItem()}
                />
              </div>
            </div>

            <button className="btn btn-outline w-full flex align-center justify-center gap-2 transition font-bold py-3 rounded-xl text-white border-gray-medium hover-bg-gray-light" onClick={handleAddManualItem}>
              <i className="ph ph-plus-circle text-lg"></i> ADICIONAR ITEM
            </button>
          </div>
        </div>

        {/* Items List */}
        <div className="w-full max-w-lg" style={{ margin: '0 auto' }}>
          <div className="bg-card border border-gray-medium rounded-2xl p-6 shadow-2xl slide-up-anim" style={{ animationDelay: '0.1s', animationFillMode: 'both' }}>
            <div className="flex justify-between align-center mb-4">
              <span className="text-xs text-gray font-bold uppercase tracking-widest">Itens Adicionados</span>
              <span className="text-xs font-bold text-primary">{manualItems.length} {manualItems.length === 1 ? 'item' : 'itens'}</span>
            </div>

            <div className="flex flex-col gap-2 mb-5" style={{ maxHeight: '280px', overflowY: 'auto' }}>
              {manualItems.length === 0 ? (
                <div className="text-center text-gray text-sm py-6 opacity-50 border border-dashed border-gray-medium rounded-lg font-medium">Nenhum item adicionado ainda.</div>
              ) : (
                manualItems.map((item, idx) => (
                  <div key={idx} className="flex justify-between align-center p-3 bg-red-light rounded-lg text-sm transition hover:shadow-md border border-red-500 border-opacity-20" style={{ border: '1px solid rgba(239, 68, 68, 0.2)' }}>
                    <span className="flex align-center gap-2 font-bold text-white">
                      <i className="ph-fill ph-check-circle text-primary text-xl"></i>
                      {item.name}
                    </span>
                    <div className="flex align-center gap-2">
                      <span className="text-xs font-black badge-red px-2 py-1 rounded-full text-white">+{item.weight}kg</span>
                      <button className="btn-icon text-gray transition hover:text-red-500" style={{ width: '24px', height: '24px' }}
                        onClick={() => setManualItems(prev => prev.filter((_, i) => i !== idx))}>
                        <i className="ph ph-trash"></i>
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>

            {manualItems.length > 0 && (
              <div className="flex align-center justify-between p-3 rounded-lg mb-4" style={{ background: 'rgba(159, 24, 24, 0.1)', border: '1px solid rgba(159, 24, 24, 0.2)' }}>
                <span className="text-sm font-bold text-gray">Total</span>
                <span className="text-xl font-black text-primary">{manualItems.reduce((acc, i) => acc + i.weight, 0).toFixed(1)}kg</span>
              </div>
            )}

            <button className="btn btn-primary w-full flex align-center justify-center gap-2 transition font-bold py-3 rounded-xl shadow-red" onClick={handleFinishManual}>
              <i className="ph ph-check-circle text-lg"></i> FINALIZAR INSERÇÃO ({manualItems.length})
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
