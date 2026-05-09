import React, { useEffect, useState } from 'react';
import { useAppState } from '../context/appStateContextValue';
import {
  listarAlimentos,
  obterOuCriarSessaoAtiva,
  criarDeteccaoManual,
} from '../services/api';
import { mergeKanbanItemsAppend, totalKgFromItems } from '../utils/kanbanItems';

// Inserção manual: escolhe alimento cadastrado na API + peso, persiste em
// deteccoes e atualiza o Kanban local (ou apenas acrescenta no modo dedicado).
export default function ManualScreen() {
  const {
    appState,
    setAppState,
    addToast,
    setCurrentScreen,
    activeGroupId: groupId,
    auditSessaoIdNumero,
    setAuditSessaoId,
    manualSomenteAcrescentar,
    setManualSomenteAcrescentar,
  } = useAppState();
  const manualGroup = appState.find((g) => g.id === groupId);
  const grupoBackendId = manualGroup?.grupoIdBackend;

  const [alimentos, setAlimentos] = useState([]);
  const [manualItems, setManualItems] = useState([]);
  const [alimentoSelecionadoId, setAlimentoSelecionadoId] = useState('');
  const [manualFoodWeight, setManualFoodWeight] = useState('');
  const [manualQuantidade, setManualQuantidade] = useState('1');

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const r = await listarAlimentos();
      if (cancelled) return;
      if (!r.ok || !Array.isArray(r.data)) {
        addToast('Não foi possível carregar alimentos do servidor.', 'error');
        return;
      }
      setAlimentos(r.data.filter((a) => a.ativo !== false));
    })();
    return () => { cancelled = true; };
  }, [addToast]);

  useEffect(() => {
    if (!groupId || grupoBackendId == null) return;
    let cancelled = false;
    (async () => {
      const r = await obterOuCriarSessaoAtiva(grupoBackendId);
      if (cancelled) return;
      if (r.erro) {
        addToast(r.erro, 'error');
        return;
      }
      if (r.sessaoId != null) {
        setAuditSessaoId(String(r.sessaoId));
        if (r.reutilizada) addToast('Sessão ativa deste grupo reaberta.', 'info');
      }
    })();
    return () => { cancelled = true; };
  }, [groupId, grupoBackendId, addToast, setAuditSessaoId]);

  const handleAddManualItem = () => {
    const idInt = parseInt(alimentoSelecionadoId, 10);
    const al = alimentos.find((x) => x.id === idInt);
    const weight = parseFloat(String(manualFoodWeight).replace(',', '.'));
    const qtd = Math.max(1, parseInt(manualQuantidade, 10) || 1);
    if (!al) { addToast('Selecione um alimento da lista.', 'error'); return; }
    if (!weight || weight <= 0) { addToast('Insira um peso válido.', 'error'); return; }
    setManualItems((prev) => [...prev, {
      alimentoId: al.id,
      grupoAlimentoId: al.grupo_alimento_id,
      name: al.nome,
      weight,
      quantity: qtd,
    }]);
    setAlimentoSelecionadoId('');
    setManualFoodWeight('');
    setManualQuantidade('1');
    addToast(`${qtd} × ${al.nome} (${weight} kg/un.) na fila.`, 'success');
  };

  const handleFinishManual = async () => {
    if (!groupId) {
      setCurrentScreen('dashboard');
      return;
    }

    if (manualSomenteAcrescentar) {
      if (manualItems.length === 0) {
        setManualSomenteAcrescentar(false);
        setCurrentScreen('dashboard');
        return;
      }
      const mapped = manualItems.map((it) => ({
        name: it.name,
        weight: it.weight,
        quantity: it.quantity || 1,
        alimentoId: it.alimentoId,
        grupoAlimentoId: it.grupoAlimentoId,
      }));
      setAppState((prev) => prev.map((group) => {
        if (group.id !== groupId) return group;
        const merged = mergeKanbanItemsAppend(group.items || [], mapped);
        const tk = totalKgFromItems(merged);
        return {
          ...group,
          items: merged,
          totalKg: parseFloat(tk.toFixed(2)),
        };
      }));
      if (auditSessaoIdNumero > 0) {
        let okAll = true;
        for (const it of manualItems) {
          const n = Math.max(1, it.quantity || 1);
          for (let i = 0; i < n; i += 1) {
            const r = await criarDeteccaoManual({
              sessaoIdNumero: auditSessaoIdNumero,
              alimentoId: it.alimentoId,
              peso_kg: it.weight,
              fonte: 'MANUAL',
            });
            if (!r.ok) okAll = false;
          }
        }
        if (!okAll) addToast('Alguns itens não foram gravados no servidor.', 'warning');
        else addToast('Novos itens registrados no servidor.', 'success');
      }
      setManualSomenteAcrescentar(false);
      addToast('Lista do grupo atualizada.', 'success');
      setCurrentScreen('dashboard');
      return;
    }

    if (manualItems.length === 0) {
      addToast('Adicione pelo menos um item antes de continuar.', 'warning');
      return;
    }
    if (auditSessaoIdNumero <= 0) {
      addToast('Sessão inválida. Abra o painel e tente novamente.', 'error');
      setCurrentScreen('dashboard');
      return;
    }

    let okAll = true;
    for (const it of manualItems) {
      const n = Math.max(1, it.quantity || 1);
      for (let i = 0; i < n; i += 1) {
        const r = await criarDeteccaoManual({
          sessaoIdNumero: auditSessaoIdNumero,
          alimentoId: it.alimentoId,
          peso_kg: it.weight,
          fonte: 'MANUAL',
        });
        if (!r.ok) okAll = false;
      }
    }
    if (!okAll) {
      addToast('Alguns itens não foram gravados no servidor.', 'warning');
    } else {
      addToast(`${manualItems.reduce((a, x) => a + Math.max(1, x.quantity || 1), 0)} registros salvos no servidor.`, 'success');
    }

    setAppState((prev) => prev.map((group) => {
      if (group.id !== groupId) return group;
      return {
        ...group,
        etapaTriagem: 'manual_ok',
        triagemSessaoId: auditSessaoIdNumero,
      };
    }));

    addToast('Inserção manual registrada. Abrindo revisão na câmera.', 'info');
    setCurrentScreen('camera');
  };

  return (
    <div className="flex flex-col h-full bg-dark">
      <header className="flex justify-between align-center gap-3 p-5 border-b border-gray-medium bg-header shadow-md relative z-10 shrink-0">
        <div className="flex align-center gap-3 min-w-0 flex-1">
          <button type="button" className="btn-icon circle-bg-gray border border-gray-dark shadow transition hover-bg-gray-light mr-4 flex-shrink-0" onClick={() => { setManualSomenteAcrescentar(false); setCurrentScreen('dashboard'); }}>
            <i className="ph ph-arrow-left text-xl text-white"></i>
          </button>
          <i className="ph ph-pencil-simple-line text-2xl text-primary flex-shrink-0"></i>
          <span className="font-bold text-xl text-white tracking-tight truncate">
            {manualSomenteAcrescentar ? 'Acrescentar itens' : 'Inserção Manual'}
          </span>
        </div>
        <div className="sessao-header-badge sessao-header-badge--manual border border-gray-medium shadow">
          <div className="pulse-dot flex-shrink-0" style={{ background: 'var(--primary)' }}></div>
          <div className="sessao-header-badge-text">
            <span className="sessao-header-badge-group">{manualGroup?.title || groupId}</span>
            {auditSessaoIdNumero > 0 && (
              <>
                <span className="sessao-header-badge-sep" aria-hidden="true">·</span>
                <span className="sessao-header-badge-sessao text-gray">sessão #{auditSessaoIdNumero}</span>
              </>
            )}
          </div>
        </div>
      </header>

      <div className="manual-screen-body flex-grow flex flex-col min-h-0 overflow-y-auto w-full px-8">
        <div className="manual-screen-inner flex flex-col gap-6 w-full max-w-lg mx-auto my-auto">
          <div className="bg-card border border-gray-medium rounded-2xl p-6 shadow-2xl slide-up-anim">
            <div className="flex align-center gap-3 mb-8">
              <div className="circle-bg-gray flex-center" style={{ width: '48px', height: '48px', borderRadius: '50%' }}>
                <i className="ph-fill ph-bowl-food text-2xl text-primary"></i>
              </div>
              <div>
                <h2 className="font-bold text-xl text-white tracking-tight">Adicionar Alimento</h2>
                <p className="text-gray text-xs font-medium mt-1">Alimentos cadastrados no servidor (API /alimentos).</p>
              </div>
            </div>

            <div className="manual-add-fields mb-5">
              <div className="manual-add-field-col min-w-0">
                <label className="text-[10px] text-gray font-bold uppercase tracking-widest ml-1">Alimento</label>
                <select
                  className="input-dark w-full rounded-lg"
                  value={alimentoSelecionadoId}
                  onChange={(e) => {
                    const v = e.target.value;
                    setAlimentoSelecionadoId(v);
                    const a = alimentos.find((x) => String(x.id) === v);
                    if (a) setManualFoodWeight(String(a.peso_padrao_kg ?? ''));
                  }}
                >
                  <option value="">Selecione…</option>
                  {alimentos.map((a) => (
                    <option key={a.id} value={a.id}>{a.nome}</option>
                  ))}
                </select>
              </div>
              <div className="manual-add-field-col manual-add-fields-peso min-w-0">
                <label className="text-[10px] text-gray font-bold uppercase tracking-widest ml-1">Peso unitário (kg)</label>
                <div className="manual-weight-field">
                  <input
                    type="number"
                    placeholder="Ex.: 0.5"
                    min="0"
                    step="0.01"
                    value={manualFoodWeight}
                    onChange={(e) => setManualFoodWeight(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleAddManualItem()}
                  />
                  <span className="manual-weight-field-suffix" aria-hidden="true">kg</span>
                </div>
              </div>
              <div className="manual-add-field-col manual-add-fields-qtd min-w-0">
                <label className="text-[10px] text-gray font-bold uppercase tracking-widest ml-1">Quantidade</label>
                <input
                  type="number"
                  className="input-dark w-full rounded-lg"
                  min="1"
                  step="1"
                  placeholder="1"
                  value={manualQuantidade}
                  onChange={(e) => setManualQuantidade(e.target.value)}
                />
              </div>
            </div>

            <button type="button" className="btn btn-outline w-full flex align-center justify-center gap-2 transition font-bold py-3 rounded-xl text-white border-gray-medium hover-bg-gray-light" onClick={handleAddManualItem}>
              <i className="ph ph-plus-circle text-lg"></i> ADICIONAR À FILA
            </button>
          </div>

          <div className="bg-card border border-gray-medium rounded-2xl p-6 shadow-2xl slide-up-anim" style={{ animationDelay: '0.1s', animationFillMode: 'both' }}>
            <div className="flex justify-between align-center mb-4">
              <span className="text-xs text-gray font-bold uppercase tracking-widest">Itens na fila</span>
              <span className="text-xs font-bold text-primary">
                {manualItems.reduce((a, x) => a + Math.max(1, x.quantity || 1), 0)}
                {' '}
                unidade(s)
              </span>
            </div>

            <div className="flex flex-col gap-2 mb-5" style={{ maxHeight: '280px', overflowY: 'auto' }}>
              {manualItems.length === 0 ? (
                <div className="text-center text-gray text-sm py-6 opacity-50 border border-dashed border-gray-medium rounded-lg font-medium">Nenhum item na fila.</div>
              ) : (
                manualItems.map((item, idx) => (
                  <div key={`${item.alimentoId}-${idx}`} className="flex justify-between align-center p-3 bg-red-light rounded-lg text-sm transition hover:shadow-md border border-red-500 border-opacity-20" style={{ border: '1px solid rgba(239, 68, 68, 0.2)' }}>
                    <span className="flex align-center gap-2 font-bold text-white min-w-0">
                      <i className="ph-fill ph-check-circle text-primary text-xl flex-shrink-0"></i>
                      <span className="truncate">
                        {(item.quantity || 1) > 1 ? `${item.quantity} × ` : ''}{item.name}
                      </span>
                    </span>
                    <div className="flex align-center gap-2 flex-shrink-0">
                      <span className="text-xs font-black badge-red px-2 py-1 rounded-full text-white whitespace-nowrap">
                        {item.weight} kg/un.
                      </span>
                      <button type="button" className="btn-icon text-gray transition hover:text-red-500" style={{ width: '24px', height: '24px' }}
                        onClick={() => setManualItems((prev) => prev.filter((_, i) => i !== idx))}>
                        <i className="ph ph-trash"></i>
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>

            {manualItems.length > 0 && (
              <div className="flex align-center justify-between p-3 rounded-lg mb-4" style={{ background: 'rgba(159, 24, 24, 0.1)', border: '1px solid rgba(159, 24, 24, 0.2)' }}>
                <span className="text-sm font-bold text-gray">Total estimado</span>
                <span className="text-xl font-black text-primary">
                  {manualItems.reduce((acc, i) => acc + i.weight * Math.max(1, i.quantity || 1), 0).toFixed(2)}
                  kg
                </span>
              </div>
            )}

            <button type="button" className="btn btn-primary w-full flex align-center justify-center gap-2 transition font-bold py-3 rounded-xl shadow-red" onClick={handleFinishManual}>
              <i className="ph ph-check-circle text-lg"></i>
              {' '}
              {manualSomenteAcrescentar
                ? `SALVAR NO GRUPO (${manualItems.reduce((a, x) => a + Math.max(1, x.quantity || 1), 0)})`
                : `CONTINUAR PARA A CÂMERA (${manualItems.reduce((a, x) => a + Math.max(1, x.quantity || 1), 0)})`}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
