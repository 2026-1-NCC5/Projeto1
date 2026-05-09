import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useAppState } from '../context/appStateContextValue';
import useAuditoriaWS from '../hooks/useAuditoriaWS';
import useCameraStream from '../hooks/useCameraStream';
import {
  criarDeteccaoManual,
  corrigirDeteccao,
  excluirDeteccao,
  finalizarSessao,
  listarAlimentos,
  obterOuCriarSessaoAtiva,
  obterConciliacaoPreviaSessao,
  decidirFonteFinalSessao,
} from '../services/api';
import { WS_BASE } from '../constants';
import DetectionPopup from '../components/DetectionPopup';
import ScannerLogPanel from '../components/ScannerLogPanel';
import SessionItemsPanel from '../components/SessionItemsPanel';
import FinalizacaoConciliacaoModal from '../components/FinalizacaoConciliacaoModal';
import RevisaoDeteccaoModal from '../components/RevisaoDeteccaoModal';
import {
  buildAggregatedItemsFromRelatorio,
  buildAggregatedItemsFromCapturasSession,
  totalKgFromItems,
} from '../utils/kanbanItems';
import soundSuccessUrl from '../assets/sound-success.wav';
import soundWarningUrl from '../assets/sound-warning-error.wav';

// Tela do scanner: liga câmera, abre WS de auditoria, exibe overlay de
// bbox/preview YOLO, popup de detecção fixo na base e painel de logs.
//
// Fluxo assíncrono: a captura nunca pausa. Cada gatilho YOLO gera uma
// `deteccao_preliminar` que entra no scoreboard com chip "validando"; o
// Gemini roda em background no backend e dispara `deteccao_atualizada`,
// que troca o chip para "validado" / "corrigido" / "sem_gemini".
export default function CameraScreen() {
  const {
    appState,
    activeGroupId,
    auditSessaoId, setAuditSessaoId, auditSessaoIdNumero,
    addToast, setCurrentScreen, setAppState,
  } = useAppState();

  const activeGroup = appState.find((g) => g.id === activeGroupId);
  const grupoBackendId = activeGroup?.grupoIdBackend;

  useEffect(() => {
    setCurrentSessionItems([]);
  }, [activeGroupId]);

  // Mantém a mesma sessão em que o manual gravou as deteccoes MANUAL.
  // Se chamarmos obterOuCriarSessaoAtiva depois do manual, a API pode devolver
  // outra sessão (ex.: POST criou sessão nova) e o relatório de conciliação
  // deixa de ver os itens declarados — só as capturas da sessão atual.
  useEffect(() => {
    if (!activeGroupId || grupoBackendId == null) return;
    const sessaoDoManual =
      activeGroup?.etapaTriagem === 'manual_ok' && activeGroup?.triagemSessaoId != null
        ? Number(activeGroup.triagemSessaoId)
        : NaN;
    if (Number.isFinite(sessaoDoManual) && sessaoDoManual > 0) {
      setAuditSessaoId(String(sessaoDoManual));
      return;
    }
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
  }, [
    activeGroupId,
    grupoBackendId,
    activeGroup?.etapaTriagem,
    activeGroup?.triagemSessaoId,
    addToast,
    setAuditSessaoId,
  ]);

  const videoRef = useRef(null);

  const [currentSessionItems, setCurrentSessionItems] = useState([]);
  const [usarGemini, setUsarGemini] = useState(true);
  const [leituraNome, setLeituraNome] = useState('Consultando Alimento...');
  const [leituraDesc, setLeituraDesc] = useState('Clique em iniciar captura para enviar frames ao modelo.');
  const [btnLeituraText, setBtnLeituraText] = useState('AGUARDANDO DETECÇÃO');
  const [currFood, setCurrFood] = useState('');
  const [currWeight, setCurrWeight] = useState(0);
  // Flag visual: aplica um "flash" no popup principal quando uma detecção
  // nova chega via WS, para chamar atenção do operador.
  const [destaquePopup, setDestaquePopup] = useState(false);
  const [mostrarLogsAuditoria, setMostrarLogsAuditoria] = useState(true);
  const [popupMinimizado, setPopupMinimizado] = useState(false);
  const [modalConciliacaoAberto, setModalConciliacaoAberto] = useState(false);
  const [modalCarregando, setModalCarregando] = useState(false);
  const [modalSalvando, setModalSalvando] = useState(false);
  const [modalErro, setModalErro] = useState('');
  const [relatorioConciliacao, setRelatorioConciliacao] = useState(null);
  const [revisaoModalDeteccaoId, setRevisaoModalDeteccaoId] = useState(null);
  const [salvandoRevisao, setSalvandoRevisao] = useState(false);
  const [alimentosCatalogo, setAlimentosCatalogo] = useState([]);

  const soundSuccessRef = useRef(null);
  const soundWarningRef = useRef(null);

  useEffect(() => {
    soundSuccessRef.current = new Audio(soundSuccessUrl);
    soundWarningRef.current = new Audio(soundWarningUrl);
    return () => {
      soundSuccessRef.current = null;
      soundWarningRef.current = null;
    };
  }, []);

  const playSuccessSound = useCallback(() => {
    const a = soundSuccessRef.current;
    if (!a) return;
    a.currentTime = 0;
    void a.play().catch(() => {});
  }, []);

  const playWarningSound = useCallback(() => {
    const a = soundWarningRef.current;
    if (!a) return;
    a.currentTime = 0;
    void a.play().catch(() => {});
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const r = await listarAlimentos();
      if (cancelled) return;
      if (!r.ok || !Array.isArray(r.data)) return;
      setAlimentosCatalogo(r.data.filter((a) => a.ativo !== false));
    })();
    return () => { cancelled = true; };
  }, []);

  // ===== AUDITORIA WS (câmera ↔ FastAPI) =====
  // O hook só conecta quando a tela é 'camera' e há um sessao_id válido.
  const {
    status: auditStatus,
    ultimaPreliminar,
    ultimaAtualizacaoGemini,
    ultimoPreview,
    logs: auditLogs,
    capturando,
    iniciarCaptura: iniciarCapturaWS,
    pararCaptura: pararCapturaWS,
    limparLogs: limparAuditLogs,
    registrarLog: registrarAuditLog,
  } = useAuditoriaWS({
    sessaoId: auditSessaoIdNumero > 0 ? auditSessaoId : null,
    videoRef,
    wsBaseUrl: WS_BASE,
    fps: 2,
    ativo: true,
    usarGemini,
  });

  const yoloPreview = ultimoPreview?.yolo || null;
  const previewBBox = yoloPreview?.bbox || null;
  const previewLabel = yoloPreview
    ? `${yoloPreview.classe} · ${Math.round((yoloPreview.confianca || 0) * 100)}%`
    : null;
  const bboxStyle = previewBBox ? {
    left: `${(previewBBox[0] / 640) * 100}%`,
    top: `${(previewBBox[1] / 480) * 100}%`,
    width: `${Math.max(1, ((previewBBox[2] - previewBBox[0]) / 640) * 100)}%`,
    height: `${Math.max(1, ((previewBBox[3] - previewBBox[1]) / 480) * 100)}%`,
  } : null;

  // 1) Quando o backend confirma a detecção preliminar (YOLO), adicionamos o
  //    item na lista imediatamente com chip "validando". A captura segue
  //    sem pausa — o Gemini é processado em background no backend.
  useEffect(() => {
    if (!ultimaPreliminar) return;
    const rf = ultimaPreliminar.resultado_final || {};
    const yolo = ultimaPreliminar.yolo || {};
    const peso = rf.peso_padrao_kg || 0;
    const nome = rf.alimento_nome || yolo.classe || 'Desconhecido';
    const deteccaoId = ultimaPreliminar.deteccao_id;

    playSuccessSound();

    setCurrentSessionItems(prev => [
      ...prev,
      {
        deteccaoId,
        name: nome,
        weight: peso,
        status: 'validando',
        nomeOriginal: nome,
        alimento_id: rf.alimento_id,
        alimento_id_original: rf.alimento_id,
        imagem_path: ultimaPreliminar.imagem_path,
        revisaoManualPendente: false,
      },
    ]);

    setDestaquePopup(true);
    setLeituraNome(nome);
    setLeituraDesc(`Último: ${nome} · +${Number(peso).toFixed(1)}kg`);
    setBtnLeituraText('AGUARDANDO DETECÇÃO');
    setCurrFood('');
    setCurrWeight(0);
    addToast(`+${Number(peso).toFixed(1)}kg ${nome} capturado`, 'success');
  // Reagimos exclusivamente à chegada de uma preliminar nova; demais setters
  // são estáveis e o efeito deve rodar exatamente uma vez por detecção.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ultimaPreliminar, playSuccessSound]);

  // 2) Quando o Gemini conclui (ou o backend sinaliza que está OFF),
  //    procuramos o item correspondente pelo `deteccaoId` e atualizamos o
  //    chip silenciosamente. Se o Gemini propôs uma classe diferente,
  //    renomeamos o item e marcamos como "corrigido".
  useEffect(() => {
    if (!ultimaAtualizacaoGemini) return;
    const {
      deteccao_id,
      gemini,
      alimento_nome,
      alimento_id,
      fonte,
      peso_kg: pesoKgWs,
      revisao_manual_pendente: revPend,
    } = ultimaAtualizacaoGemini;

    let deveAvisar = false;
    if (revPend === true) {
      deveAvisar = true;
    } else if (gemini) {
      let concordaClasse = gemini.concorda_classe;
      let concordaPeso = gemini.concorda_peso;
      if (concordaClasse === undefined && gemini) {
        concordaClasse = gemini.concorda;
      }
      if (concordaPeso === undefined) {
        concordaPeso = true;
      }
      if (concordaClasse === false || concordaPeso === false) {
        deveAvisar = true;
      }
    }
    if (deveAvisar) playWarningSound();

    setCurrentSessionItems((prev) => {
      const idx = prev.findIndex((it) => it.deteccaoId === deteccao_id);
      if (idx < 0) return prev;
      const atual = prev[idx];
      let novoStatus = atual.status;
      let novoNome = atual.name;
      let novoAlimentoId = atual.alimento_id;
      let novoPeso = atual.weight;
      if (pesoKgWs != null && Number.isFinite(Number(pesoKgWs))) {
        novoPeso = Number(pesoKgWs);
      }
      if (revPend === true) {
        novoStatus = 'revisao_pendente';
      } else if (!gemini) {
        novoStatus = 'sem_gemini';
      } else {
        let concordaClasse = gemini?.concorda_classe;
        let concordaPeso = gemini?.concorda_peso;
        if (concordaClasse === undefined && gemini) {
          concordaClasse = gemini.concorda;
        }
        if (concordaPeso === undefined) {
          concordaPeso = true;
        }
        const pesoMudou = Math.abs(Number(novoPeso) - Number(atual.weight)) > 1e-3;
        if (concordaClasse && concordaPeso) {
          novoStatus = 'validado';
        } else if (alimento_nome) {
          novoStatus = 'corrigido';
          novoNome = alimento_nome;
          novoAlimentoId = alimento_id ?? atual.alimento_id;
        } else if (pesoMudou) {
          novoStatus = 'corrigido';
        } else {
          novoStatus = 'sem_gemini';
        }
      }
      const atualizado = {
        ...atual,
        status: novoStatus,
        name: novoNome,
        weight: novoPeso,
        alimento_id: novoAlimentoId,
        gemini,
        fonte: fonte || atual.fonte,
        revisaoManualPendente: revPend === true,
      };
      const cpy = prev.slice();
      cpy[idx] = atualizado;
      return cpy;
    });
  }, [ultimaAtualizacaoGemini, playWarningSound]);

  // Remove o flash do popup depois da animação (~700ms).
  useEffect(() => {
    if (!destaquePopup) return undefined;
    const t = setTimeout(() => setDestaquePopup(false), 720);
    return () => clearTimeout(t);
  }, [destaquePopup]);

  const pesoTotalSessao = useMemo(
    () => currentSessionItems.reduce((acc, it) => acc + (it.weight || 0), 0),
    [currentSessionItems]
  );

  const itemEmRevisao = useMemo(
    () => currentSessionItems.find((it) => it.deteccaoId === revisaoModalDeteccaoId) || null,
    [currentSessionItems, revisaoModalDeteccaoId]
  );

  const handleRevisarItem = (index) => {
    const it = currentSessionItems[index];
    if (!it?.deteccaoId) return;
    setRevisaoModalDeteccaoId(it.deteccaoId);
  };

  const handleSalvarRevisao = async ({ alimentoId, pesoKg }) => {
    const targetId = revisaoModalDeteccaoId;
    if (targetId == null) return;
    setSalvandoRevisao(true);
    try {
      const r = await corrigirDeteccao(targetId, { alimentoId, pesoKg });
      if (!r.ok) {
        const det = r.data?.detail;
        let msg = `Falha ao salvar (${r.status})`;
        if (typeof det === 'string') msg = det;
        else if (Array.isArray(det)) msg = det.map((x) => x?.msg || '').filter(Boolean).join(' ') || msg;
        addToast(msg, 'error');
        return;
      }
      setCurrentSessionItems((prev) => {
        const idx = prev.findIndex((x) => x.deteccaoId === targetId);
        if (idx < 0) return prev;
        const cur = prev[idx];
        const al = alimentosCatalogo.find((x) => x.id === alimentoId);
        const nomeNovo = al?.nome || cur.name;
        const mudouCat = alimentoId !== cur.alimento_id;
        const cpy = prev.slice();
        cpy[idx] = {
          ...cur,
          name: nomeNovo,
          weight: pesoKg,
          alimento_id: alimentoId,
          status: mudouCat ? 'corrigido' : 'validado',
          revisaoManualPendente: false,
          nomeOriginal: mudouCat ? cur.name : cur.nomeOriginal,
        };
        return cpy;
      });
      addToast('Correção registrada.', 'success');
      setRevisaoModalDeteccaoId(null);
    } finally {
      setSalvandoRevisao(false);
    }
  };

  // Câmera local (getUserMedia). O envio de frames é feito pelo useAuditoriaWS.
  useCameraStream({
    ativo: true,
    videoRef,
    onAberta: () => registrarAuditLog?.({ stage: 'camera', mensagem: 'Câmera aberta. Captura ainda pausada.' }),
    onErro: (err) => registrarAuditLog?.({ stage: 'camera', mensagem: 'Falha ao acessar câmera', dados: { erro: String(err) } }),
  });

  const handleToggleCapture = () => {
    if (capturando) {
      pararCapturaWS();
      return;
    }
    if (auditSessaoIdNumero <= 0) {
      addToast('Preparando sessão ativa... tente novamente em instantes.', 'warning');
      return;
    }
    iniciarCapturaWS();
  };

  // Fallback manual: registra um item caso o operador queira adicionar algo
  // sem detecção automática (ou caso queira corrigir após digitar). Como a
  // captura não pausa mais, este handler só persiste via REST manual e
  // adiciona o item localmente; nada mais.
  const handleSimularClick = async () => {
    if (!currFood || currWeight <= 0) return;

    const novoItem = {
      deteccaoId: null,
      name: currFood,
      weight: currWeight,
      status: 'sem_gemini',
      nomeOriginal: currFood,
      alimento_id: null,
      alimento_id_original: null,
      fonte: 'MANUAL',
    };
    setCurrentSessionItems(prev => [...prev, novoItem]);

    if (auditSessaoIdNumero > 0 && ultimaPreliminar?.resultado_final?.alimento_id) {
      try {
        const r = await criarDeteccaoManual({
          sessaoIdNumero: auditSessaoIdNumero,
          alimentoId: ultimaPreliminar.resultado_final.alimento_id,
          peso_kg: currWeight,
          alimentoIdOriginal: ultimaPreliminar.resultado_final.alimento_id,
          fonte: 'MANUAL',
        });
        if (!r.ok) {
          addToast(`Falha ao registrar manual (${r.status})`, 'error');
        } else {
          addToast(`+${currWeight}kg ${currFood} registrado`, 'success');
        }
      } catch {
        addToast('Backend indisponível — registrado localmente.', 'warning');
      }
    }

    setCurrFood('');
    setCurrWeight(0);
    setLeituraNome('Buscando próximo...');
    setLeituraDesc('--');
    setBtnLeituraText('AGUARDANDO DETECÇÃO');
  };

  const aplicarNoKanban = (fonteFinal, relatorio) => {
    if (!activeGroupId) return;
    if (fonteFinal === 'manual' && relatorio) {
      const itemsFinal = buildAggregatedItemsFromRelatorio(relatorio);
      const tk = totalKgFromItems(itemsFinal);
      setAppState((prev) => prev.map((group) => {
        if (group.id !== activeGroupId) return group;
        return {
          ...group,
          items: itemsFinal,
          totalKg: parseFloat(tk.toFixed(2)),
          etapaTriagem: 'inicio',
          triagemSessaoId: null,
        };
      }));
      return;
    }

    if (fonteFinal === 'capturas') {
      const itemsFinal = buildAggregatedItemsFromCapturasSession(currentSessionItems);
      const tk = totalKgFromItems(itemsFinal);
      setAppState((prev) => prev.map((group) => {
        if (group.id !== activeGroupId) return group;
        return {
          ...group,
          items: itemsFinal,
          totalKg: parseFloat(tk.toFixed(2)),
          etapaTriagem: 'inicio',
          triagemSessaoId: null,
        };
      }));
    }
  };

  const finalizarComFonte = async (fonteFinal) => {
    if (auditSessaoIdNumero <= 0) {
      addToast('Sessão inválida para finalizar.', 'error');
      return;
    }
    setModalSalvando(true);
    try {
      const decisaoResp = await decidirFonteFinalSessao({
        sessaoIdNumero: auditSessaoIdNumero,
        fonteFinal,
      });
      if (!decisaoResp.ok) {
        addToast(`Falha ao salvar decisão final (${decisaoResp.status})`, 'error');
        return;
      }
      aplicarNoKanban(fonteFinal, relatorioConciliacao);
      const r = await finalizarSessao(auditSessaoIdNumero);
      if (!r.ok) {
        addToast(`Falha ao finalizar sessão (${r.status})`, 'error');
        return;
      }
      addToast('Sessão finalizada e consolidada.', 'success');
      setAuditSessaoId('');
      pararCapturaWS();
      setModalConciliacaoAberto(false);
      setCurrentScreen('dashboard');
    } catch {
      addToast('Backend indisponível ao finalizar conferência.', 'warning');
    } finally {
      setModalSalvando(false);
    }
  };

  const handleAbrirConciliacao = async () => {
    if (auditSessaoIdNumero <= 0) {
      addToast('Sessão inválida. Aguarde a abertura da sessão ativa.', 'error');
      return;
    }
    setModalConciliacaoAberto(true);
    setModalErro('');
    setRelatorioConciliacao(null);
    setModalCarregando(true);
    try {
      const preview = await obterConciliacaoPreviaSessao(auditSessaoIdNumero);
      if (!preview.ok || !preview.data) {
        setModalErro('Não foi possível carregar a conferência desta sessão.');
        return;
      }
      setRelatorioConciliacao(preview.data);
    } catch {
      setModalErro('Erro ao comparar declarado e capturado.');
    } finally {
      setModalCarregando(false);
    }
  };

  const handleFinishCount = async () => {
    await handleAbrirConciliacao();
  };

  const handleConfirmarManual = async () => {
    await finalizarComFonte('manual');
  };

  const handleConfirmarCapturas = async () => {
    await finalizarComFonte('capturas');
  };

  const handleSair = () => {
    pararCapturaWS();
    setCurrentScreen('dashboard');
  };

  const handleRemoverItem = async (index) => {
    const it = currentSessionItems[index];
    if (!it) return;
    if (it.deteccaoId != null) {
      const r = await excluirDeteccao(it.deteccaoId);
      if (!r.ok && r.status !== 404) {
        addToast('Não foi possível remover o registro no servidor.', 'error');
        return;
      }
    }
    setCurrentSessionItems((prev) => prev.filter((_, i) => i !== index));
  };

  return (
    <div className="flex flex-col h-full relative overflow-hidden bg-black">
      <div className="camera-feed-container">
        <video ref={videoRef} id="camera-feed" autoPlay playsInline muted
          onLoadedData={() => registrarAuditLog?.({ stage: 'camera', mensagem: 'Vídeo pronto para captura manual' })}></video>
        {bboxStyle && (
          <div className="yolo-bbox" style={bboxStyle}>
            <span>{previewLabel}</span>
          </div>
        )}
      </div>

      <header className="flex justify-between align-center p-5 absolute top-0 w-full z-20 bg-gradient-to-b from-black/80 to-transparent">
        <button className="btn-icon circle-bg-dark border border-gray-medium shadow-sm transition hover:bg-white/10" onClick={handleSair}>
          <i className="ph ph-arrow-left text-white text-xl"></i>
        </button>
        <div className="sessao-header-badge sessao-header-badge--camera border border-gray-medium shadow flex-shrink-0">
          <div className="pulse-dot flex-shrink-0"></div>
          <div className="sessao-header-badge-text">
            <span className="sessao-header-badge-group">{activeGroup?.title || activeGroupId}</span>
            <span className="sessao-header-badge-sep" aria-hidden="true">·</span>
            <span className="sessao-header-badge-sessao text-gray">
              sessão #{auditSessaoIdNumero > 0 ? auditSessaoIdNumero : '...'}
            </span>
            <span className="sessao-header-badge-sep" aria-hidden="true">·</span>
            <span className="sessao-header-badge-status">{auditStatus.toUpperCase()}</span>
          </div>
        </div>
        <div style={{ width: '36px' }}></div>
      </header>

      <div className="scanner-control-panel">
        <button
          className={`btn ${capturando ? 'btn-outline' : 'btn-primary'} shadow-red`}
          onClick={handleToggleCapture}
        >
          <i className={`ph ${capturando ? 'ph-pause-circle' : 'ph-play-circle'} text-lg`}></i>
          <span>{capturando ? 'Pausar captura' : 'Iniciar captura'}</span>
        </button>
        <button
          className={`scanner-toggle ${usarGemini ? 'active' : ''}`}
          onClick={() => setUsarGemini(prev => !prev)}
          type="button"
        >
          <span className="scanner-toggle-dot"></span>
          Gemini {usarGemini ? 'ON' : 'OFF'}
        </button>
        <button
          className={`scanner-toggle ${capturando ? 'active' : ''}`}
          onClick={handleToggleCapture}
          type="button"
        >
          <span className="scanner-toggle-dot"></span>
          Modelo {capturando ? 'capturando' : 'pausado'}
        </button>
        <button
          className={`scanner-toggle ${mostrarLogsAuditoria ? 'active' : ''}`}
          onClick={() => setMostrarLogsAuditoria((v) => !v)}
          type="button"
          title={mostrarLogsAuditoria ? 'Ocultar painel de logs ao vivo' : 'Mostrar painel de logs ao vivo'}
        >
          <span className="scanner-toggle-dot"></span>
          Logs {mostrarLogsAuditoria ? 'visíveis' : 'ocultos'}
        </button>
      </div>

      {capturando && (
        <div className="camera-capturing-indicator-host">
          <div className="camera-capturing-indicator" role="status" aria-live="polite">
            <span className="pulse-dot" aria-hidden></span>
            <span>Capturando</span>
          </div>
        </div>
      )}

      <DetectionPopup
        capturando={capturando}
        leituraNome={leituraNome}
        leituraDesc={leituraDesc}
        previewLabel={previewLabel}
        btnLeituraText={btnLeituraText}
        currentSessionItems={currentSessionItems}
        destaque={destaquePopup}
        minimizado={popupMinimizado}
        onSimular={handleSimularClick}
        onToggleMinimizado={() => setPopupMinimizado((v) => !v)}
        onRemoverItem={handleRemoverItem}
        onFinalizar={handleFinishCount}
        onRevisarItem={handleRevisarItem}
      />

      <SessionItemsPanel
        items={currentSessionItems}
        pesoTotal={pesoTotalSessao}
        onRemoverItem={handleRemoverItem}
        onRevisarItem={handleRevisarItem}
      />

      {mostrarLogsAuditoria ? (
        <ScannerLogPanel
          logs={auditLogs}
          onLimpar={limparAuditLogs}
          onOcultar={() => setMostrarLogsAuditoria(false)}
        />
      ) : (
        <button
          type="button"
          className="scanner-log-reveal-tab"
          onClick={() => setMostrarLogsAuditoria(true)}
          title="Mostrar logs ao vivo (WebSocket / YOLO / API)"
        >
          <i className="ph ph-sidebar-simple text-lg" aria-hidden></i>
          <span>Logs</span>
        </button>
      )}

      <FinalizacaoConciliacaoModal
        aberto={modalConciliacaoAberto}
        carregando={modalCarregando}
        erro={modalErro}
        relatorio={relatorioConciliacao}
        salvando={modalSalvando}
        onFechar={() => !modalSalvando && setModalConciliacaoAberto(false)}
        onConfirmarManual={handleConfirmarManual}
        onConfirmarCapturas={handleConfirmarCapturas}
      />

      {revisaoModalDeteccaoId != null && itemEmRevisao ? (
        <RevisaoDeteccaoModal
          key={revisaoModalDeteccaoId}
          item={itemEmRevisao}
          alimentos={alimentosCatalogo}
          salvando={salvandoRevisao}
          onFechar={() => !salvandoRevisao && setRevisaoModalDeteccaoId(null)}
          onSalvar={handleSalvarRevisao}
        />
      ) : null}
    </div>
  );
}
