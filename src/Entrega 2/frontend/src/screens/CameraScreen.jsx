import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useAppState } from '../context/appStateContextValue';
import useAuditoriaWS from '../hooks/useAuditoriaWS';
import useCameraStream from '../hooks/useCameraStream';
import {
  criarDeteccaoManual,
  finalizarSessao,
  obterOuCriarSessaoAtiva,
  obterConciliacaoPreviaSessao,
  decidirFonteFinalSessao,
} from '../services/api';
import { WS_BASE } from '../constants';
import DetectionPopup from '../components/DetectionPopup';
import ScannerLogPanel from '../components/ScannerLogPanel';
import SessionItemsPanel from '../components/SessionItemsPanel';
import FinalizacaoConciliacaoModal from '../components/FinalizacaoConciliacaoModal';

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

  useEffect(() => {
    if (!activeGroupId || grupoBackendId == null) return;
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
  }, [activeGroupId, grupoBackendId, addToast, setAuditSessaoId]);

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
  }, [ultimaPreliminar]);

  // 2) Quando o Gemini conclui (ou o backend sinaliza que está OFF),
  //    procuramos o item correspondente pelo `deteccaoId` e atualizamos o
  //    chip silenciosamente. Se o Gemini propôs uma classe diferente,
  //    renomeamos o item e marcamos como "corrigido".
  useEffect(() => {
    if (!ultimaAtualizacaoGemini) return;
    const { deteccao_id, gemini, alimento_nome, alimento_id, fonte } = ultimaAtualizacaoGemini;
    setCurrentSessionItems(prev => {
      const idx = prev.findIndex(it => it.deteccaoId === deteccao_id);
      if (idx < 0) return prev;
      const atual = prev[idx];
      let novoStatus = atual.status;
      let novoNome = atual.name;
      let novoAlimentoId = atual.alimento_id;
      if (!gemini) {
        novoStatus = 'sem_gemini';
      } else if (gemini.concorda) {
        novoStatus = 'validado';
      } else if (alimento_nome) {
        novoStatus = 'corrigido';
        novoNome = alimento_nome;
        novoAlimentoId = alimento_id ?? atual.alimento_id;
      } else {
        // Gemini discordou mas não propôs classe nova — mantém YOLO
        novoStatus = 'sem_gemini';
      }
      const atualizado = {
        ...atual,
        status: novoStatus,
        name: novoNome,
        alimento_id: novoAlimentoId,
        gemini,
        fonte: fonte || atual.fonte,
      };
      const cpy = prev.slice();
      cpy[idx] = atualizado;
      return cpy;
    });
  }, [ultimaAtualizacaoGemini]);

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
      const itensManuais = [];
      relatorio.linhas.forEach((linha) => {
        if (!linha.qtd_declarada || linha.qtd_declarada <= 0) return;
        const qtd = linha.qtd_declarada;
        const pesoTotal = Number(linha.peso_declarado_kg || 0);
        const pesoUnitario = qtd > 0 ? Number((pesoTotal / qtd).toFixed(2)) : pesoTotal;
        for (let i = 0; i < qtd; i += 1) {
          itensManuais.push({ name: linha.alimento_nome, weight: pesoUnitario });
        }
      });
      if (itensManuais.length === 0) return;
      setAppState(prev => prev.map(group => {
        if (group.id === activeGroupId) {
          const newItems = [...group.items, ...itensManuais];
          const addedKg = itensManuais.reduce((acc, obj) => acc + obj.weight, 0);
          return {
            ...group,
            items: newItems,
            totalKg: parseFloat((group.totalKg + addedKg).toFixed(2))
          };
        }
        return group;
      }));
      return;
    }

    if (currentSessionItems.length > 0) {
      setAppState(prev => prev.map(group => {
        if (group.id === activeGroupId) {
          // Sanitiza itens para o shape do appState ({ name, weight }) — campos
          // de tracking (deteccaoId, status, gemini) só importam na tela do scanner.
          const novosSimples = currentSessionItems.map(it => ({ name: it.name, weight: it.weight }));
          const newItems = [...group.items, ...novosSimples];
          const addedKg = currentSessionItems.reduce((acc, obj) => acc + obj.weight, 0);
          return {
            ...group,
            items: newItems,
            totalKg: parseFloat((group.totalKg + addedKg).toFixed(2))
          };
        }
        return group;
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

  const handleRemoverItem = (index) => {
    setCurrentSessionItems(prev => prev.filter((_, i) => i !== index));
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
      />

      <SessionItemsPanel
        items={currentSessionItems}
        pesoTotal={pesoTotalSessao}
        onRemoverItem={handleRemoverItem}
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
    </div>
  );
}
