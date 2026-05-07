import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useAppState } from '../context/appStateContextValue';
import useAuditoriaWS from '../hooks/useAuditoriaWS';
import useCameraStream from '../hooks/useCameraStream';
import useDraggablePopup from '../hooks/useDraggablePopup';
import { criarDeteccao, finalizarSessao } from '../services/api';
import { WS_BASE } from '../constants';
import DetectionPopup from '../components/DetectionPopup';
import ScannerLogPanel from '../components/ScannerLogPanel';
import DetectionConfirmedOverlay from '../components/DetectionConfirmedOverlay';

// Tempo de cooldown (em ms) entre confirmar um alimento e retomar a captura
// automaticamente. O overlay mostra a barra de progresso correspondente.
const COOLDOWN_PROXIMA_CAPTURA_MS = 3000;

// Tela do scanner: liga câmera, abre WS de auditoria, exibe overlay de
// bbox/preview YOLO, popup de detecção arrastável e painel de logs.
export default function CameraScreen() {
  const {
    activeGroupId,
    auditSessaoId, setAuditSessaoId, auditSessaoIdNumero,
    addToast, setCurrentScreen, setAppState,
  } = useAppState();

  const videoRef = useRef(null);
  const popupRef = useRef(null);

  const [currentSessionItems, setCurrentSessionItems] = useState([]);
  const [usarGemini, setUsarGemini] = useState(true);
  const [leituraNome, setLeituraNome] = useState('Consultando Alimento...');
  const [leituraDesc, setLeituraDesc] = useState('Clique em iniciar captura para enviar frames ao modelo.');
  const [btnLeituraText, setBtnLeituraText] = useState('AGUARDANDO DETECÇÃO');
  const [currFood, setCurrFood] = useState('');
  const [currWeight, setCurrWeight] = useState(0);
  // Quando != null, mostra o overlay de confirmação animado por 3s e
  // pausa a captura. Após o cooldown, a captura é retomada automaticamente.
  const [confirmacao, setConfirmacao] = useState(null);
  // Flag visual: aplica um "flash" no popup principal quando uma detecção
  // nova chega via WS, para chamar atenção do operador.
  const [destaquePopup, setDestaquePopup] = useState(false);

  // ===== AUDITORIA WS (câmera ↔ FastAPI) =====
  // O hook só conecta quando a tela é 'camera' e há um sessao_id válido.
  const {
    status: auditStatus,
    ultimaDeteccao,
    ultimoPreview,
    logs: auditLogs,
    capturando,
    iniciarCaptura: iniciarCapturaWS,
    pararCaptura: pararCapturaWS,
    reset: resetAuditWS,
    limparLogs: limparAuditLogs,
    registrarLog: registrarAuditLog,
  } = useAuditoriaWS({
    sessaoId: auditSessaoId,
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

  // Quando uma detecção chega, popula o popup existente da câmera.
  // O estado é mutável depois (o usuário pode rejeitar/limpar), então não dá
  // para derivar via useMemo puro — sincronizamos via efeito controlado.
  useEffect(() => {
    if (!ultimaDeteccao) return;
    const rf = ultimaDeteccao.resultado_final || {};
    const yolo = ultimaDeteccao.yolo || {};
    const gemini = ultimaDeteccao.gemini || null;
    const peso = rf.peso_padrao_kg || 0;

    const partes = [];
    if (yolo.confianca != null) partes.push(`YOLO ${Math.round((yolo.confianca || 0) * 100)}%`);
    if (gemini) {
      partes.push(gemini.concorda ? 'Gemini concorda' : 'Gemini discordou');
    }

    /* eslint-disable react-hooks/set-state-in-effect */
    setCurrFood(rf.alimento_nome || yolo.classe || 'Desconhecido');
    setCurrWeight(peso);
    setLeituraNome(rf.alimento_nome || yolo.classe || 'Desconhecido');
    setLeituraDesc(`Peso ${peso}kg · ${partes.join(' · ')}`);
    setBtnLeituraText(`ADICIONAR +${peso}KG`);
    setDestaquePopup(true);
    /* eslint-enable react-hooks/set-state-in-effect */
  }, [ultimaDeteccao]);

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

  const { handlePopupDown } = useDraggablePopup({ ativo: true, popupRef });

  const handleToggleCapture = () => {
    if (capturando) {
      pararCapturaWS();
      return;
    }
    iniciarCapturaWS();
  };

  const handleSimularClick = async () => {
    if (!currFood || currWeight <= 0) return;

    // Pausa a captura enquanto mostramos o overlay de confirmação. Evita que
    // novos frames disparem detecções enquanto o operador absorve o feedback.
    pararCapturaWS();

    // Adiciona localmente para a UI atualizar imediatamente
    const novoItem = { name: currFood, weight: currWeight };
    const novosItens = [...currentSessionItems, novoItem];
    setCurrentSessionItems(novosItens);

    // Snapshot dos dados para o overlay (currFood/currWeight são limpos abaixo).
    const snapshot = {
      alimento: currFood,
      peso: currWeight,
      categoria: ultimaDeteccao?.resultado_final?.categoria
        || ultimaDeteccao?.yolo?.categoria
        || null,
      geminiConcorda: ultimaDeteccao?.gemini ? !!ultimaDeteccao.gemini.concorda : null,
      totalCapturados: novosItens.length,
      pesoTotalSessao: novosItens.reduce((acc, it) => acc + (it.weight || 0), 0),
    };
    setConfirmacao(snapshot);

    // Persiste no backend quando há uma detecção real (vinda do WS).
    if (ultimaDeteccao && ultimaDeteccao.resultado_final && ultimaDeteccao.resultado_final.alimento_id) {
      try {
        const r = await criarDeteccao({
          sessaoIdNumero: auditSessaoIdNumero,
          deteccao: ultimaDeteccao,
          currWeight,
          currFood,
        });
        if (!r.ok) {
          addToast(`Falha ao registrar (${r.status})`, 'error');
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
    resetAuditWS();
  };

  // Fecha o overlay de confirmação e retoma a captura. Chamado tanto pelo
  // término da progress bar quanto pelo botão "Pular".
  const concluirCooldown = () => {
    setConfirmacao(null);
    iniciarCapturaWS();
  };

  const handleRejectClick = () => {
    setLeituraNome('Descartado. Buscando...');
    setLeituraDesc('--');
    setBtnLeituraText('AGUARDANDO DETECÇÃO');
    resetAuditWS();
  };

  const handleFinishCount = async () => {
    if (activeGroupId && currentSessionItems.length > 0) {
      setAppState(prev => prev.map(group => {
        if (group.id === activeGroupId) {
          const newItems = [...group.items, ...currentSessionItems];
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
    try {
      const r = await finalizarSessao(auditSessaoIdNumero);
      if (!r.ok) {
        addToast(`Falha ao finalizar sessão (${r.status})`, 'error');
      } else {
        addToast('Sessão finalizada no backend.', 'success');
      }
    } catch {
      addToast('Backend indisponível — sessão não finalizada.', 'warning');
    }
    pararCapturaWS();
    setCurrentScreen('dashboard');
  };

  const handleSair = () => {
    pararCapturaWS();
    setCurrentScreen('dashboard');
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
        <div className="text-xs font-bold text-white bg-black/60 backdrop-blur px-3 py-1.5 rounded-full border border-gray-medium shadow flex align-center gap-2">
          <div className="pulse-dot"></div>
          GRUPO {activeGroupId} · SESSÃO
          <input
            type="number"
            min="1"
            value={auditSessaoId}
            onChange={(e) => {
              const next = e.target.value;
              if (/^\d*$/.test(next)) setAuditSessaoId(next);
            }}
            onBlur={() => {
              if (!auditSessaoId || Number(auditSessaoId) < 1) setAuditSessaoId('1');
            }}
            style={{ width: '60px', background: 'transparent', color: 'white', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '4px', padding: '0 4px' }}
            title="ID da sessão criada via /api/v1/sessoes (Swagger)"
          />
          · {auditStatus.toUpperCase()}
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
      </div>

      <div className="flex-grow flex align-center justify-center relative w-full h-full pointer-events-none z-10">
        <div className="scan-frame relative pointer-events-none">
          <div className="corner top-left"></div>
          <div className="corner top-right"></div>
          <div className="corner bottom-left"></div>
          <div className="corner bottom-right"></div>
          <div className="scan-line"></div>
        </div>
      </div>

      <DetectionPopup
        ref={popupRef}
        capturando={capturando}
        leituraNome={leituraNome}
        leituraDesc={leituraDesc}
        previewLabel={previewLabel}
        btnLeituraText={btnLeituraText}
        currentSessionItems={currentSessionItems}
        destaque={destaquePopup}
        onPopupDown={handlePopupDown}
        onSimular={handleSimularClick}
        onReject={handleRejectClick}
        onRemoverItem={(index) => setCurrentSessionItems(prev => prev.filter((_, i) => i !== index))}
        onFinalizar={handleFinishCount}
      />

      <ScannerLogPanel logs={auditLogs} onLimpar={limparAuditLogs} />

      <DetectionConfirmedOverlay
        visivel={!!confirmacao}
        alimento={confirmacao?.alimento}
        peso={confirmacao?.peso}
        categoria={confirmacao?.categoria}
        geminiConcorda={confirmacao?.geminiConcorda}
        totalCapturados={confirmacao?.totalCapturados ?? currentSessionItems.length}
        pesoTotalSessao={confirmacao?.pesoTotalSessao ?? pesoTotalSessao}
        duracaoMs={COOLDOWN_PROXIMA_CAPTURA_MS}
        onConcluir={concluirCooldown}
        onPular={concluirCooldown}
      />
    </div>
  );
}
