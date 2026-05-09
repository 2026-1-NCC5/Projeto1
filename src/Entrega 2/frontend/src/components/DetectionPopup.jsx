import React from 'react';
import SessionItemStatusChip from './SessionItemStatusChip';

// Popup fixo na base do feed da câmera (bottom + centralizado). Mostra o resultado
// da última detecção e a lista de itens já confirmados na sessão atual.
export default function DetectionPopup({
  capturando,
  leituraNome,
  leituraDesc,
  previewLabel,
  btnLeituraText,
  currentSessionItems,
  destaque,
  minimizado,
  onSimular,
  onToggleMinimizado,
  onRemoverItem,
  onFinalizar,
  onRevisarItem,
}) {
  const popupPositionStyle = { bottom: '40px', left: '50%', transform: 'translateX(-50%)' };

  if (minimizado) {
    return (
      <div
        className={`detection-popup-minimized absolute bg-black-80 backdrop-blur rounded-2xl shadow-2xl border border-gray-medium w-full max-w-sm flex flex-row align-center justify-between gap-3 z-50 px-4 py-3 ${destaque ? 'detection-popup-flash' : ''}`}
        style={popupPositionStyle}>

        <div className="flex align-center gap-2 min-w-0 pointer-events-none flex-grow">
          <span className={`text-[10px] font-bold tracking-widest uppercase flex align-center gap-1 flex-shrink-0 ${capturando ? 'text-primary' : 'text-red-500'}`}>
            <i className="ph-fill ph-scan"></i>
            {capturando ? 'DETECTANDO' : 'PAUSADO'}
          </span>
          <span className="text-white font-bold text-sm truncate">{leituraNome}</span>
          <span className="text-primary font-bold text-xs flex-shrink-0">
            {currentSessionItems.length} {currentSessionItems.length === 1 ? 'item' : 'itens'}
          </span>
        </div>
        <button type="button"
          title="Expandir painel de contagem"
          aria-label="Expandir painel de contagem"
          className="btn-icon circle-bg-gray border border-gray-medium transition shadow-sm relative z-10 hover:bg-white/15 flex-shrink-0"
          style={{ width: '34px', height: '34px', touchAction: 'manipulation' }}
          onClick={onToggleMinimizado}>
          <i className="ph ph-caret-up text-white"></i>
        </button>
      </div>
    );
  }

  return (
    <div
      className={`absolute bg-black-80 backdrop-blur rounded-2xl p-5 shadow-2xl border border-gray-medium w-full max-w-sm flex flex-col max-h-[85vh] z-50 ${destaque ? 'detection-popup-flash' : ''}`}
      style={popupPositionStyle}>

      <div className="flex justify-between align-start gap-3 mb-1">
        <div className="pointer-events-none min-w-0 flex-1 pr-1">
          <span className="text-primary text-[10px] font-bold tracking-widest uppercase flex align-center gap-1">
            <i className="ph-fill ph-scan"></i> {capturando ? 'DETECTANDO ALIMENTO' : 'CAPTURA PAUSADA'}
          </span>
          <h2 className="text-white font-bold text-xl mt-1 tracking-tight">{leituraNome}</h2>
          <p className="text-gray text-xs mt-1">
            {previewLabel ? `Preview YOLO: ${previewLabel}` : leituraDesc}
          </p>
        </div>
        <button type="button"
          title="Minimizar painel de contagem"
          aria-label="Minimizar painel de contagem"
          className="btn-icon circle-bg-gray border border-gray-medium transition shadow-sm relative z-10 flex-shrink-0 hover:bg-white/15"
          style={{ width: '34px', height: '34px', touchAction: 'manipulation' }}
          onClick={onToggleMinimizado}>
          <i className="ph ph-caret-down text-white"></i>
        </button>
      </div>

      <button className="btn btn-outline w-full flex align-center justify-center gap-2 transition font-bold py-2 rounded-xl mb-4 mt-4 text-white border-gray-medium hover-bg-gray-light"
        onClick={onSimular}>
        <i className="ph ph-plus-circle text-lg"></i> <span>{btnLeituraText}</span>
      </button>

      <div id="session-items-list" className="flex flex-col gap-2 mb-4 overflow-y-auto pointer-events-auto" style={{ maxHeight: '120px' }}>
        {currentSessionItems.length === 0 ? (
          <div className="text-center text-gray text-xs py-2 opacity-50 font-medium">Nenhum item adicionado nesta contagem.</div>
        ) : (
          currentSessionItems.map((item, index) => (
            <div key={`${item.deteccaoId ?? 'manual'}-${index}`} className="flex justify-between align-center p-2 rounded text-sm text-white mb-2 detection-popup-item" style={{ background: 'rgba(0,0,0,0.3)' }}>
              <div className="flex align-center gap-2 flex-grow min-w-0">
                <span className="font-medium truncate">{item.name}</span>
                {item.status && (
                  <SessionItemStatusChip
                    status={item.status}
                    nomeOriginal={item.nomeOriginal}
                    nomeAtual={item.name}
                    justificativaGemini={item.gemini?.justificativa}
                  />
                )}
              </div>
              <div className="flex align-center gap-2 flex-shrink-0 ml-2">
                <span className="text-primary font-bold">+{item.weight}kg</span>
                {onRevisarItem && item.status === 'revisao_pendente' && item.deteccaoId != null ? (
                  <button
                    type="button"
                    className="session-items-row-revisar"
                    title="Revisar peso e categoria"
                    aria-label="Revisar item"
                    onClick={() => onRevisarItem(index)}
                  >
                    <i className="ph ph-pencil-simple" />
                  </button>
                ) : null}
                <button className="btn-remove-session-item text-gray transition btn-icon hover:text-red-500" style={{ width: '24px', height: '24px' }}
                  onClick={() => onRemoverItem(index)}>
                  <i className="ph ph-trash"></i>
                </button>
              </div>
            </div>
          ))
        )}
      </div>

      <button className="btn btn-primary w-full flex align-center justify-center gap-2 transition font-bold py-3 rounded-xl shadow-red mt-auto" onClick={onFinalizar}>
        <i className="ph ph-check-circle text-lg"></i> <span>FINALIZAR CONTAGEM ({currentSessionItems.length})</span>
      </button>
    </div>
  );
}
