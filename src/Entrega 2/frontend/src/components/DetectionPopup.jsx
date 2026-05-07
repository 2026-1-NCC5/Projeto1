import React, { forwardRef } from 'react';

// Popup arrastável que aparece sobre o feed da câmera. Mostra o resultado
// da última detecção e a lista de itens já confirmados na sessão atual.
const DetectionPopup = forwardRef(function DetectionPopup({
  capturando,
  leituraNome,
  leituraDesc,
  previewLabel,
  btnLeituraText,
  currentSessionItems,
  destaque,
  onPopupDown,
  onSimular,
  onReject,
  onRemoverItem,
  onFinalizar,
}, popupRef) {
  return (
    <div ref={popupRef}
      onMouseDown={onPopupDown} onTouchStart={onPopupDown}
      className={`absolute bg-black-80 backdrop-blur rounded-2xl p-5 shadow-2xl border border-gray-medium w-full max-w-sm flex flex-col max-h-[85vh] z-50 cursor-pointer ${destaque ? 'detection-popup-flash' : ''}`}
      style={{ bottom: '40px', left: '50%', transform: 'translateX(-50%)', touchAction: 'none' }}>

      <div className="flex justify-between align-start mb-1 pointer-events-none">
        <div>
          <span className="text-primary text-[10px] font-bold tracking-widest uppercase flex align-center gap-1">
            <i className="ph-fill ph-scan"></i> {capturando ? 'DETECTANDO ALIMENTO' : 'CAPTURA PAUSADA'}
          </span>
          <h2 className="text-white font-bold text-xl mt-1 tracking-tight">{leituraNome}</h2>
          <p className="text-gray text-xs mt-1">
            {previewLabel ? `Preview YOLO: ${previewLabel}` : leituraDesc}
          </p>
        </div>
        <button className="btn-icon circle-bg-gray border border-gray-medium transition shadow-sm pointer-events-auto hover:bg-red-500"
          onClick={onReject} style={{ width: '34px', height: '34px' }}>
          <i className="ph ph-x text-white"></i>
        </button>
      </div>

      <button className="btn btn-outline w-full flex align-center justify-center gap-2 transition font-bold py-2 rounded-xl mb-4 mt-4 text-white border-gray-medium hover-bg-gray-light"
        onClick={onSimular}>
        <i className="ph ph-plus-circle text-lg"></i> <span>{btnLeituraText}</span>
      </button>

      <div id="session-items-list" className="flex flex-col gap-2 mb-4 overflow-y-auto pointer-events-auto" style={{ maxHeight: '120px' }} onMouseDown={e => e.stopPropagation()} onTouchStart={e => e.stopPropagation()}>
        {currentSessionItems.length === 0 ? (
          <div className="text-center text-gray text-xs py-2 opacity-50 font-medium">Nenhum item adicionado nesta contagem.</div>
        ) : (
          currentSessionItems.map((item, index) => (
            <div key={index} className="flex justify-between align-center p-2 rounded text-sm text-white mb-2" style={{ background: 'rgba(0,0,0,0.3)' }}>
              <span className="font-medium truncate flex-grow">{item.name}</span>
              <div className="flex align-center gap-2 flex-shrink-0 ml-2">
                <span className="text-primary font-bold">+{item.weight}kg</span>
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
});

export default DetectionPopup;
