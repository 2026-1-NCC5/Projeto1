import React, { useState } from 'react';
import { API_BASE } from '../constants';

export default function RevisaoDeteccaoModal({
  item,
  alimentos = [],
  salvando,
  onFechar,
  onSalvar,
}) {
  const [alimentoId, setAlimentoId] = useState(() =>
    item.alimento_id != null ? String(item.alimento_id) : ''
  );
  const [pesoStr, setPesoStr] = useState(() =>
    item.weight != null && item.weight > 0 ? String(item.weight).replace('.', ',') : ''
  );

  const imagemUrl = item.imagem_path ? `${API_BASE}/${item.imagem_path}` : null;
  const hintGemini = item.gemini?.justificativa || '';

  const handleSubmit = (e) => {
    e.preventDefault();
    const peso = parseFloat(String(pesoStr).replace(',', '.'));
    const idAl = alimentoId ? parseInt(alimentoId, 10) : NaN;
    if (!Number.isFinite(peso) || peso <= 0) return;
    if (!Number.isFinite(idAl) || idAl <= 0) return;
    onSalvar?.({ alimentoId: idAl, pesoKg: peso });
  };

  return (
    <div className="modal-overlay active">
      <div className="modal-content revisao-deteccao-modal">
        <div className="revisao-deteccao-header">
          <div>
            <span className="revisao-deteccao-eyebrow">Revisão obrigatória</span>
            <h3 className="revisao-deteccao-title">Confirmar peso e categoria</h3>
            {hintGemini ? (
              <p className="revisao-deteccao-hint text-gray text-xs mt-1">{hintGemini}</p>
            ) : null}
          </div>
          <button type="button" className="btn-icon" onClick={onFechar} disabled={salvando}>
            <i className="ph ph-x" />
          </button>
        </div>

        <div className="revisao-deteccao-body">
          <div className="revisao-deteccao-media">
            {imagemUrl ? (
              <img
                className="revisao-deteccao-img"
                src={imagemUrl}
                alt="Evidência enviada ao Gemini"
              />
            ) : (
              <div className="revisao-deteccao-sem-imagem">
                <i className="ph ph-image-broken text-2xl" />
                <span>Evidência desta captura não disponível no servidor.</span>
              </div>
            )}
          </div>

          <form className="revisao-deteccao-form" onSubmit={handleSubmit}>
            <label className="revisao-deteccao-field">
              <span>Alimento (categoria)</span>
              <select
                value={alimentoId}
                onChange={(e) => setAlimentoId(e.target.value)}
                required
                disabled={salvando}
              >
                <option value="">Selecione...</option>
                {alimentos.map((a) => (
                  <option key={a.id} value={a.id}>
                    {a.nome}
                  </option>
                ))}
              </select>
            </label>
            <label className="revisao-deteccao-field">
              <span>Peso (kg)</span>
              <input
                type="text"
                inputMode="decimal"
                value={pesoStr}
                onChange={(e) => setPesoStr(e.target.value)}
                placeholder="ex.: 1,5"
                required
                disabled={salvando}
              />
            </label>
            <div className="revisao-deteccao-actions">
              <button type="button" className="btn btn-outline" onClick={onFechar} disabled={salvando}>
                Cancelar
              </button>
              <button type="submit" className="btn btn-primary" disabled={salvando}>
                {salvando ? 'Salvando…' : 'Salvar correção'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
