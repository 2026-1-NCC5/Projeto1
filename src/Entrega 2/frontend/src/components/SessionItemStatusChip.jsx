import React from 'react';

// Chip discreto que mostra o estado da validação Gemini de uma detecção:
//  - validando: spinner + "validando" (Gemini ainda processando em background)
//  - validado:  check verde + "Gemini ok"
//  - corrigido: warn amarelo + tooltip "Gemini corrigiu X→Y"
//  - sem_gemini: ícone neutro + "sem Gemini"
//
// Usado tanto no SessionItemsPanel (scoreboard à esquerda) quanto na lista
// interna do DetectionPopup. O componente é defensivo: se `status` não for
// reconhecido, não renderiza nada.
export default function SessionItemStatusChip({
  status,
  nomeOriginal,
  nomeAtual,
  justificativaGemini,
}) {
  if (!status) return null;

  if (status === 'validando') {
    return (
      <span
        className="session-item-chip session-item-chip--validando"
        title="Aguardando confirmação do Gemini"
      >
        <span className="session-item-chip__spinner" />
        validando
      </span>
    );
  }

  if (status === 'validado') {
    return (
      <span
        className="session-item-chip session-item-chip--validado"
        title="Gemini concorda com a classificação YOLO"
      >
        <i className="ph-fill ph-check"></i>
        Gemini ok
      </span>
    );
  }

  if (status === 'corrigido') {
    const tooltip = nomeOriginal && nomeOriginal !== nomeAtual
      ? `Gemini corrigiu: ${nomeOriginal} → ${nomeAtual}`
      : 'Gemini corrigiu a classificação';
    return (
      <span
        className="session-item-chip session-item-chip--corrigido"
        title={tooltip}
      >
        <i className="ph-fill ph-warning"></i>
        corrigido
      </span>
    );
  }

  if (status === 'sem_gemini') {
    return (
      <span
        className="session-item-chip session-item-chip--sem-gemini"
        title="Gemini desligado ou indisponível"
      >
        <i className="ph ph-sparkle"></i>
        sem Gemini
      </span>
    );
  }

  if (status === 'revisao_pendente') {
    const tip = justificativaGemini && String(justificativaGemini).trim()
      ? String(justificativaGemini).trim()
      : 'Peso ilegível ou incerto na imagem — toque no lápis para revisar';
    return (
      <span
        className="session-item-chip session-item-chip--revisao-pendente"
        title={tip}
      >
        <i className="ph-fill ph-warning-octagon"></i>
        revisar
      </span>
    );
  }

  return null;
}
