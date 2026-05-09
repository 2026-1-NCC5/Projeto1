/**
 * Itens do Kanban no appState: linhas agregadas por alimento + peso unitário.
 * `quantity` = número de unidades iguais; peso na linha = peso por unidade.
 */

function roundWeight(w) {
  const n = Number(w);
  if (!Number.isFinite(n)) return 0;
  return Math.round(n * 1000) / 1000;
}

function itemKey(it) {
  const aid = it.alimentoId != null ? String(it.alimentoId) : '';
  const ga = it.grupoAlimentoId != null ? String(it.grupoAlimentoId) : '';
  const name = String(it.name || '').trim().toLowerCase();
  const w = roundWeight(it.weight);
  return `${aid}|${ga}|${name}|${w}`;
}

/**
 * Normaliza array heterogêneo (legado só name/weight ou novo shape) para lista agregada.
 */
export function aggregateKanbanItems(rawItems) {
  if (!Array.isArray(rawItems) || rawItems.length === 0) return [];
  const map = new Map();
  for (const it of rawItems) {
    if (!it) continue;
    const weight = roundWeight(it.weight);
    const quantity = Math.max(1, parseInt(it.quantity, 10) || 1);
    const base = {
      name: String(it.name || '').trim() || '—',
      weight,
      quantity,
      alimentoId: it.alimentoId != null ? it.alimentoId : undefined,
      grupoAlimentoId: it.grupoAlimentoId != null ? it.grupoAlimentoId : undefined,
      deteccaoIds: Array.isArray(it.deteccaoIds) ? [...it.deteccaoIds] : [],
    };
    const k = itemKey(base);
    if (!map.has(k)) {
      map.set(k, { ...base });
    } else {
      const cur = map.get(k);
      cur.quantity += quantity;
      if (base.deteccaoIds.length) {
        cur.deteccaoIds = [...(cur.deteccaoIds || []), ...base.deteccaoIds];
      }
    }
  }
  return Array.from(map.values());
}

export function totalKgFromItems(items) {
  if (!Array.isArray(items)) return 0;
  return items.reduce((acc, it) => acc + roundWeight(it.weight) * (it.quantity || 1), 0);
}

/**
 * Monta linhas agregadas a partir do relatório de conciliação (lado declarado).
 */
export function buildAggregatedItemsFromRelatorio(relatorio) {
  if (!relatorio?.linhas?.length) return [];
  const flat = [];
  relatorio.linhas.forEach((linha) => {
    const qtd = Math.max(0, parseInt(linha.qtd_declarada, 10) || 0);
    if (qtd <= 0) return;
    const pesoTotal = Number(linha.peso_declarado_kg || 0);
    const pesoUnitario = qtd > 0 ? roundWeight(pesoTotal / qtd) : pesoTotal;
    flat.push({
      name: linha.alimento_nome,
      weight: pesoUnitario,
      quantity: qtd,
    });
  });
  return aggregateKanbanItems(flat);
}

/**
 * Monta linhas agregadas a partir dos itens da sessão de câmera (scoreboard).
 */
export function buildAggregatedItemsFromCapturasSession(sessionItems) {
  if (!Array.isArray(sessionItems) || sessionItems.length === 0) return [];
  const flat = sessionItems.map((it) => ({
    name: it.name,
    weight: it.weight,
    quantity: 1,
    alimentoId: it.alimento_id ?? it.alimentoId,
    grupoAlimentoId: it.grupoAlimentoId,
    deteccaoIds: it.deteccaoId != null ? [it.deteccaoId] : [],
  }));
  return aggregateKanbanItems(flat);
}

/**
 * Une itens existentes com novos (modo acrescentar na inserção manual).
 */
export function mergeKanbanItemsAppend(existing, novos) {
  const combined = [];
  if (Array.isArray(existing)) {
    for (const it of existing) {
      combined.push({
        ...it,
        quantity: Math.max(1, it.quantity || 1),
      });
    }
  }
  if (Array.isArray(novos)) {
    for (const it of novos) {
      combined.push({
        name: it.name,
        weight: it.weight,
        quantity: Math.max(1, it.quantity || 1),
        alimentoId: it.alimentoId,
        grupoAlimentoId: it.grupoAlimentoId,
      });
    }
  }
  return aggregateKanbanItems(combined);
}
