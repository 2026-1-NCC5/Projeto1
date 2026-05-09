# CLAUDE.md — AbraceAI

> **Fonte canônica deste repositório: [`AGENTS.md`](./AGENTS.md).**
> Leia-o **inteiro** antes da primeira ação. Este arquivo só adiciona overrides e
> instruções específicas para o Claude (Cursor, Claude Code, Anthropic API).

---

## 0. O que ler primeiro

1. [`AGENTS.md`](./AGENTS.md) — visão geral, estrutura, stack, comandos, convenções,
   pitfalls, glossário e Definition of Done.
2. [`Documentos/AbraceAI_Concepcao_Produto.md`](./Documentos/AbraceAI_Concepcao_Produto.md) —
   especificação completa do produto (RFs, RNFs, modelo de dados, arquitetura).
3. O `TODO-*.md` da área que você for tocar (em [`Documentos/ToDo/`](./Documentos/ToDo)).

Se houver conflito entre este arquivo e `AGENTS.md`, **siga `AGENTS.md`**, exceto
nas seções abaixo (que se aplicam apenas ao Claude).

---

## 1. Postura e tom

- Respostas em **português do Brasil**, curtas e diretas, alinhadas ao idioma do código.
- Sem emojis em código, comentários ou commits, salvo solicitação explícita.
- Não anuncie planos genéricos (“vou ler o arquivo X, depois editar Y…”): execute,
  agrupe ferramentas em paralelo quando independentes, e relate ao final.
- Não cite o nome de ferramentas internas ao usuário; descreva a ação em linguagem natural.

## 2. Uso de ferramentas

- **Leitura/edição de arquivos:** use `Read`, `Glob`, `Grep`, `StrReplace`, `Write` —
  **nunca** `cat`/`sed`/`awk`/heredoc para manipular arquivos.
- **Buscas:** prefira `Grep` para strings/símbolos exatos; `SemanticSearch` apenas
  para perguntas conceituais amplas.
- **Caminhos com espaço:** `src/Entrega 2/...` exige aspas em qualquer comando shell.
- **Tarefas em paralelo independentes:** dispare múltiplas chamadas no mesmo turno.
- **Long-running:** use `block_until_ms: 0` para servidores (Vite, Uvicorn, Docker
  Compose). Faça apenas um smoke check e siga trabalhando.
- **TodoWrite:** use proativamente para tarefas com 3+ passos não-triviais ou múltiplos
  arquivos. Atualize status conforme avança.

## 3. Edições

- **Não remova comentários existentes** (regra do mantenedor — vale para todo o
  repositório).
- **Não adicione comentários óbvios** descrevendo o que o código faz; explique apenas
  intenção, trade-off ou restrição não óbvia. Nunca use comentários para “narrar”
  uma alteração.
- **Não crie `*.md` proativamente.** Edite arquivos existentes. Crie docs somente
  quando o usuário pedir.
- **Não introduza dependências sem necessidade.** Se for inevitável, use o gerenciador
  (`pip install <pkg>`, `npm i <pkg>`) — não chute versões.
- **Não troque o stack** (Tailwind, Shadcn, TS, libs novas) sem alinhamento explícito,
  mesmo que o documento de concepção liste a tecnologia como alvo futuro.
- **Preserve PT-BR** em identificadores de domínio (`grupo`, `sessao`, `deteccao`,
  `alimento`) e mensagens visíveis.

## 4. Domínio — atalhos mentais

- 8 tabelas: `usuarios`, `grupos`, `alunos`, `grupos_alimentos`, `alimentos`,
  `sessoes`, `itens_declarados`, `deteccoes`.
- PoC atual não usa Auth/JWT/senha: `usuarios` é cadastro simples; endpoints REST e
  WebSocket estão liberados para teste local. Não reintroduza `security.py`,
  `/auth`, `token.py`, `senha_hash` ou `Depends(get_current_active_user)` sem pedido explícito.
- `deteccoes.alimento_id_original` fixa o alimento no **INSERT** preliminar (YOLO).
  O Gemini pode corrigir `alimento_id`/`fonte` em `UPDATE`; preserve `alimento_id_original`
  em migrações (métricas e auditoria).
- Anti-duplicidade = **tracking + zona + cooldown** (3 mecanismos combinados,
  RF04). Mexer em um sem revisar os outros é red flag.
- Scanner atual usa WebSocket nativo em `ws://localhost:8000/ws/auditoria/{sessao_id}`.
  A captura começa por botão, tem toggle Gemini ON/OFF, bbox/label YOLO sobre o vídeo ao
  vivo e painel de logs. **Durante uma detecção bem-sucedida o envio de frames não pausa**
  (Gemini corre em thread + `asyncio.create_task`). Não reintroduza Socket.IO nem mock
  hardcoded de alimentos/pesos na tela de câmera.
- O scanner é **auto-registro assíncrono**: no gatilho de estabilidade o backend faz
  INSERT em `deteccoes`, envia `deteccao_preliminar` (`deteccao_id`), dispara Gemini em
  background e só então envia `deteccao_atualizada`. `useAuditoriaWS` expõe `ultimaPreliminar`
  e `ultimaAtualizacaoGemini` (não use mais `ultimaDeteccao`/`analisandoTs`/`DetectionConfirmedOverlay`).
  `CameraScreen` adiciona o item ao scoreboard ao preliminar (chip **validando** + toast opcional),
  atualiza o chip quando a atualização chega (**sem** toast em correção). **Nunca** chame
  `criarDeteccao()` nesse ciclo automático — duplica linha no banco. Fallback manual via
  `criarDeteccaoManual()` + campos digitados. Log *“Sem alimento mapeado para a classe”* =
  classe YOLO sem registro correspondente em `alimentos.classe_yolo` (ajustar seeds/mapeamento).
- O `SessionItemsPanel` lista itens com `SessionItemStatusChip`; compartilha remoção com o
  `DetectionPopup`. Simétrico ao `ScannerLogPanel` à direita.
- Frontend já está modularizado: `App.jsx` é só roteador, telas vivem em
  `src/screens/`, hooks em `src/hooks/`, components em `src/components/` e o
  estado global em `src/context/`. Use `useAppState()` (de
  `context/appStateContextValue.js`) para acessar grupos, sessão, toasts e
  navegação. Não reintroduza lógica no `App.jsx`.

## 5. Antes de declarar “pronto”

Use a checklist completa em `AGENTS.md` §7 (Definition of Done). Resumo:

- Lint/build limpos no front; servidor sobe limpo no back.
- Migrations rodam de banco zerado + seed + validate.
- Sem segredos, sem binários grandes acidentais, sem `*.db` modificado sem motivo.
- Comentários preservados; mensagens em PT-BR; RFs/RNFs intactos.

## 6. Git

- Só comite quando solicitado explicitamente.
- Mensagens em PT-BR, curtas, focadas no _porquê_.
- Nunca `--force` em `main`/`master`. Nunca amend após push, exceto pedido explícito.
- Nunca altere `git config`, nunca use `-i` em nenhum comando git.

## 7. Ambiguidade

Se o pedido for ambíguo entre duas abordagens com trade-offs reais (ex.: SQLite
vs. Postgres em dev; Context único vs. Zustand; granularidade fina vs. média na
quebra de componentes), **pare e pergunte** com `AskQuestion` antes de executar.
Para decisões pequenas e reversíveis, escolha a opção mais conservadora e siga.

## 8. Lições recorrentes do front

- **Lazy initializer no `useState`** para hidratar de `localStorage`/sessão —
  evita o erro `react-hooks/set-state-in-effect`. Veja
  `hooks/usePersistedAppState.js`.
- **Provider em `.jsx`, hook + context em `.js` irmão** — exigência da regra
  `react-refresh/only-export-components`. Padrão atual: `AppStateContext.jsx`
  exporta só o `<AppStateProvider>`; `appStateContextValue.js` exporta
  `AppStateContext` + `useAppState`.
- **`useEffect` que copia `videoRef.current` para variável local** antes do
  cleanup — sem isso, o lint reclama porque o ref pode ter mudado.
- **`forwardRef`** quando um componente precisar expor ref ao pai (o `DetectionPopup`
  atual não usa ref).
- **Animações pesadas** (`@keyframes`, gradientes, sombras) ficam em `index.css` com
  prefixos namespaced (`detection-popup-flash`, `session-item-chip*`,
  animação das linhas do scoreboard com `sessionItemsRowEnter`); evite `style={{...}}`
  inline para classes reutilizáveis.
