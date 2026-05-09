# AGENTS.md — AbraceAI

> Guia canônico para agentes de IA (Claude, Cursor, Codex, Copilot, Gemini, etc.) que
> trabalham neste repositório. **Leia este arquivo inteiro antes da primeira ação.**
> Outras LLMs (incluindo `CLAUDE.md`) referenciam este documento como fonte da verdade.

---

## 1. Visão geral do produto

**AbraceAI** é um sistema web de **classificação e contagem inteligente de alimentos
arrecadados** para o programa _Lideranças Empáticas_ (FECAP). Substitui a contagem
manual em planilhas por um pipeline automatizado: câmera do navegador → detecção
YOLO no backend → contagem por equipe e por categoria, com evidência visual.

- Documento de concepção completo: `Documentos/AbraceAI_Concepcao_Produto.md` (sempre
  consulte para dúvidas de produto, RFs/RNFs, modelo de dados ou arquitetura).
- README institucional do repositório: `README.md`.
- README da entrega corrente: `src/README.md`.

**Stack alvo/atual (MVP PoC):** React + Vite (frontend), FastAPI + WebSocket nativo (backend),
SQLAlchemy + Alembic + SQLite/PostgreSQL (dados), YOLOv8 (Ultralytics) + OpenCV +
ByteTrack (visão computacional), Azure (deploy futuro).

**Idioma do código e do domínio: Português (Brasil).** Identificadores, modelos,
endpoints, tabelas, schemas e mensagens ao usuário são em PT-BR. Comentários técnicos
e mensagens de commit também em PT-BR, salvo termos consagrados (`bbox`, `tracker`,
`fine-tuning`, `frame`).

---

## 2. Estrutura do repositório

```
projeto-1-abrace-ai/
├── AGENTS.md                       ← este arquivo (fonte canônica)
├── CLAUDE.md                       ← delega para AGENTS.md
├── README.md                       ← visão institucional do repositório
├── Documentos/
│   ├── AbraceAI_Concepcao_Produto.md   ← especificação completa do produto
│   ├── Banner_FECAP_CCOMP5_AbraceAI.*  ← material de apresentação
│   ├── Entrega01/                       ← entregas acadêmicas (notebooks, relatórios)
│   ├── Entrega02/                       ← idem
│   └── ToDo/                            ← TODOs por área (cada arquivo = 1 escopo)
│       ├── TODO-api-python.md
│       ├── TODO-auditoria-camera.md
│       ├── TODO-banco-de-dados.md
│       ├── TODO-cadastro-alimentos-grupo.md
│       ├── TODO-dashboard-tempo-real.md
│       └── TODO-integracao-camera-frontend.md
├── Imagens/                        ← assets do projeto (banners, logos, mockups)
└── src/
    ├── README.md
    └── Entrega 2/                  ← código atual de desenvolvimento (PASTA COM ESPAÇO!)
        ├── backend/                ← FastAPI + SQLAlchemy + Alembic + YOLO
        │   ├── Dockerfile
        │   ├── docker-compose.yml
        │   ├── requirements.txt
        │   ├── README-API.md
        │   ├── api/
        │   │   ├── main.py                 ← FastAPI app, CORS, routers
        │   │   ├── dependencies.py         ← get_db
        │   │   ├── core/
        │   │   │   ├── config.py           ← Settings (Pydantic), DATABASE_URL
        │   │   ├── routers/                ← sessoes, deteccoes, relatorios, grupos, alimentos, ws_auditoria
        │   │   ├── schemas/                ← Pydantic schemas (sessao, deteccao, alimento, …)
        │   │   └── services/               ← YOLO, Gemini, estabilidade, evidências
        │   ├── bd/
        │   │   ├── README.md               ← detalhes do schema, alembic, seeds
        │   │   ├── alembic.ini
        │   │   ├── diagrama-db.mmd         ← ER em Mermaid (8 tabelas)
        │   │   ├── abraceai.db             ← SQLite local (não versionar dados)
        │   │   ├── models/                 ← SQLAlchemy ORM (PT-BR)
        │   │   │   ├── database.py         ← engine, SessionLocal, Base, get_db
        │   │   │   ├── usuario.py
        │   │   │   ├── grupo.py · aluno.py
        │   │   │   ├── grupo_alimento.py · alimento.py
        │   │   │   ├── sessao.py · item_declarado.py · deteccao.py
        │   │   ├── migrations/              ← Alembic (env.py + versions/)
        │   │   └── seeds/
        │   │       ├── seed_data.py
        │   │       └── validate_db.py
        │   └── modelo-visao-computacional/
        │       ├── live_detect_v3.py       ← detecção em tempo real via webcam
        │       ├── v3_final.pt             ← pesos YOLO (15 classes, ~19 MB)
        │       └── requirements.txt
        └── frontend/                ← React 19 + Vite (sem TS/Tailwind/Shadcn ainda)
            ├── package.json
            ├── vite.config.js
            ├── eslint.config.js
            ├── index.html
            ├── public/                     ← logos, favicon, ícones
            └── src/
                ├── main.jsx
                ├── App.jsx                 ← roteador de telas (~37 linhas, switch sobre currentScreen)
                ├── constants.js            ← API_BASE, WS_BASE, REALTIME_BASE
                ├── index.css               ← design system (custom CSS, vars, animações)
                ├── context/
                │   ├── AppStateContext.jsx     ← AppStateProvider (componente)
                │   └── appStateContextValue.js ← createContext + useAppState() hook
                ├── hooks/
                │   ├── useAuditoriaWS.js       ← WebSocket nativo da câmera (frames/preview/detecção preliminar + atualização Gemini)
                │   ├── useToasts.js            ← addToast + array de toasts
                │   ├── usePersistedAppState.js ← grupos + sessao_id em localStorage
                │   ├── useRealtimeSocket.js    ← Socket.IO :5000 (dashboard realtime)
                │   ├── useCameraStream.js      ← getUserMedia + cleanup do <video>
                ├── services/
                │   └── api.js                  ← criarDeteccao()/criarDeteccaoManual(), finalizarSessao(); fluxo WS não usa POST para auto‑captura
                ├── screens/                    ← uma tela por arquivo, consomem useAppState()
                │   ├── HomeScreen.jsx
                │   ├── CadastroScreen.jsx
                │   ├── ConfigScreen.jsx
                │   ├── DashboardScreen.jsx
                │   ├── CameraScreen.jsx        ← scanner + WS; captura contínua + chips Gemini no scoreboard
                │   ├── ManualScreen.jsx
                │   └── RealtimeScreen.jsx
                ├── components/
                │   ├── ToastContainer.jsx
                │   ├── UserPopup.jsx
                │   ├── GroupModal.jsx
                │   ├── KanbanCard.jsx
                │   ├── DetectionPopup.jsx          ← popup da câmera (fixo na base)
                │   ├── SessionItemStatusChip.jsx   ← chip de estado Gemini (validando/validado/corrigido/sem_gemini) por item
                │   ├── SessionItemsPanel.jsx       ← card lateral esquerdo (scoreboard) com itens da sessão
                │   ├── ScannerLogPanel.jsx
                │   └── realtime/
                │       ├── MetricCards.jsx
                │       ├── GroupBarCharts.jsx
                │       ├── ProductTable.jsx
                │       ├── RecentActivity.jsx
                │       └── GroupBreakdown.jsx
                └── assets/
```

**Convenções que afetam ferramentas e comandos:**

- O caminho `src/Entrega 2/` **contém um espaço** — sempre cite com aspas duplas
  (`"src/Entrega 2/backend"`) em comandos shell.
- O git status atual mostra muitos arquivos `D` (deletados) na raiz e renomeações
  para `src/Entrega 2/`. **Não restaure os arquivos da raiz** — a estrutura nova é a
  correta.

---

## 3. Stack e versões esperadas

| Camada                | Tecnologia                                      | Notas                                                     |
| --------------------- | ----------------------------------------------- | --------------------------------------------------------- |
| Frontend              | React 19, Vite 5, ESLint 9                      | JSX puro (sem TS). Sem Tailwind/Shadcn ainda no MVP atual |
| Estado/Realtime       | WebSocket nativo do browser                     | Scanner em `ws://localhost:8000/ws/auditoria/{sessao_id}` |
| Backend               | FastAPI, Uvicorn, Pydantic v2, pydantic-settings | Python 3.10+ (Dockerfile usa `python:3.10-slim`)          |
| Auth                  | Sem autenticação na PoC                         | Endpoints REST e WS liberados para teste local             |
| ORM/Migrations        | SQLAlchemy 2.0+, Alembic                        | `bd/migrations/versions/`; usar `-c bd/alembic.ini`       |
| Banco                 | SQLite (dev) / PostgreSQL (prod)                | URL via env `DATABASE_URL`                                |
| Visão Computacional   | YOLOv8 (Ultralytics), OpenCV, NumPy, PyTorch    | Modelo treinado: `v3_final.pt` (15 classes)               |
| Tracking              | ByteTrack / BoTSORT (integrados ao Ultralytics) | Para anti-duplicidade na contagem                         |
| Storage de evidências | Azure Blob Storage (prod) / FS local (dev)      | Coluna `imagem_path` em `deteccoes`                       |
| Deploy                | Docker, Azure Container Apps, GitHub Actions    | Roadmap das semanas 11–12                                 |

---

## 4. Comandos essenciais

> Sempre execute a partir da raiz do repositório, exceto quando indicado. Lembre que
> `Entrega 2` tem espaço.

### 4.1 Backend (FastAPI)

```bash
# Local
cd "src/Entrega 2/backend"
pip install -r requirements.txt
pip install -r bd/requirements.txt
./.venv/bin/uvicorn api.main:app --reload     # http://127.0.0.1:8000/docs

# Banco de dados
./.venv/bin/alembic -c bd/alembic.ini upgrade head  # aplicar migrations em bd/abraceai.db
./.venv/bin/python bd/seeds/seed_data.py            # popular seed (8 categorias + 15 alimentos YOLO)
./.venv/bin/python bd/seeds/validate_db.py          # validar schema e seeds

# Docker
docker-compose up --build                     # API em :8000 com SQLite montado
```

### 4.2 Frontend (React + Vite)

```bash
cd "src/Entrega 2/frontend"
npm install
npm run dev                                    # http://localhost:5173
npm run lint
npm run build
```

### 4.3 Visão computacional (live demo standalone)

```bash
cd "src/Entrega 2/backend/modelo-visao-computacional"
pip install ultralytics opencv-python numpy
python live_detect_v3.py --model v3_final.pt --source 0 --conf 0.35
# Atalhos durante a execução: q sai, s pausa, p salva screenshot
```

### 4.4 Variáveis de ambiente relevantes

| Var              | Default                          | Onde é lida                                |
| ---------------- | -------------------------------- | ------------------------------------------ |
| `DATABASE_URL`   | `sqlite:///bd/abraceai.db`       | `api/core/config.py`, `bd/models/database.py` |
| `YOLO_MODEL_PATH` | `modelo-visao-computacional/v3_final.pt` | `api/core/config.py` |
| `YOLO_CONF_THRESHOLD` | `0.35` | `api/core/config.py` |
| `GEMINI_API_KEY` | vazio | `api/services/gemini_service.py` — sem chave roda YOLO puro |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite-preview` | `api/services/gemini_service.py` |
| `STABILITY_SECONDS` / `STABILITY_IOU_MIN` / `LOCK_SECONDS` | `1.5` / `0.85` / `3.0` | `api/services/estabilidade_service.py` |
| `EVIDENCIA_DIR` | `backend/evidencias` | `api/services/evidencia_service.py` |

---

## 5. Arquitetura (resumo operacional)

```
[Câmera browser] ──getUserMedia──► [Frontend React]
                                         │
                                         ├── HTTP REST  (TanStack Query/Axios planejado)
                                        └── WebSocket nativo (frames base64 JPEG @ ~2 FPS)
                                                  │
                                                  ▼
                                         [Backend FastAPI + WS nativo]
                                                  │
                                ┌─────────────────┼──────────────────┐
                                ▼                 ▼                  ▼
                        [Pipeline YOLO]   [SQLAlchemy ORM]   [Storage de evidências]
                        Ultralytics+CV2    SQLite/Postgres    FS local → Azure Blob
                        + ByteTrack
```

**Fluxo do scanner (RF02–RF06) — auto-registro assíncrono:** o operador escolhe um
grupo na Home Kanban → abre Tela do Scanner → informa/usa o `sessao_id` → clica em
**Iniciar captura** → o front envia frames por WebSocket nativo (~2 FPS, JPEG
q=0.6, 640×480) → backend roda YOLO em cada frame e envia `preview` com bbox/
label/confiança → quando a bbox fica estável o suficiente (IoU + `STABILITY_SECONDS`),
backend aplica **lock** (`LOCK_SECONDS`, ver `estabilidade_service`) e envia
`status: analisando` (apenas sinal de gatilho; **não** pausa o front) → salva
evidência (`imagem_path`), faz **INSERT** em `deteccoes` já com YOLO (`fonte=YOLO`,
`gemini_*` nulos) e envia `deteccao_preliminar` com `deteccao_id` → o front adiciona o
item em `currentSessionItems` com chip **validando**, toast curto e **flash** no
`DetectionPopup`; a **captura de frames segue** sem snapshot/congelamento → o Gemini
roda em **background** (`asyncio.create_task` em `ws_auditoria.py`): ao terminar, o
backend **UPDATE** na mesma linha (`gemini_*`, e se discordar com classe mapeada troca
`alimento_id` + `fonte=GEMINI`) e envia `deteccao_atualizada` → o front localiza o item
por `deteccaoId` e troca o chip silenciosamente para **validado** / **corrigido** /
**sem_gemini** (tooltip em correção: `nomeOriginal` → `name`). Se Gemini estiver OFF ou
indisponível, vem `deteccao_atualizada` imediata com `gemini: null`. **`POST
/api/v1/deteccoes/` não participa** deste fluxo automático (persistência é do próprio WS);
`criarDeteccao()` em `api.js` fica como legado/testes ou integrações futuras. O botão
"ADICIONAR" do popup é fallback **manual**: `criarDeteccaoManual()` quando faz sentido +
campos digitados válidos (`currFood`/`currWeight`).

**Anti-duplicidade (RF04):** mecanismos combinados — `tracking ID` (ByteTrack) quando
aplicável + `zona de detecção` (ROI na UI) + **lock temporal** no WebSocket
(`LOCK_SECONDS` após cada disparo de estabilidade). Toda mudança em qualquer um deles
deve preservar a invariante: cada item físico é registrado **uma vez**.

**Modelo de dados (8 tabelas, ver `bd/diagrama-db.mmd` e `bd/README.md`):**
`usuarios`, `grupos`, `alunos`, `grupos_alimentos`, `alimentos`, `sessoes`,
`itens_declarados`, `deteccoes`. `deteccoes.alimento_id_original` fixa o alimento
mapeado no **INSERT** preliminar (YOLO); correções via Gemini (`UPDATE`) podem alterar
`alimento_id` e `fonte` sem reescrever `alimento_id_original` (importante para métricas
RNF02 e análise de erros). Correções **manuais** via API de correção seguem o fluxo REST
dedicado.

---

## 6. Convenções de código

### 6.1 Gerais

- **Nunca** remova comentários existentes (regra do mantenedor).
- Não adicione comentários óbvios que descrevem _o que_ o código faz; só explique
  intenção, trade-offs ou restrições não evidentes.
- Mantenha mensagens ao usuário, labels e logs em PT-BR.
- Antes de criar arquivos novos, prefira editar existentes. **Não crie READMEs ou
  docs** sem solicitação explícita.

### 6.2 Backend Python

- Estrutura por camada: `api/routers/<recurso>.py`, `api/schemas/<recurso>.py`,
  `bd/models/<entidade>.py`. Cada recurso novo segue esse trio.
- Roteadores incluídos em `api/main.py` com prefixo `settings.API_V1_STR` (`/api/v1`).
- PoC atual **não usa autenticação**: não há JWT, senha, `/auth` nem
  `Depends(get_current_active_user)`. `usuarios` é apenas cadastro simples de
  operador/admin; `sessoes.usuario_id` é opcional e pode usar o primeiro usuário
  seedado como fallback.
- Usar `Session = Depends(get_db)` (definido em `api/dependencies.py`).
- Schemas Pydantic separam `…Create`, `…Response`, `…Correcao` (padrão atual em
  `schemas/deteccao.py`).
- Migrations: `./.venv/bin/alembic -c bd/alembic.ini revision --autogenerate -m "descricao_em_snake_case"` →
  revisar diff → `./.venv/bin/alembic -c bd/alembic.ini upgrade head`. Nunca edite
  uma migration já aplicada em ambiente compartilhado; crie uma nova.
- SQLite tem `PRAGMA foreign_keys=ON` ativado via event listener (`bd/models/database.py`)
  — não desabilitar.

### 6.3 Frontend React

- **Arquitetura modular (já refatorada).** `App.jsx` é só um roteador de telas
  (`switch (currentScreen)`) envolto pelo `<AppStateProvider>`. Cada tela vive em
  `src/screens/<Nome>Screen.jsx` e consome o estado compartilhado via
  `useAppState()` importado de `src/context/appStateContextValue.js`. **Não
  reintroduza** lógica/JSX dentro do `App.jsx`.
- **Provider vs hook em arquivos separados.** O `<AppStateProvider>` mora em
  `context/AppStateContext.jsx` (componente) e o `useAppState()` + `createContext`
  em `context/appStateContextValue.js`. Essa separação satisfaz a regra
  `react-refresh/only-export-components` — ao adicionar novos hooks/contexts,
  mantenha o mesmo padrão (componente sozinho num arquivo `.jsx`, valores/hooks
  num `.js`).
- **Onde mora cada coisa:**
  - Estado global → `context/AppStateContext.jsx` + `useAppState()`.
  - Side-effects reutilizáveis → `hooks/use*.js` (toasts, localStorage, WS,
    camera stream).
  - Chamadas HTTP → `services/api.js` (`criarDeteccao`, `criarDeteccaoManual`, `finalizarSessao` — auto-captura persiste só pelo WS).
  - Sub-blocos grandes da UI → `components/*` (ex.: `DetectionPopup`,
    `GroupModal`, `KanbanCard`) e `components/realtime/*` para a Visualização Gráfica.
- **Fluxo do estado.** Estado persistido em `localStorage` sob duas chaves:
  `abraceai_appState` (grupos, shape `{ id, title, members:[{name,ra}], totalKg,
  items:[{name,weight}] }`) e `abraceai_sessao_id`. A leitura usa **lazy
  initializer** do `useState` para evitar `setState` dentro de `useEffect`
  (regra `react-hooks/set-state-in-effect`).
- **Câmera (`CameraScreen`).** Usa `hooks/useAuditoriaWS.js` (WebSocket nativo) +
  `hooks/useCameraStream.js` (getUserMedia). O `DetectionPopup` é fixo na base da tela.
  A captura começa manualmente por botão; **não reintroduza** envio
  automático no `onLoadedData` nem simulação hardcoded de alimentos/pesos.
  Eventos WS relevantes (servidor → cliente): `preview`, `status`, `deteccao_preliminar`,
  `deteccao_atualizada`, `log`, `erro` (mensagens cliente → servidor: `frame`,
  `reset`). O hook expõe `ultimaPreliminar` e `ultimaAtualizacaoGemini`; `CameraScreen`
  mantém dois `useEffect`s — um faz push na lista ao preliminar, o outro atualiza o chip
  ao Gemini concluir. **Sem** overlay de fullscreen nem pausa obrigatória durante o ciclo.
- **UX do scanner (auto-registro assíncrono).** Botão iniciar/pausar captura, toggle
  Gemini ON/OFF, bbox/label YOLO sobre o vídeo ao vivo, `ScannerLogPanel` (direita),
  `SessionItemsPanel` (esquerda) e `DetectionPopup` (fixo na base) com lista resumida. Cada
  item em `currentSessionItems` pode ter `deteccaoId`, `status` (`validando` |
  `validado` | `corrigido` | `sem_gemini`), `nomeOriginal` e campos opcionais mapeados
  em `CameraScreen`; o estado visual nos cards é renderizado por `SessionItemStatusChip`.
  Correção Gemini **não** dispara toast (só atualização discreta da linha); preliminar
  pode disparar toast curto de confirmação. O fluxo anterior com `DetectionConfirmedOverlay`
  (snapshot + cooldown 3 s) foi **removido** — não recriar.
- **Design system** em `src/index.css` com CSS vars (`--primary`, `--dark`,
  `--card`, `--gray-medium`, `--yellow`, etc.) e classes utilitárias custom
  (`btn`, `btn-primary`, `kanban-card`, `realtime-card`, `toast`,
  `detection-popup-flash`, `session-items-*`, `session-item-chip*`,
  `scanner-log-panel`, animação `sessionItemsRowEnter` nas novas linhas do scoreboard).
  **Não introduzir** Tailwind/Shadcn sem alinhamento — está no roadmap mas não no MVP atual.
- **Ícones:** Phosphor Icons via classes `ph` / `ph-fill` (CDN no `index.html`).
- **Granularidade.** Ao crescer uma tela, prefira extrair sub-componentes para
  `components/` em vez de inflar o arquivo. Mantenha cada tela ≲ 300 linhas
  para que uma LLM consiga reler isolada.

### 6.4 Convenções de nomenclatura

| Contexto                | Padrão                | Exemplo                           |
| ----------------------- | --------------------- | --------------------------------- |
| Tabelas                 | `snake_case` plural   | `itens_declarados`, `deteccoes`   |
| Colunas                 | `snake_case`          | `peso_kg`, `corrigido_manualmente` |
| Modelos SQLAlchemy      | `PascalCase` singular | `ItemDeclarado`, `Deteccao`       |
| Schemas Pydantic        | `PascalCase` + sufixo | `DeteccaoCreate`, `DeteccaoResponse` |
| Endpoints REST          | plural em PT-BR       | `/api/v1/sessoes`, `/api/v1/deteccoes` |
| Eventos WS              | `snake_case` em PT-BR | `frame`, `preview`, `deteccao_preliminar`, `deteccao_atualizada`, `reset` |
| Componentes React       | `PascalCase` JSX      | `DeteccaoModal`, `SessionBadge`   |
| Variáveis JS            | `camelCase`           | `currentSessionItems`, `realtimeStatus` |

---

## 7. Definition of Done para uma tarefa

Antes de declarar uma tarefa concluída, verifique:

- [ ] Lint passa: `npm run lint` (front) e/ou nenhum erro de import/Pylance (back).
- [ ] Build front passa: `npm run build`.
- [ ] Backend sobe sem erro: `uvicorn api.main:app --reload`.
- [ ] Migrations aplicam de banco zerado: `rm bd/abraceai.db && ./.venv/bin/alembic -c bd/alembic.ini upgrade head
      && ./.venv/bin/python bd/seeds/seed_data.py && ./.venv/bin/python bd/seeds/validate_db.py`.
- [ ] Não há credenciais ou tokens hardcoded em arquivos versionados.
- [ ] Mensagens visíveis ao usuário em PT-BR.
- [ ] RFs/RNFs afetados continuam atendidos (consulte `Documentos/AbraceAI_Concepcao_Produto.md` §6).
- [ ] Comentários originais preservados.
- [ ] `git status` revisado: nenhum binário grande inesperado, nenhum `*.db` modificado
      sem motivo, nenhum `node_modules/` ou `__pycache__` adicionado.

---

## 8. Pitfalls e armadilhas conhecidas

1. **Espaço em `Entrega 2/`** — quebra comandos sem aspas. Sempre quote.
2. **PoC sem autenticação** — endpoints REST/WS estão liberados para teste local.
   Antes de deploy real, reavaliar auth/autorização e CORS (`allow_origins=["*"]`).
3. **`v3_final.pt` (~19 MB)** é binário grande já versionado. Ao retreinar, considere
   git-lfs ou storage externo antes de comitar nova versão.
4. **`bd/abraceai.db` está no repositório.** O arquivo deve ser tratado como seed
   inicial; não comitar mudanças incidentais (use `git checkout -- bd/abraceai.db`
   se modificou sem querer durante testes).
5. **Scanner usa WebSocket nativo em `localhost:8000`** — não reintroduzir Socket.IO
   na câmera. O dashboard/realtime antigo ainda pode ter código legado separado.
6. **`api/main.py` faz `sys.path.append(BASE_DIR)`** para importar `bd.*` — não
   remover sem reorganizar os imports e o `Dockerfile`.
7. **Não reintroduzir mock de detecção no scanner.** O peso padrão deve vir de
   `alimentos.peso_padrao_kg`; a UI só deve registrar detecções reais vindas do WS
   ou entradas manuais explícitas.
7.1. **Auto-registro do scanner (WS).** A linha em `deteccoes` é criada no **backend**
   ao enviar `deteccao_preliminar` (INSERT com YOLO) e enriquecida em background com
   Gemini (UPDATE + `deteccao_atualizada`). O front **não** deve chamar `criarDeteccao()`
   nesse fluxo — duplicaria registro. Use `criarDeteccaoManual()` só no fallback quando o
   operador preenche dados manuais válidos. Tasks `asyncio.create_task` do Gemini podem
   terminar após o cliente desconectar (UPDATE no banco conclui mesmo se o `send_json`
   falhar).
7.2. **Classes YOLO sem mapeamento no banco.** Se o modelo prediz uma `classe_yolo` que não
   existe (ou difere por grafia) na tabela `alimentos` (`classe_yolo` ativa), o WS loga
   *“Sem alimento mapeado para a classe — preliminar descartada”* — nada é gravado e a
   sensação é de “demora” até o modelo estabilizar numa classe cadastrada ou o operador
   ajustar o seed/mapeamento.
8. **Modelo YOLO atual tem 15 classes**, mas o documento de concepção fala em 3
   classes base (Arroz/Feijão/Outros). Ao mexer no mapeamento, sincronize:
   `alimentos.classe_yolo` ↔ classes do `v3_final.pt`.
9. **Sensação de lentidão até “pegar” o objeto.** Somam-se: estabilização por IoU +
   `STABILITY_SECONDS` com apenas ~2 FPS de frames válidos; movimento do objeto ou da
   mão; lock
   `LOCK_SECONDS` após cada disparo; e baixa confiança (~30–45%) aumentando jitter da
   bbox. Não é necessariamente a API Gemini na primeira lista — ver item 7.2.
10. **Regras de lint do front que mordem.** O ESLint do frontend usa
   `eslint-plugin-react-hooks` na versão flat/recommended, que ativa duas regras
   menos óbvias:
   - `react-hooks/set-state-in-effect` — **prefira `useState(() => ...)` ou
     `useMemo`** em vez de `useEffect` que apenas chama `setState`. Quando a
     sincronização é genuinamente externa (ex.: limpar status quando um socket
     desconecta), use `// eslint-disable-next-line react-hooks/set-state-in-effect`
     com um comentário curto justificando.
   - `react-refresh/only-export-components` — um `.jsx` com `export function
     Provider()` não pode também exportar hook/`createContext` (quebra fast
     refresh). Solução: mover o hook + context para um `.js` irmão (padrão
     atual: `AppStateContext.jsx` + `appStateContextValue.js`).
11. **Cleanup de refs em hooks.** Em `useEffect`, copie `videoRef.current` para
    uma variável local antes do `return () => ...` — caso contrário o lint
    avisa que o ref já pode ter mudado quando o cleanup roda
    (`react-hooks/exhaustive-deps`). Padrão em `hooks/useCameraStream.js`.

---

## 9. Como navegar a documentação

| Pergunta do agente                                | Vá para                                                      |
| ------------------------------------------------- | ------------------------------------------------------------ |
| O que o produto faz? requisitos? jornada?         | `Documentos/AbraceAI_Concepcao_Produto.md`                   |
| Schema do banco, migrations, seeds                | `src/Entrega 2/backend/bd/README.md` + `diagrama-db.mmd`     |
| Endpoints da API e exemplo de payload             | `src/Entrega 2/backend/README-API.md` + `/docs` (Swagger)    |
| Tarefas em aberto por área                        | `Documentos/ToDo/TODO-*.md` (cada arquivo nomeia o responsável) |
| UI atual — estado global                          | `src/Entrega 2/frontend/src/context/AppStateContext.jsx` + `appStateContextValue.js` |
| UI atual — uma tela específica                    | `src/Entrega 2/frontend/src/screens/<Nome>Screen.jsx`        |
| UI atual — câmera (scanner contínuo + chips Gemini + scoreboard) | `screens/CameraScreen.jsx` + `hooks/useAuditoriaWS.js` + `components/DetectionPopup.jsx` + `components/SessionItemsPanel.jsx` + `components/SessionItemStatusChip.jsx` |
| UI atual — visualização gráfica                   | `screens/RealtimeScreen.jsx` + `components/realtime/*`       |
| Hooks reutilizáveis (toasts, WS, câmera)          | `src/Entrega 2/frontend/src/hooks/`                          |
| Demo do modelo de VC                              | `src/Entrega 2/backend/modelo-visao-computacional/live_detect_v3.py` |
| Roadmap e cronograma de 13 semanas                | `Documentos/AbraceAI_Concepcao_Produto.md` §12.2             |

---

## 10. Glossário (PT-BR)

| Termo                | Significado                                                                                  |
| -------------------- | -------------------------------------------------------------------------------------------- |
| **Lideranças Empáticas (LE)** | Programa da FECAP que organiza arrecadações solidárias por equipes de alunos.       |
| **Grupo / Equipe**   | Unidade arrecadadora (modelo `Grupo`). Cada grupo agrega `Aluno`s e `Sessao`s.               |
| **Sessão**           | Janela temporal de contagem operada por um usuário (`Sessao`, com `inicio`/`fim`/`status`).  |
| **Detecção**         | Item registrado pelo YOLO durante uma sessão (`Deteccao`).                                   |
| **Item declarado**   | Declaração manual feita pelo grupo **antes** da sessão (`ItemDeclarado`).                    |
| **Cooldown / lock**  | Janela anti-duplicidade no servidor após um disparo (`LOCK_SECONDS` em `EstadoSessao`): novos gatilhos são ignorados até expirar. Distinto do antigo overlay de 3 s (removido). |
| **Chip Gemini**      | `SessionItemStatusChip`: indica `validando` (background), `validado` (concorda), `corrigido` (tooltip `nomeOriginal` → nome atual) ou `sem_gemini`. |
| **Scoreboard (sessão)** | `SessionItemsPanel` à esquerda da câmera: métricas (itens, peso) + lista persistente com chips. Usa `currentSessionItems` + handler de remoção compartilhado com o popup. |
| **Zona de detecção** | Retângulo overlay na UI; só objetos dentro dela são candidatos a contagem.                   |
| **Auditoria**        | Conferência posterior usando a evidência visual (`imagem_path`) salva em cada detecção.      |
| **Triagem**          | Sinônimo operacional de "sessão de contagem" usado na UI atual.                              |

---

## 11. Princípios para o agente

1. **Leia antes de escrever.** Use Read/Grep/Glob para entender contexto adjacente
   antes de editar.
2. **Pequenas mudanças, alta densidade.** Evite refactors amplos sem solicitação.
3. **Preserve comentários, idioma e estilo existentes.**
4. **Em dúvida sobre produto, abra `Documentos/AbraceAI_Concepcao_Produto.md`.**
5. **Em dúvida sobre escopo de uma área, abra o `TODO-*.md` correspondente.**
6. **Não invente dependências.** Use as versões do `requirements.txt` /
   `package.json`. Para novas, prefira `pip install`/`npm i` (lockfile atualizado)
   em vez de chutar versões.
7. **Não comite segredos** nem arquivos `.env`/`.db` modificados acidentalmente.
8. **Não force push.** Não amend de commits que já foram para `origin`.
