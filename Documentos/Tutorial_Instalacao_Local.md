# Guia de execução local — AbraceAI

Este documento descreve, passo a passo, como clonar o repositório, instalar ferramentas e dependências e rodar o **backend** (FastAPI) e o **frontend** (React + Vite) do projeto **AbraceAI**. O código ativo da PoC está em `src/Entrega 2/`. Depois do clone, a ideia central é **dois terminais** (um para a API, outro para o Vite).

---

## 1. Pré-requisitos

### 1.1 Obrigatórios

| Ferramenta | Versão recomendada | Uso |
|------------|-------------------|-----|
| **Git** | Qualquer versão recente | Clonar e atualizar o repositório |
| **Python** | **3.10 ou superior** | Backend (o `Dockerfile` usa `python:3.10-slim`) |
| **pip** | Acompanha o Python | Instalar dependências Python |
| **Node.js** | **18 LTS ou superior** (20+ recomendado) | Frontend (React 19 + Vite 5) |
| **npm** | Vem com o Node | Instalar dependências JavaScript |

**Como verificar no terminal:**

```bash
git --version
python3 --version
pip3 --version
node --version
npm --version
```

### 1.2 Opcionais

| Ferramenta | Uso |
|------------|-----|
| **Docker** + **Docker Compose** | Subir só a API com container, sem configurar Python local |
| **Editor/IDE** | VS Code, Cursor, PyCharm, etc. |
| **Navegador moderno** | Chrome, Edge ou Firefox, necessário para **câmera** na tela do scanner |

### 1.3 Observações sobre hardware e IA

- **YOLO / Ultralytics**: o modelo (`v3_final.pt`) roda em CPU; GPU acelera mas não é obrigatória para desenvolvimento local.
- **Gemini (Google)**: opcional. Sem `GEMINI_API_KEY`, o sistema usa detecção YOLO e fluxos sem enriquecimento Gemini (comportamento documentado no projeto).

---

## 2. Clonar o repositório

```bash
git clone git@github.com:2026-1-NCC5/Projeto1.git
cd Projeto1
```

Confirme que a pasta `src/Entrega 2/backend` e `src/Entrega 2/frontend` existem.

---

## 3. Como rodar: dois terminais (backend + frontend)

Para desenvolver localmente, você usa **dois terminais ao mesmo tempo**:

| Terminal | Pasta | O que roda | URL usual |
|----------|--------|------------|-----------|
| **1º** | `src/Entrega 2/backend` | API FastAPI (Uvicorn) | http://127.0.0.1:8000 — Swagger: `/docs` |
| **2º** | `src/Entrega 2/frontend` | Interface React (Vite) | http://localhost:5173 |

O **frontend** é só a interface no navegador; **REST**, **WebSocket** (`/ws/auditoria/...`), **YOLO** e **banco** ficam no **backend**. Por isso a API precisa estar de pé no terminal 1 enquanto você usa o app servido pelo Vite no terminal 2.

**Ordem prática:** deixe o backend rodando no terminal 1; no terminal 2 rode o frontend (`npm run dev`). Abra o navegador na URL que o Vite mostrar (geralmente porta **5173**).

### 3.1 Copiar e colar: primeiro uso (dois terminais)

**Terminal 1 — backend**

```bash
cd Projeto1
cd "src/Entrega 2/backend"
python3 -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r bd/requirements.txt
alembic -c bd/alembic.ini upgrade head
python bd/seeds/seed_data.py
python bd/seeds/validate_db.py
uvicorn api.main:app --reload
```

**Terminal 2 — frontend:**

```bash
cd Projeto1
cd "src/Entrega 2/frontend"
npm install
npm run dev
```

Nas próximas vezes, costuma bastar: terminal 1 → pasta `backend`, venv ativo e `uvicorn api.main:app --reload`; terminal 2 → pasta `frontend` e `npm run dev`. Não é preciso recriar o `.venv` nem rodar `npm install` de novo, salvo mudança de dependências.

No navegador: app em **http://localhost:5173**; documentação da API em **http://127.0.0.1:8000/docs**.

O passo a passo detalhado de cada lado está nas seções **4** (backend) e **5** (frontend).

---

## 4. Backend (FastAPI)

Todos os comandos abaixo assumem que você está na **raiz do repositório** clonado, a menos que indicado o contrário.

### 4.1 Entrar na pasta do backend

```bash
cd "src/Entrega 2/backend"
```

### 4.2 Ambiente virtual Python

```bash
python3 -m venv .venv
```

**Ativar o ambiente:**

- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```
- **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```

### 4.3 Instalar dependências

```bash
pip install -r requirements.txt
pip install -r bd/requirements.txt
```

### 4.4 Banco de dados: migrations e seed

A URL padrão do SQLite está em `api/core/config.py` / variável `DATABASE_URL` (padrão típico: `sqlite:///bd/abraceai.db` relativo ao diretório de trabalho).

**Aplicar migrations:**

```bash
./.venv/bin/alembic -c bd/alembic.ini upgrade head
```

**Popular dados iniciais (categorias, alimentos YOLO, etc.):**

```bash
./.venv/bin/python bd/seeds/seed_data.py
```

**Validar schema e seeds (recomendado após seed):**

```bash
./.venv/bin/python bd/seeds/validate_db.py
```

### 4.5 Variáveis de ambiente (opcional)

Você pode criar um arquivo `.env` na pasta do backend ou exportar variáveis no shell. Exemplos úteis:

| Variável | Descrição |
|----------|-----------|
| `DATABASE_URL` | URL do banco (SQLite local ou PostgreSQL em produção) |
| `GEMINI_API_KEY` | Chave da API Google Gemini para validação/correção assistida |
| `GEMINI_MODEL` | Modelo Gemini (há default no código se não definir) |
| `YOLO_MODEL_PATH` | Caminho para `v3_final.pt` (default aponta para `modelo-visao-computacional/`) |

Sem `GEMINI_API_KEY`, a API e o scanner continuam funcionando com YOLO.

### 4.6 Subir o servidor com Uvicorn

O **Uvicorn** é o servidor ASGI que executa a aplicação FastAPI. Ele já vem instalado com o backend (`uvicorn[standard]` em `requirements.txt`). Não é um programa separado para instalar além do `pip install -r requirements.txt`.

Com o ambiente virtual **ativado** e estando na pasta `backend`, use o comando usual:

```bash
uvicorn api.main:app --reload
```

Se preferir **sem** ativar o venv (chama o executável dentro do `.venv`):

```bash
./.venv/bin/uvicorn api.main:app --reload
```

- API: **http://127.0.0.1:8000**
- Documentação interativa (Swagger): **http://127.0.0.1:8000/docs**

Deixe esse terminal aberto enquanto desenvolve ou testa o frontend.

---

## 5. Frontend (React + Vite)

Use um **segundo terminal** com o backend ainda rodando no primeiro.

### 5.1 Entrar na pasta do frontend

Da raiz do repositório:

```bash
cd "src/Entrega 2/frontend"
```

### 5.2 Instalar dependências

```bash
npm install
```

### 5.3 Configuração da URL da API (opcional)

Por padrão o frontend usa `http://localhost:8000` para REST e deriva `ws://` para o WebSocket (`constants.js`). Para outro host/porta, use variáveis do Vite ao subir o dev server, por exemplo:

```bash
VITE_API_BASE=http://127.0.0.1:8000 npm run dev
```

### 5.4 Modo desenvolvimento

```bash
npm run dev
```

O Vite normalmente expõe a aplicação em **http://localhost:5173** (confira a URL impressa no terminal).

### 5.5 Qualidade de código e build (opcional)

```bash
npm run lint
npm run build
```

---

## 6. Alternativa: Docker (somente API)

Se você tem Docker instalado e prefere não configurar Python local para a API:

```bash
cd Projet1
cd "src/Entrega 2/backend"
docker compose up --build
```

A API fica em **http://localhost:8000**. O frontend continua sendo executado com `npm run dev` na pasta `frontend`, apontando para essa URL.

---

## 7. Problemas comuns

| Situação | O que fazer |
|----------|-------------|
| Comando `cd` falha em `Entrega 2` | Use aspas: `cd "src/Entrega 2/backend"` |
| Porta **8000** ocupada | Encerre o processo que usa a porta ou altere a porta do `uvicorn` (`--port 8001`) e ajuste `VITE_API_BASE` no frontend |
| Porta **5173** ocupada | O Vite sugere outra porta automaticamente; use a URL mostrada no terminal |
| Erros de migration ou banco inconsistente | Da pasta `backend`, após backup: remover `bd/abraceai.db` (se aplicável), rodar `alembic upgrade head`, `seed_data.py`, `validate_db.py` |
| Câmera não abre no navegador | Use **HTTPS ou localhost**; conceda permissão de câmera no navegador |
| Dependências Python quebram (PyTorch/Ultralytics) | Confira versão do Python (3.10+); em ambiente restrito, consulte `requirements.txt` e documentação do Ultralytics |

---