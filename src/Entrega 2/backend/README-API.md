# AbraceAI Backend API (FastAPI)

API REST para o sistema de triagem e classificação de alimentos com Visão Computacional.

## Tecnologias
- **FastAPI**: Framework web de alta performance.
- **SQLAlchemy + Alembic**: ORM e Migrations (aproveitando os modelos em português já existentes na pasta `bd`).
- **Pydantic**: Validação de dados de entrada e saída.
- **PoC sem autenticação**: endpoints REST liberados para simplificar testes locais.

## Como Executar (Local)

1. Na pasta raiz do backend (`src/Entrega 2/backend/`), instale as dependências combinadas:
   ```bash
   pip install -r requirements.txt
   pip install -r bd/requirements.txt
   ```

2. Aplique migrations e seed no banco local:
   ```bash
   ./.venv/bin/alembic -c bd/alembic.ini upgrade head
   ./.venv/bin/python bd/seeds/seed_data.py
   ```

3. Inicie o servidor FastAPI:
   ```bash
   ./.venv/bin/uvicorn api.main:app --reload
   ```

4. Acesse a documentação Swagger interativa em:
   [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

> A PoC atual não exige login nem token. Crie sessões, grupos, alimentos e itens
> declarados diretamente pelo Swagger.

## Como Executar (Docker)

Caso tenha o Docker instalado na sua máquina:

```bash
cd "src/Entrega 2/backend"
docker-compose up --build
```
A API estará rodando em [http://localhost:8000](http://localhost:8000).

## Integração com a IA
O scanner do frontend usa WebSocket nativo em `/ws/auditoria/{sessao_id}`:

- cliente envia `{ "tipo": "frame", "imagem_b64": "...", "usar_gemini": true|false }`;
- backend responde `preview` com bbox/classe/confiança do YOLO para overlay visual;
- backend envia `log`, `status`, `erro` e `deteccao` consolidada;
- frontend confirma a detecção via `POST /api/v1/deteccoes/`.

Exemplo de Payload:
```json
{
  "sessao_id": 1,
  "alimento_id": 3,
  "alimento_id_original": 3,
  "peso_kg": 1.5,
  "quantidade": 1,
  "confianca": 0.95,
  "fonte": "YOLO",
  "imagem_path": "evidencias/1/20260507T170000_000.jpg"
}
```
