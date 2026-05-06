# AbraceAI Backend API (FastAPI)

API REST para o sistema de triagem e classificação de alimentos com Visão Computacional.

## Tecnologias
- **FastAPI**: Framework web de alta performance.
- **SQLAlchemy + Alembic**: ORM e Migrations (aproveitando os modelos em português já existentes na pasta `bd`).
- **Pydantic**: Validação de dados de entrada e saída.
- **JWT (python-jose)**: Autenticação baseada em token.

## Como Executar (Local)

1. Na pasta raiz do backend (`src/Entrega 2/backend/`), instale as dependências combinadas:
   ```bash
   pip install -r requirements.txt
   pip install -r bd/requirements.txt
   ```

2. Inicie o servidor FastAPI:
   ```bash
   uvicorn api.main:app --reload
   ```

3. Acesse a documentação Swagger interativa em:
   [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Como Executar (Docker)

Caso tenha o Docker instalado na sua máquina:

```bash
cd "src/Entrega 2/backend"
docker-compose up --build
```
A API estará rodando em [http://localhost:8000](http://localhost:8000).

## Integração com a IA
O sistema de Visão Computacional pode enviar as detecções enviando um `POST` para `/api/v1/deteccoes/`.
Exemplo de Payload:
```json
{
  "sessao_id": 1,
  "alimento_id": 3,
  "peso_kg": 1.5,
  "quantidade": 1,
  "confianca": 0.95
}
```
