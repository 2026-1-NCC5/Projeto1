# Backend AbraceAI — Banco de Dados

Banco de dados relacional SQLite para o sistema AbraceAI, responsável por gerenciar grupos de triagem, alunos, sessões de contagem/auditoria, detecções via visão computacional e declarações manuais de alimentos.

---

## Stack Técnica

| Componente | Ferramenta |
|------------|-----------|
| ORM | SQLAlchemy 2.0+ |
| Migrations | Alembic |
| Banco (Dev) | SQLite (`abraceai.db`) |
| Banco (Prod) | PostgreSQL (via `DATABASE_URL`) |

---

## Estrutura de Diretórios

```
backend/
├── requirements.txt          # Dependências Python
├── alembic.ini              # Configuração do Alembic
├── abraceai.db              # Banco SQLite (não versionado)
├── .gitignore               # Ignora *.db, __pycache__, etc.
├── models/                  # Models SQLAlchemy
│   ├── __init__.py
│   ├── database.py          # Engine, SessionLocal, Base, get_db()
│   ├── usuario.py
│   ├── grupo.py
│   ├── aluno.py
│   ├── grupo_alimento.py
│   ├── alimento.py
│   ├── sessao.py
│   ├── item_declarado.py
│   └── deteccao.py
├── migrations/              # Migrations Alembic
│   ├── env.py
│   ├── script.py.mako
│   ├── README
│   └── versions/
│       ├── 40469500dd70_initial_schema.py
│       └── 8172a49d6648_add_indexes.py
└── seeds/
    ├── seed_data.py         # Popula dados iniciais
    └── validate_db.py       # Valida estrutura do banco
```

---

## Schema (8 Tabelas)

| Tabela | Descrição |
|--------|-----------|
| `usuarios` | Operadores e administradores do sistema |
| `grupos` | Grupos de triagem/arrecadação |
| `alunos` | Alunos vinculados a um único grupo |
| `grupos_alimentos` | Categorias de alimentos (Grãos, Café, etc.) |
| `alimentos` | Catálogo oficial de alimentos + classe YOLO |
| `sessoes` | Sessão de triagem/auditoria por grupo |
| `itens_declarados` | Declaração manual do que o grupo arrecadou |
| `deteccoes` | Detecções da câmera/YOLO + correção manual |

> Veja o diagrama completo em `diagrama-db.mermaid`.

---

## Como Rodar

### 1. Instalar Dependências

```bash
cd "src/Entrega 2/backend"
pip install -r requirements.txt
```

### 2. Criar/Aplicar Migrations

O banco já vem criado e populado no repositório. Caso precise recriar do zero:

```bash
# Apagar banco existente
rm abraceai.db

# Aplicar todas as migrations
alembic upgrade head
```

### 3. Popular Seed Data

```bash
python seeds/seed_data.py
```

Saída esperada:
```
🌱 Iniciando seed data...
✅ Grupos de alimentos inseridos com sucesso.
✅ Alimentos iniciais inseridos com sucesso.
🎉 Seed data concluído!
```

### 4. Validar Banco

```bash
python seeds/validate_db.py
```

Saída esperada:
```
🔍 Validando banco de dados AbraceAI...

✅ Todas as 8 tabelas presentes.
✅ Foreign keys ativadas (PRAGMA foreign_keys=ON).
✅ Todos os 9 índices presentes.
✅ Seed data OK (8 grupos, 3 alimentos).
✅ Constraints verificadas.

🎉 Banco de dados validado com sucesso!
```

---

## Comandos Úteis do Alembic

| Comando | Descrição |
|---------|-----------|
| `alembic revision --autogenerate -m "descricao"` | Cria nova migration a partir dos models |
| `alembic upgrade head` | Aplica todas as migrations pendentes |
| `alembic downgrade -1` | Reverte a última migration |
| `alembic current` | Mostra a migration atual |
| `alembic history` | Lista todas as migrations |

---

## Configurações Importantes

### Foreign Keys no SQLite

O SQLite não enforce foreign keys por padrão. Isso é ativado automaticamente via evento no `database.py`:

```python
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()
```

### Variável de Ambiente

Para usar PostgreSQL em produção, defina:

```bash
export DATABASE_URL="postgresql://user:pass@host:port/dbname"
```

Se não definida, o padrão é `sqlite:///abraceai.db`.

---

## Decisões de Design

| Decisão | Justificativa |
|---------|---------------|
| `itens_declarados` vinculado a `grupos` | Declaração manual é feita **antes** da sessão de triagem |
| `deteccoes` com `quantidade` | Usuário informa quantas unidades iguais existem, evitando passar o mesmo item várias vezes |
| `deteccoes` com histórico de correção | `alimento_id_original` guarda o que o YOLO detectou vs. o que o operador corrigiu |
| `grupo_membros` removida | Operador/admin acessa todos os grupos; não há vínculo fixo |
| Índices parciais | `idx_alimentos_classe_yolo` usa `WHERE classe_yolo IS NOT NULL` para otimizar buscas |

---

## Seed Data Inicial

### Grupos de Alimentos (8)
Grãos 🌾 | Enlatados 🥫 | Massas 🍝 | Laticínios 🧀 | Óleos 🫒 | Farináceos 🌽 | Açúcar 🧂 | Café ☕

### Alimentos Iniciais (3)
- Arroz — 5.0 kg — classe YOLO: `arroz`
- Feijão — 1.0 kg — classe YOLO: `feijao`
- Café — 0.5 kg — classe YOLO: `cafe`
