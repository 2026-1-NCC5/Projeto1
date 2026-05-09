# AbraceAI Analytics

> Sistema academico de monitoramento inteligente de recursos em nuvem com IA.
> Coleta metricas via `psutil`, treina 4 exercicios de IA (regressao, classificacao,
> clusterizacao + PCA) e expoe um dashboard Streamlit empacotado em Docker.
> Projetado para deploy em EC2 t3.micro (AWS Free Tier) e teardown rapido.

Pasta-raiz do projeto: `Documentos/Entrega02/Sistemas_Operacionais/abrace-ai-analytics/`.
Projeto isolado do AbraceAI principal (captura de alimentos) que vive em `src/Entrega 2/`.

---

## 1. Objetivo

Implementar a proposta descrita em
[`relatorio_projeto_monitoramento_ia_nuvem.md`](../relatorio_projeto_monitoramento_ia_nuvem.md):

1. **Coletar** CPU / memoria / disco / rede / load de uma instancia Linux.
2. **Armazenar** o historico (CSV rotacionado por dia).
3. **Pre-processar** (limpeza, features, rotulos `normal`/`atencao`/`critico`).
4. **Treinar** 4 modelos de IA:
   - Regressao para prever `cpu_percent` no proximo passo.
   - Classificacao do estado da instancia.
   - Clusterizacao com K-Means e DBSCAN.
   - PCA para visualizacao 2D.
5. **Visualizar** num dashboard Streamlit em tempo real.
6. **Empacotar** com Docker + Docker Compose.
7. **Deploy** em EC2 com checklist de teardown (priorizando custo zero).

---

## 2. Estrutura

```text
abrace-ai-analytics/
├── README.md                       # este arquivo
├── pyproject.toml                  # Python 3.12, deps fixadas
├── requirements.txt                # mesmo que pyproject (compat com scripts)
├── requirements-dev.txt            # pytest, ruff
├── .env.example
├── docker/
│   ├── Dockerfile                  # multi-stage; entrypoint parametrizado por RUN_MODE
│   └── entrypoint.sh
├── docker-compose.yml              # services: collector, dashboard, trainer (manual)
├── app/
│   ├── common/                     # config (pydantic-settings), logging, storage CSV, paths
│   ├── collector/                  # collect_metrics + load_generator (idle/cpu/mem/net/mixed)
│   ├── training/                   # preprocess + 3 trainers + evaluate_models
│   └── dashboard/                  # Streamlit com 5 abas e auto-refresh
├── data/
│   ├── raw/                        # CSV diarios gerados pelo coletor
│   ├── processed/                  # dataset.parquet/csv (preprocessado)
│   └── models/                     # *.joblib (regressao, classificacao, KMeans, DBSCAN, PCA, scaler)
├── reports/
│   ├── figures/                    # PNGs de regressao, matriz de confusao, clusters, PCA
│   └── metrics/                    # *.json + summary.md (pronto para colar no relatorio)
├── scripts/
│   ├── seed_synthetic.py           # gera ~12h de dataset sintetico para destravar treino
│   └── generate_load.sh            # wrapper sobre o load_generator
├── tests/                          # pytest (storage, preprocess, smoke do coletor)
└── infra/
    ├── cloudformation.yml          # stack EC2 + SG (validado via MCP awsiac)
    ├── deploy_aws.sh               # cria/atualiza a stack
    ├── teardown_aws.sh             # destroi tudo + lista volumes/EIPs remanescentes
    └── AWS_DEPLOY.md               # passo a passo detalhado + checklist de teardown
```

---

## 3. Stack

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.12 |
| Dados | pandas, numpy, pyarrow |
| ML | scikit-learn 1.5+ |
| Sistema | psutil |
| Persistencia | CSV rotacionado (interface trocavel para SQLite/Postgres) |
| Visualizacao | matplotlib, seaborn, Streamlit + streamlit-autorefresh |
| Config | pydantic-settings + .env |
| Container | Docker multi-stage (~600 MB final) + Docker Compose v2 |
| IaC | CloudFormation (validado via MCP `awsiac`) |
| Deploy | AWS EC2 t3.micro Ubuntu 24.04 (Free Tier) |

---

## 4. Execucao local

### 4.1 Setup

```bash
cd Documentos/Entrega02/Sistemas_Operacionais/abrace-ai-analytics

python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]   # ou: pip install -r requirements-dev.txt
cp .env.example .env
```

### 4.2 Pipeline completa em 4 comandos

```bash
# 1) gera ~12h de dataset sintetico (alternativa rapida ao coletor real)
PYTHONPATH=. python scripts/seed_synthetic.py --hours 12 --interval 5

# 2) preprocessamento (features + rotulos)
PYTHONPATH=. python -m app.training.preprocess

# 3) treina os 4 exercicios em sequencia
PYTHONPATH=. python -m app.training.train_regression
PYTHONPATH=. python -m app.training.train_classification
PYTHONPATH=. python -m app.training.train_clustering
PYTHONPATH=. python -m app.training.evaluate_models

# 4) dashboard
PYTHONPATH=. streamlit run app/dashboard/dashboard.py --server.port 8501
```

Abra `http://localhost:8501`. Cinco abas: **Tempo real / Previsao CPU / Estado / Clusters / Metricas dos modelos**.

### 4.3 Coleta real (opcional)

```bash
# Em um terminal: coletor em loop infinito (Ctrl+C para parar)
PYTHONPATH=. python -m app.collector.collect_metrics

# Em outro terminal: gerar carga durante a coleta
bash scripts/generate_load.sh mixed 600
```

### 4.4 Testes

```bash
PYTHONPATH=. pytest -v
```

---

## 5. Execucao com Docker

```bash
cd Documentos/Entrega02/Sistemas_Operacionais/abrace-ai-analytics

# Sobe collector (loop infinito) + dashboard (porta 8501)
cp .env.example .env
docker compose up -d --build

# Status
docker compose ps
docker compose logs -f collector

# Treino on-demand (perfil 'manual', nao sobe automaticamente)
docker compose run --rm trainer python -m app.training.preprocess
docker compose run --rm trainer python -m app.training.train_regression
docker compose run --rm trainer python -m app.training.train_classification
docker compose run --rm trainer python -m app.training.train_clustering
docker compose run --rm trainer python -m app.training.evaluate_models

# Encerrar
docker compose down
```

**Volumes:** `./data` e `./reports` sao montados nos containers, entao tudo gerado fica visivel no host.

---

## 6. Deploy AWS - resumo

> Detalhes completos, custos, troubleshooting e **checklist de teardown** em [`infra/AWS_DEPLOY.md`](infra/AWS_DEPLOY.md).

### 6.1 Custo (validado via MCP `awspricing`)

| Recurso | On-demand | Free Tier (conta antiga) |
|---|---|---|
| EC2 t3.micro Linux us-east-1 | US$ 0.0104/h | 750h/mes gratis |
| EBS gp3 12 GB | ~US$ 0.96/mes | 30 GB gratis |
| IPv4 publico | US$ 0.005/h | 750h/mes gratis |

**Custo esperado durante a captura de evidencias: US$ 0.00.**
Risco real: esquecer ligado por 1 mes (~US$ 7.50). Mitigado pelo `teardown_aws.sh`.

### 6.2 Subir

```bash
# Pre-requisito: aws CLI configurado, KeyPair criada, IP publico conhecido
chmod +x infra/*.sh
bash infra/deploy_aws.sh
```

O script pergunta KeyPair, CIDR autorizado e URL do git, valida tudo e cria a stack
`abrace-ai-analytics` em `us-east-1`. Outputs incluem `SshCommand`, `DashboardUrl`
e `TeardownCommand` ja prontos para colar.

### 6.3 Acessar

```bash
ssh -i ~/Downloads/abrace-ai-analytics.pem ubuntu@<PublicDnsName>
sudo tail -50 /var/log/cloud-init-output.log     # bootstrap
docker compose ps
```

Navegador: `http://<PublicDnsName>:8501` (so funciona se voce estiver no IP do `AllowedAppCidr`).

### 6.4 TEARDOWN OBRIGATORIO

```bash
bash infra/teardown_aws.sh
```

O script confirma, deleta a stack, aguarda conclusao e **lista volumes EBS, Elastic IPs e snapshots remanescentes** para voce conferir manualmente. Detalhes do checklist em `infra/AWS_DEPLOY.md`.

---

## 7. Resultados produzidos

Sobre o dataset sintetico de 12h (8640 amostras):

- **Regressao** (target `cpu_percent_t+1`): RandomForestRegressor vence com **R²=0.75**, MAE 8.24%, RMSE 12.47%. Polinomial deg=2 explode (overfit no spline com fail validation -> R² negativo) e Linear fica em R²=0.61.
- **Classificacao** (3 classes): DecisionTree e RandomForest atingem **F1-macro 0.998** com CV 5-fold consistente. KNN cai para 0.58 por sensibilidade a feature scaling+ruido.
- **Clusterizacao**: K-Means com `k=2` (silhouette 0.65) separa "ocioso" (cpu medio 28%) de "carregado" (cpu medio 74%). DBSCAN encontra 8 sub-grupos (eps=0.241).
- **PCA(2)**: 95.6% de variancia explicada (PC1 77.4% + PC2 18.1%) - reducao excelente.

Veja `reports/metrics/summary.md` para a tabela completa pronta para o relatorio academico.

---

## 8. Configuracao

Todas as variaveis ficam em `.env` (exemplo em `.env.example`):

| Variavel | Default | Descricao |
|---|---|---|
| `COLLECTION_INTERVAL_SECONDS` | `5` | intervalo entre coletas |
| `DATA_DIR` / `MODEL_DIR` / `REPORT_DIR` | `./data`, `./data/models`, `./reports` | diretorios base |
| `DASHBOARD_PORT` / `DASHBOARD_HOST` | `8501`, `0.0.0.0` | exposicao do Streamlit |
| `DASHBOARD_WINDOW_MIN` | `60` | janela exibida em tempo real |
| `DASHBOARD_REFRESH_SECONDS` | `10` | auto-refresh (0 desativa) |
| `THRESH_NORMAL_CPU` / `THRESH_NORMAL_MEM` | `50` / `60` | limite superior do estado "normal" |
| `THRESH_CRITICAL_CPU` / `THRESH_CRITICAL_MEM` | `80` / `80` | limite inferior do estado "critico" |
| `RUN_MODE` | `collector` | usado pelo entrypoint Docker (`collector\|dashboard\|trainer\|shell`) |
| `LOG_LEVEL` / `LOG_FORMAT` | `INFO` / `text` | formato `text` ou `json` (CloudWatch-friendly) |

---

## 9. Reprodutibilidade

- `random_state=42` em todos os modelos.
- Versoes fixadas em `pyproject.toml` / `requirements.txt`.
- Split temporal (sem shuffle) na regressao e classificacao -> evita data leakage.
- Pipelines `sklearn` encapsulam scaler+modelo -> mesmo objeto serve para predict no dashboard.
- `seed_synthetic.py --seed 42` gera o mesmo dataset bit a bit.

---

## 10. O que ainda nao esta no escopo

Conforme combinado com o orientador (PoC academica, custo zero):

- Sem TensorFlow / redes neurais (aumenta imagem Docker e nao e exigido).
- Sem Postgres/RDS (CSV resolve para 8k+ amostras).
- Sem CloudWatch agent / Grafana (logs de `docker compose logs` bastam).
- Sem Nginx/proxy reverso (acesso direto na 8501 com SG restritivo).
- Sem CI/CD.
- Sem integracao com o repositorio principal AbraceAI (projeto deliberadamente isolado).

---

## 11. Referencias

- Relatorio com a especificacao completa: [`relatorio_projeto_monitoramento_ia_nuvem.md`](../relatorio_projeto_monitoramento_ia_nuvem.md).
- Documentacao AWS Free Tier (pesquisada via MCP `awsknowledge`): https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-free-tier-usage.html
- Pricing oficial t3.micro us-east-1 (consultado via MCP `awspricing`): US$ 0.0104/h on-demand.
- Template CFN validado via MCP `awsiac` (`validate_cloudformation_template`).
