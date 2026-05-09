"""Dashboard Streamlit: monitoramento + saidas dos modelos.

Execucao local:
    streamlit run app/dashboard/dashboard.py --server.port 8501
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from app.common.config import get_settings
from app.common.storage import CsvStorage
from app.training.preprocess import FEATURE_COLUMNS, LABEL_CLASSES

st.set_page_config(
    page_title="AbraceAI Analytics",
    page_icon=":bar_chart:",
    layout="wide",
)

SETTINGS = get_settings()


# ----------------------------------------------------------------------------
# Carregamentos com cache curto - dados crescem em tempo real
# ----------------------------------------------------------------------------


@st.cache_data(ttl=5, show_spinner=False)
def _load_recent_raw(window_min: int) -> pd.DataFrame:
    storage = CsvStorage(SETTINGS.raw_dir)
    return storage.read_recent(window_min)


@st.cache_data(ttl=15, show_spinner=False)
def _load_processed() -> pd.DataFrame:
    parquet = SETTINGS.processed_dir / "dataset.parquet"
    if not parquet.exists():
        return pd.DataFrame()
    return pd.read_parquet(parquet)


@st.cache_resource(show_spinner=False)
def _load_model(name: str):
    path = SETTINGS.model_dir / name
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data(ttl=30, show_spinner=False)
def _load_summary() -> dict | None:
    path = SETTINGS.metrics_dir / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _classify_now(cpu: float, mem: float) -> str:
    if cpu > SETTINGS.thresh_critical_cpu or mem > SETTINGS.thresh_critical_mem:
        return "critico"
    if cpu < SETTINGS.thresh_normal_cpu and mem < SETTINGS.thresh_normal_mem:
        return "normal"
    return "atencao"


_LABEL_COLORS = {"normal": "#198754", "atencao": "#ffc107", "critico": "#dc3545"}


def _badge(label: str) -> str:
    color = _LABEL_COLORS.get(label, "#6c757d")
    return (
        f"<span style='background:{color};color:white;padding:6px 14px;"
        f"border-radius:8px;font-weight:600;'>{label.upper()}</span>"
    )


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------


with st.sidebar:
    st.header("AbraceAI Analytics")
    st.caption("Monitoramento inteligente com IA")
    st.divider()
    st.write(f"Janela exibida: **{SETTINGS.dashboard_window_min} min**")
    st.write(f"Refresh: **{SETTINGS.dashboard_refresh_seconds}s**")
    if SETTINGS.dashboard_refresh_seconds > 0:
        st_autorefresh(interval=SETTINGS.dashboard_refresh_seconds * 1000, key="dash_refresh")
    if st.button("Limpar cache", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.caption("Diretorios")
    st.code(
        f"data: {SETTINGS.data_dir}\nmodels: {SETTINGS.model_dir}\nreports: {SETTINGS.report_dir}",
        language="text",
    )


# ----------------------------------------------------------------------------
# Topo
# ----------------------------------------------------------------------------

st.title("AbraceAI Analytics")
st.caption("Monitoramento de recursos em nuvem com IA - PoC academica")

raw = _load_recent_raw(SETTINGS.dashboard_window_min)
if raw.empty:
    st.warning(
        "Nenhuma metrica encontrada em `data/raw/`. "
        "Rode `python -m app.collector.collect_metrics` ou `python scripts/seed_synthetic.py`."
    )
    st.stop()

latest = raw.iloc[-1]
classe_atual = _classify_now(latest["cpu_percent"], latest["memory_percent"])

tab_realtime, tab_forecast, tab_state, tab_clusters, tab_metrics = st.tabs(
    ["Tempo real", "Previsao CPU", "Estado", "Clusters", "Metricas dos modelos"]
)


# --------------------------- Tempo real ------------------------------------

with tab_realtime:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CPU (%)", f"{latest['cpu_percent']:.1f}")
    c2.metric("Memoria (%)", f"{latest['memory_percent']:.1f}")
    c3.metric("Disco (%)", f"{latest['disk_percent']:.1f}")
    c4.metric("Memoria livre (MB)", f"{latest['memory_available_mb']:.0f}")

    st.subheader("Serie temporal")
    df_plot = raw.set_index("timestamp")[
        ["cpu_percent", "memory_percent", "disk_percent"]
    ].rename(
        columns={
            "cpu_percent": "CPU %",
            "memory_percent": "Memoria %",
            "disk_percent": "Disco %",
        }
    )
    st.line_chart(df_plot, height=300)

    if {"net_bytes_sent", "net_bytes_recv"}.issubset(raw.columns):
        st.subheader("Rede (taxas)")
        net = raw[["timestamp", "net_bytes_sent", "net_bytes_recv"]].copy()
        net["sent_rate_kb_s"] = net["net_bytes_sent"].diff().clip(lower=0).fillna(0) / 1024
        net["recv_rate_kb_s"] = net["net_bytes_recv"].diff().clip(lower=0).fillna(0) / 1024
        st.line_chart(
            net.set_index("timestamp")[["sent_rate_kb_s", "recv_rate_kb_s"]],
            height=240,
        )

    st.subheader("Ultimas amostras")
    st.dataframe(raw.tail(20).iloc[::-1], width="stretch", hide_index=True)


# --------------------------- Previsao CPU ----------------------------------

with tab_forecast:
    st.subheader("Previsao do proximo valor de CPU")
    model = _load_model("regression_best.joblib")
    if model is None:
        st.info(
            "Modelo de regressao ainda nao foi treinado. "
            "Rode `python -m app.training.train_regression`."
        )
    else:
        processed = _load_processed()
        if processed.empty:
            st.warning("Dataset processado vazio. Rode preprocess.")
        else:
            features = processed[list(FEATURE_COLUMNS)].iloc[[-1]].to_numpy()
            try:
                yhat = float(model.predict(features)[0])
                cpu_atual = float(processed["cpu_percent"].iloc[-1])
                delta = yhat - cpu_atual
                col_a, col_b = st.columns(2)
                col_a.metric("CPU agora (%)", f"{cpu_atual:.1f}")
                col_b.metric("CPU prevista t+1 (%)", f"{yhat:.1f}", delta=f"{delta:+.1f}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Falha ao prever: {exc}")

            # Comparativo real vs previsto sobre janela recente
            recent = processed.tail(min(len(processed), 200))
            preds = model.predict(recent[list(FEATURE_COLUMNS)].to_numpy())
            vis = pd.DataFrame(
                {
                    "timestamp": recent["timestamp"].to_numpy(),
                    "real": recent["cpu_percent_t+1"].to_numpy(),
                    "previsto": preds,
                }
            ).set_index("timestamp")
            st.line_chart(vis, height=320)

            summary = _load_summary()
            if summary and summary.get("regression"):
                reg = summary["regression"]
                st.caption(f"Melhor modelo: **{reg['best_model']}**")
                st.dataframe(
                    pd.DataFrame(reg["results"]).T.round(4),
                    width="stretch",
                )


# --------------------------- Estado ----------------------------------------

with tab_state:
    st.subheader("Estado atual da instancia")
    st.markdown(_badge(classe_atual), unsafe_allow_html=True)
    st.caption(
        f"Limites - normal: CPU<{SETTINGS.thresh_normal_cpu} & MEM<{SETTINGS.thresh_normal_mem} | "
        f"critico: CPU>{SETTINGS.thresh_critical_cpu} ou MEM>{SETTINGS.thresh_critical_mem}"
    )

    model = _load_model("classification_best.joblib")
    if model is not None:
        processed = _load_processed()
        if not processed.empty:
            features = processed[list(FEATURE_COLUMNS)].iloc[[-1]].to_numpy()
            try:
                pred = str(model.predict(features)[0])
                st.write(
                    f"**Classificador (melhor modelo):** {_badge(pred)}",
                    unsafe_allow_html=True,
                )
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(features)[0]
                    classes = list(getattr(model, "classes_", LABEL_CLASSES))
                    st.bar_chart(pd.Series(proba, index=classes, name="probabilidade"))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Falha ao classificar: {exc}")
    else:
        st.info(
            "Modelo de classificacao ainda nao foi treinado. "
            "Rode `python -m app.training.train_classification`."
        )

    # Distribuicao de classes na janela recente
    recent_with_label = raw.copy()
    recent_with_label["label"] = recent_with_label.apply(
        lambda r: _classify_now(r["cpu_percent"], r["memory_percent"]), axis=1
    )
    dist = recent_with_label["label"].value_counts().reindex(LABEL_CLASSES, fill_value=0)
    st.subheader("Distribuicao na janela recente")
    st.bar_chart(dist)


# --------------------------- Clusters --------------------------------------

with tab_clusters:
    st.subheader("Perfis operacionais (K-Means + PCA)")
    summary = _load_summary()
    if summary and summary.get("clustering"):
        clu = summary["clustering"]
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("k (KMeans)", clu["kmeans"]["best_k"])
        col_b.metric("Silhouette KMeans", f"{clu['kmeans']['silhouette']:.3f}")
        col_c.metric("Variancia PCA(2)", f"{clu['pca']['explained_variance_ratio_cumulative']:.2%}")

        st.markdown("### Perfis identificados (KMeans)")
        perfis_df = (
            pd.DataFrame(clu["kmeans"]["perfis"]).T.reset_index().rename(columns={"index": "cluster"})
        )
        st.dataframe(perfis_df, width="stretch", hide_index=True)

        st.markdown("### Visualizacoes")
        col1, col2 = st.columns(2)
        kmeans_img = SETTINGS.figures_dir / "cluster_pca_kmeans.png"
        dbscan_img = SETTINGS.figures_dir / "cluster_pca_dbscan.png"
        if kmeans_img.exists():
            col1.image(str(kmeans_img), caption="PCA + KMeans", width="stretch")
        if dbscan_img.exists():
            col2.image(str(dbscan_img), caption="PCA + DBSCAN", width="stretch")

        col3, col4 = st.columns(2)
        elbow = SETTINGS.figures_dir / "cluster_elbow.png"
        sil = SETTINGS.figures_dir / "cluster_silhouette.png"
        if elbow.exists():
            col3.image(str(elbow), caption="Elbow", width="stretch")
        if sil.exists():
            col4.image(str(sil), caption="Silhouette", width="stretch")
    else:
        st.info(
            "Clusterizacao ainda nao foi executada. "
            "Rode `python -m app.training.train_clustering`."
        )

    # Cluster do snapshot atual em tempo real
    scaler = _load_model("clustering_scaler.joblib")
    kmeans = _load_model("clustering_kmeans.joblib")
    pca = _load_model("clustering_pca.joblib")
    if all(m is not None for m in (scaler, kmeans, pca)):
        cluster_features = (
            "cpu_percent",
            "memory_percent",
            "disk_percent",
            "net_bytes_sent_rate",
            "net_bytes_recv_rate",
            "load_1m",
        )
        processed = _load_processed()
        if not processed.empty and all(c in processed.columns for c in cluster_features):
            x_now = processed[list(cluster_features)].iloc[[-1]].to_numpy()
            x_scaled = scaler.transform(x_now)
            current_cluster = int(kmeans.predict(x_scaled)[0])
            st.success(f"Cluster atual: **cluster_{current_cluster}**")


# --------------------------- Metricas dos modelos --------------------------

with tab_metrics:
    summary = _load_summary()
    if not summary:
        st.info("Sem `reports/metrics/summary.json`. Rode `python -m app.training.evaluate_models`.")
    else:
        if summary.get("regression"):
            st.subheader("Regressao")
            st.dataframe(pd.DataFrame(summary["regression"]["results"]).T.round(4), width="stretch")
        if summary.get("classification"):
            st.subheader("Classificacao")
            rows = {name: payload["metrics"] for name, payload in summary["classification"]["results"].items()}
            st.dataframe(pd.DataFrame(rows).T.round(4), width="stretch")
        if summary.get("clustering"):
            st.subheader("Clusterizacao")
            clu = summary["clustering"]
            st.json(
                {
                    "kmeans_best_k": clu["kmeans"]["best_k"],
                    "kmeans_silhouette": clu["kmeans"]["silhouette"],
                    "dbscan_n_clusters": clu["dbscan"]["n_clusters"],
                    "dbscan_n_noise": clu["dbscan"]["n_noise"],
                    "pca_explained": clu["pca"]["explained_variance_ratio_cumulative"],
                }
            )

        md_path: Path = SETTINGS.metrics_dir / "summary.md"
        if md_path.exists():
            st.subheader("Resumo (markdown)")
            st.markdown(md_path.read_text(encoding="utf-8"))
