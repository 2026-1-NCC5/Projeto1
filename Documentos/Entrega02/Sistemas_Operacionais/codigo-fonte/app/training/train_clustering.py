"""Clusterizacao de perfis operacionais (K-Means, DBSCAN) e PCA 2D para visualizacao.

Saidas:
- data/models/clustering_kmeans.joblib (k otimo escolhido por silhouette)
- data/models/clustering_dbscan.joblib
- data/models/clustering_scaler.joblib
- data/models/clustering_pca.joblib
- reports/figures/cluster_elbow.png
- reports/figures/cluster_silhouette.png
- reports/figures/cluster_pca_kmeans.png
- reports/figures/cluster_pca_dbscan.png
- reports/figures/cluster_pca_variancia_explicada.png
- reports/metrics/clustering.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from app.common.config import get_settings
from app.common.logging import get_logger
from app.common.paths import ensure_dir
from app.training.preprocess import load_processed

logger = get_logger(__name__)

RANDOM_STATE = 42
KMEANS_K_RANGE = range(2, 7)
# Subconjunto interpretavel para descricao operacional dos clusters
CLUSTER_FEATURES: tuple[str, ...] = (
    "cpu_percent",
    "memory_percent",
    "disk_percent",
    "net_bytes_sent_rate",
    "net_bytes_recv_rate",
    "load_1m",
)


def _kmeans_sweep(X: np.ndarray) -> tuple[dict[int, dict[str, float]], int]:
    metrics: dict[int, dict[str, float]] = {}
    for k in KMEANS_K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        try:
            sil = float(silhouette_score(X, labels))
        except Exception:  # noqa: BLE001
            sil = float("nan")
        metrics[k] = {"inertia": float(km.inertia_), "silhouette": sil}
    best_k = max(metrics, key=lambda k: metrics[k]["silhouette"])
    return metrics, best_k


def _plot_elbow(metrics: dict[int, dict[str, float]], out_path: Path) -> None:
    ks = sorted(metrics)
    inertias = [metrics[k]["inertia"] for k in ks]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, inertias, marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    ax.set_title("KMeans - metodo do cotovelo")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_silhouette(metrics: dict[int, dict[str, float]], best_k: int, out_path: Path) -> None:
    ks = sorted(metrics)
    sils = [metrics[k]["silhouette"] for k in ks]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(ks, sils, color="#4c72b0")
    bars[ks.index(best_k)].set_color("#dd8452")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette")
    ax.set_title(f"KMeans - silhouette por k (melhor={best_k})")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_pca_clusters(
    pca_xy: np.ndarray, labels: np.ndarray, title: str, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    unique = sorted(set(labels.tolist()))
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(max(len(unique), 3))
    for i, lab in enumerate(unique):
        mask = labels == lab
        color = "lightgrey" if lab == -1 else cmap(i)
        nome = "ruido" if lab == -1 else f"cluster {lab}"
        ax.scatter(pca_xy[mask, 0], pca_xy[mask, 1], s=10, alpha=0.7, label=nome, c=[color])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(markerscale=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_pca_variance(pca: PCA, out_path: Path) -> None:
    var = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, len(var) + 1), var, color="#4c72b0", alpha=0.8, label="por PC")
    ax.plot(range(1, len(var) + 1), np.cumsum(var), marker="o", color="#dd8452", label="cumulativo")
    ax.set_xlabel("Componente principal")
    ax.set_ylabel("Variancia explicada")
    ax.set_title("PCA - variancia explicada")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _interpret_clusters(df: pd.DataFrame, labels: np.ndarray) -> dict[str, dict]:
    work = df[list(CLUSTER_FEATURES)].copy()
    work["cluster"] = labels
    profiles: dict[str, dict] = {}
    for cluster_id, group in work.groupby("cluster"):
        nome = f"cluster_{int(cluster_id)}" if cluster_id != -1 else "ruido"
        profiles[nome] = {
            "tamanho": int(len(group)),
            "fracao": round(len(group) / len(work), 4),
            "medias": {col: round(float(group[col].mean()), 3) for col in CLUSTER_FEATURES},
        }
    return profiles


def _dbscan_eps_estimate(X: np.ndarray, k: int = 5) -> float:
    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(X)
    distances, _ = nn.kneighbors(X)
    kth = np.sort(distances[:, k - 1])
    # Heuristica: percentil 90 da k-distance
    return float(np.percentile(kth, 90))


def run() -> dict:
    settings = get_settings()
    df = load_processed()
    cols = list(CLUSTER_FEATURES)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Colunas ausentes para clusterizacao: {missing}")
    X_raw = df[cols].to_numpy()

    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw)
    logger.info("padronizado", extra={"n": len(X), "features": cols})

    # KMeans sweep + best k
    sweep, best_k = _kmeans_sweep(X)
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit(X)
    kmeans_labels = kmeans.labels_
    kmeans_sil = sweep[best_k]["silhouette"]
    logger.info("kmeans pronto", extra={"k": best_k, "silhouette": kmeans_sil})

    # DBSCAN com eps estimado
    eps = _dbscan_eps_estimate(X, k=5)
    dbscan = DBSCAN(eps=eps, min_samples=10, n_jobs=-1).fit(X)
    db_labels = dbscan.labels_
    n_clusters_db = int(len(set(db_labels)) - (1 if -1 in db_labels else 0))
    n_noise_db = int((db_labels == -1).sum())
    try:
        mask_no_noise = db_labels != -1
        db_sil = float(silhouette_score(X[mask_no_noise], db_labels[mask_no_noise])) if mask_no_noise.sum() > 1 and n_clusters_db > 1 else float("nan")
    except Exception:  # noqa: BLE001
        db_sil = float("nan")
    logger.info(
        "dbscan pronto",
        extra={"eps": round(eps, 4), "n_clusters": n_clusters_db, "n_noise": n_noise_db, "silhouette": db_sil},
    )

    # PCA 2D para visualizacao
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(X)
    pca_xy = pca.transform(X)

    figures_dir = ensure_dir(settings.figures_dir)
    metrics_dir = ensure_dir(settings.metrics_dir)
    ensure_dir(settings.model_dir)

    _plot_elbow(sweep, figures_dir / "cluster_elbow.png")
    _plot_silhouette(sweep, best_k, figures_dir / "cluster_silhouette.png")
    _plot_pca_clusters(
        pca_xy, kmeans_labels, f"PCA + KMeans (k={best_k})", figures_dir / "cluster_pca_kmeans.png"
    )
    _plot_pca_clusters(
        pca_xy, db_labels, f"PCA + DBSCAN (eps={eps:.3f})", figures_dir / "cluster_pca_dbscan.png"
    )
    _plot_pca_variance(pca, figures_dir / "cluster_pca_variancia_explicada.png")

    joblib.dump(kmeans, settings.model_dir / "clustering_kmeans.joblib")
    joblib.dump(dbscan, settings.model_dir / "clustering_dbscan.joblib")
    joblib.dump(scaler, settings.model_dir / "clustering_scaler.joblib")
    joblib.dump(pca, settings.model_dir / "clustering_pca.joblib")

    summary = {
        "task": "clustering",
        "features": cols,
        "n_samples": int(len(X)),
        "kmeans": {
            "best_k": best_k,
            "sweep": {str(k): v for k, v in sweep.items()},
            "silhouette": kmeans_sil,
            "perfis": _interpret_clusters(df, kmeans_labels),
        },
        "dbscan": {
            "eps": eps,
            "min_samples": 10,
            "n_clusters": n_clusters_db,
            "n_noise": n_noise_db,
            "silhouette": db_sil,
            "perfis": _interpret_clusters(df, db_labels),
        },
        "pca": {
            "n_components": 2,
            "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
            "explained_variance_ratio_cumulative": float(np.sum(pca.explained_variance_ratio_)),
        },
    }
    (metrics_dir / "clustering.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("clusterizacao concluida", extra={"summary": str(metrics_dir / 'clustering.json')})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Clusterizacao + PCA")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
