"""Consolida os JSONs de regression / classification / clustering em um summary.

Saidas:
- reports/metrics/summary.json
- reports/metrics/summary.md (pronto para colar no relatorio academico)
"""

from __future__ import annotations

import json
from pathlib import Path

from app.common.config import get_settings
from app.common.logging import get_logger
from app.common.paths import ensure_dir

logger = get_logger(__name__)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        logger.warning("arquivo nao encontrado", extra={"path": str(path)})
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _md_table_regression(reg: dict) -> str:
    lines = ["| Modelo | MAE | MSE | RMSE | R2 |", "|---|---:|---:|---:|---:|"]
    for name, metrics in reg["results"].items():
        flag = "**" if name == reg["best_model"] else ""
        lines.append(
            f"| {flag}{name}{flag} | {metrics['mae']:.3f} | {metrics['mse']:.3f} | "
            f"{metrics['rmse']:.3f} | {metrics['r2']:.3f} |"
        )
    return "\n".join(lines)


def _md_table_classification(cls: dict) -> str:
    lines = [
        "| Modelo | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) | CV F1 (mean) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, payload in cls["results"].items():
        m = payload["metrics"]
        flag = "**" if name == cls["best_model"] else ""
        cv_mean = m.get("cv_f1_macro_mean")
        cv_str = f"{cv_mean:.3f}" if isinstance(cv_mean, (int, float)) and cv_mean == cv_mean else "n/a"
        lines.append(
            f"| {flag}{name}{flag} | {m['accuracy']:.3f} | {m['precision_macro']:.3f} | "
            f"{m['recall_macro']:.3f} | {m['f1_macro']:.3f} | {m['f1_weighted']:.3f} | {cv_str} |"
        )
    return "\n".join(lines)


def _md_clustering(clu: dict) -> str:
    km = clu["kmeans"]
    db = clu["dbscan"]
    pca = clu["pca"]
    lines = [
        f"- **K-Means**: melhor k = {km['best_k']}, silhouette = {km['silhouette']:.3f}",
        f"- **DBSCAN**: eps = {db['eps']:.3f}, min_samples = {db['min_samples']}, "
        f"clusters = {db['n_clusters']}, ruido = {db['n_noise']}",
        f"- **PCA(2)**: variancia explicada = "
        f"{pca['explained_variance_ratio'][0]:.3f} + {pca['explained_variance_ratio'][1]:.3f} "
        f"= {pca['explained_variance_ratio_cumulative']:.3f}",
        "",
        "### Perfis K-Means (medias por cluster)",
        "",
        "| Cluster | Tamanho | Fracao | " + " | ".join(km["perfis"][next(iter(km["perfis"]))]["medias"].keys()) + " |",
        "|---|---:|---:|" + "|".join(["---:"] * len(km["perfis"][next(iter(km["perfis"]))]["medias"])) + "|",
    ]
    for name, perfil in km["perfis"].items():
        medias = perfil["medias"]
        lines.append(
            f"| {name} | {perfil['tamanho']} | {perfil['fracao']:.3f} | "
            + " | ".join(f"{v:.2f}" for v in medias.values())
            + " |"
        )
    return "\n".join(lines)


def run() -> Path:
    settings = get_settings()
    metrics_dir = ensure_dir(settings.metrics_dir)
    reg = _load_json(metrics_dir / "regression.json")
    cls = _load_json(metrics_dir / "classification.json")
    clu = _load_json(metrics_dir / "clustering.json")

    summary = {"regression": reg, "classification": cls, "clustering": clu}
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    parts = ["# AbraceAI Analytics - Resumo dos Modelos\n"]
    if reg:
        parts.append("## Regressao (CPU futura)\n")
        parts.append(f"Target: `{reg['target']}` | Treino: {reg['n_train']} | Teste: {reg['n_test']} | "
                     f"Melhor modelo: **{reg['best_model']}**\n")
        parts.append(_md_table_regression(reg))
        parts.append("")
        parts.append("Figuras: `reports/figures/regression_real_vs_previsto.png`, `regression_residuos.png`.\n")
    if cls:
        parts.append("## Classificacao (estado da instancia)\n")
        parts.append(
            f"Target: `{cls['target']}` | Treino: {cls['n_train']} | Teste: {cls['n_test']} | "
            f"Melhor modelo: **{cls['best_model']}**\n"
        )
        parts.append(f"Distribuicao das classes: `{cls['class_distribution']}`\n")
        parts.append(_md_table_classification(cls))
        parts.append("")
        parts.append("Figuras: `reports/figures/classification_matriz_confusao_<modelo>.png`.\n")
    if clu:
        parts.append("## Clusterizacao + PCA\n")
        parts.append(_md_clustering(clu))
        parts.append("")
        parts.append("Figuras: `cluster_elbow.png`, `cluster_silhouette.png`, "
                     "`cluster_pca_kmeans.png`, `cluster_pca_dbscan.png`, "
                     "`cluster_pca_variancia_explicada.png`.\n")

    md_path = metrics_dir / "summary.md"
    md_path.write_text("\n".join(parts), encoding="utf-8")
    logger.info("summary consolidado", extra={"json": str(metrics_dir / 'summary.json'), "md": str(md_path)})
    return md_path


def main() -> None:
    run()


if __name__ == "__main__":
    main()
