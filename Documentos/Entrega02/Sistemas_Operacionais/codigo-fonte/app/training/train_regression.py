"""Treina modelos de regressao para prever CPU futura.

Modelos: Linear, Polinomial(deg=2)+Ridge, RandomForestRegressor.

Saidas:
- data/models/regression_<modelo>.joblib
- reports/metrics/regression.json
- reports/figures/regression_real_vs_previsto.png
- reports/figures/regression_residuos.png
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from app.common.config import get_settings
from app.common.logging import get_logger
from app.common.paths import ensure_dir
from app.training.preprocess import FEATURE_COLUMNS, load_processed

logger = get_logger(__name__)

TARGET_COL = "cpu_percent_t+1"
RANDOM_STATE = 42


def _temporal_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split por tempo (sem shuffle) para evitar data leakage em serie temporal."""
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def _build_models() -> dict[str, Pipeline]:
    return {
        "linear": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "polinomial_deg2_ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)),
                ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=120,
                        max_depth=14,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _plot_real_vs_pred(
    results: dict[str, dict], y_test: np.ndarray, predictions: dict[str, np.ndarray], out_path: Path
) -> None:
    fig, axes = plt.subplots(1, len(predictions), figsize=(5 * len(predictions), 4), sharey=True)
    if len(predictions) == 1:
        axes = [axes]
    for ax, (name, y_pred) in zip(axes, predictions.items(), strict=True):
        ax.scatter(y_test, y_pred, alpha=0.4, s=12, edgecolor="none")
        lo, hi = float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))
        ax.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=1.0)
        ax.set_title(f"{name}\nR2={results[name]['r2']:.3f}  MAE={results[name]['mae']:.2f}")
        ax.set_xlabel("CPU real (%)")
    axes[0].set_ylabel("CPU prevista (%)")
    fig.suptitle("Regressao - real vs previsto (CPU t+1)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_residuals(predictions: dict[str, np.ndarray], y_test: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, y_pred in predictions.items():
        residuos = y_pred - y_test
        ax.hist(residuos, bins=40, alpha=0.5, label=name)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Distribuicao dos residuos (previsto - real)")
    ax.set_xlabel("Residuo")
    ax.set_ylabel("Frequencia")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run() -> dict[str, dict[str, float]]:
    settings = get_settings()
    df = load_processed()
    if TARGET_COL not in df.columns:
        raise SystemExit(f"Coluna alvo {TARGET_COL} ausente; rode preprocess.")

    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    train, test = _temporal_split(df, test_frac=0.2)

    X_train = train[list(FEATURE_COLUMNS)].to_numpy()
    y_train = train[TARGET_COL].to_numpy()
    X_test = test[list(FEATURE_COLUMNS)].to_numpy()
    y_test = test[TARGET_COL].to_numpy()
    logger.info(
        "split temporal pronto",
        extra={"n_train": len(train), "n_test": len(test), "features": len(FEATURE_COLUMNS)},
    )

    models = _build_models()
    results: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}

    ensure_dir(settings.model_dir)
    for name, pipe in models.items():
        logger.info("treinando regressor", extra={"modelo": name})
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics = _evaluate(y_test, y_pred)
        results[name] = metrics
        predictions[name] = y_pred
        joblib.dump(pipe, settings.model_dir / f"regression_{name}.joblib")
        logger.info("regressor treinado", extra={"modelo": name, **metrics})

    # Melhor modelo por R2
    best_name = max(results, key=lambda k: results[k]["r2"])
    joblib.dump(models[best_name], settings.model_dir / "regression_best.joblib")
    logger.info("melhor regressor", extra={"modelo": best_name, **results[best_name]})

    figures_dir = ensure_dir(settings.figures_dir)
    metrics_dir = ensure_dir(settings.metrics_dir)
    _plot_real_vs_pred(results, y_test, predictions, figures_dir / "regression_real_vs_previsto.png")
    _plot_residuals(predictions, y_test, figures_dir / "regression_residuos.png")

    summary = {
        "task": "regression",
        "target": TARGET_COL,
        "features": list(FEATURE_COLUMNS),
        "n_train": len(train),
        "n_test": len(test),
        "best_model": best_name,
        "results": results,
    }
    (metrics_dir / "regression.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("regressao concluida", extra={"summary": str(metrics_dir / 'regression.json')})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino de regressao para CPU futura")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
