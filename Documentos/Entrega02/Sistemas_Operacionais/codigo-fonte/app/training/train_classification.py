"""Treina classificadores para o estado da instancia (normal/atencao/critico).

Modelos: LogisticRegression, KNN, DecisionTree, RandomForest.

Saidas:
- data/models/classification_<modelo>.joblib
- reports/metrics/classification.json
- reports/figures/classification_matriz_confusao_<modelo>.png
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.common.config import get_settings
from app.common.logging import get_logger
from app.common.paths import ensure_dir
from app.training.preprocess import FEATURE_COLUMNS, LABEL_CLASSES, load_processed

logger = get_logger(__name__)

TARGET_COL = "risk_label"
RANDOM_STATE = 42
CV_FOLDS = 5


def _temporal_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def _build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)),
            ]
        ),
        "decision_tree": Pipeline(
            [
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=10, min_samples_leaf=8, random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=14,
                        n_jobs=-1,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def _eval_classifier(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _plot_confusion(name: str, cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
    ax.set_title(f"Matriz de confusao - {name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run() -> dict[str, dict]:
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
    label_order = list(LABEL_CLASSES)
    distrib = pd.Series(df[TARGET_COL]).value_counts(normalize=True).round(3).to_dict()
    logger.info(
        "split temporal pronto",
        extra={"n_train": len(train), "n_test": len(test), "distrib": distrib},
    )

    models = _build_models()
    figures_dir = ensure_dir(settings.figures_dir)
    metrics_dir = ensure_dir(settings.metrics_dir)
    ensure_dir(settings.model_dir)

    results: dict[str, dict] = {}
    for name, pipe in models.items():
        logger.info("treinando classificador", extra={"modelo": name})
        # CV estratificado no conjunto de treino
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        try:
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
            cv_mean = float(cv_scores.mean())
            cv_std = float(cv_scores.std())
        except Exception as exc:  # noqa: BLE001
            logger.warning("CV falhou; seguindo com fit direto", extra={"modelo": name, "erro": str(exc)})
            cv_mean = float("nan")
            cv_std = float("nan")

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics = _eval_classifier(y_test, y_pred, label_order)
        metrics["cv_f1_macro_mean"] = cv_mean
        metrics["cv_f1_macro_std"] = cv_std

        cm = confusion_matrix(y_test, y_pred, labels=label_order)
        report = classification_report(
            y_test, y_pred, labels=label_order, zero_division=0, output_dict=True
        )
        results[name] = {"metrics": metrics, "report": report, "confusion_matrix": cm.tolist()}

        joblib.dump(pipe, settings.model_dir / f"classification_{name}.joblib")
        _plot_confusion(name, cm, label_order, figures_dir / f"classification_matriz_confusao_{name}.png")
        logger.info("classificador treinado", extra={"modelo": name, **metrics})

    best_name = max(results, key=lambda k: results[k]["metrics"]["f1_macro"])
    joblib.dump(models[best_name], settings.model_dir / "classification_best.joblib")
    logger.info("melhor classificador", extra={"modelo": best_name, **results[best_name]["metrics"]})

    summary = {
        "task": "classification",
        "target": TARGET_COL,
        "features": list(FEATURE_COLUMNS),
        "labels": label_order,
        "n_train": len(train),
        "n_test": len(test),
        "best_model": best_name,
        "class_distribution": distrib,
        "results": results,
    }
    (metrics_dir / "classification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("classificacao concluida", extra={"summary": str(metrics_dir / 'classification.json')})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino de classificacao do estado da instancia")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
