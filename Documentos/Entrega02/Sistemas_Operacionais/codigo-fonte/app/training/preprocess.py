"""Limpeza, engenharia de atributos e geracao de rotulos.

Le todos os CSVs em data/raw/, gera dataset analitico em data/processed/.

Execucao: `python -m app.training.preprocess`
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from app.common.config import Settings, get_settings
from app.common.logging import get_logger
from app.common.paths import ensure_dir
from app.common.storage import CsvStorage

logger = get_logger(__name__)


# Colunas usadas como features pelos modelos supervisionados.
FEATURE_COLUMNS: tuple[str, ...] = (
    "cpu_percent",
    "memory_percent",
    "memory_available_mb",
    "disk_percent",
    "cpu_ma_1m",
    "cpu_ma_5m",
    "cpu_delta",
    "mem_delta",
    "net_bytes_sent_rate",
    "net_bytes_recv_rate",
    "disk_read_rate",
    "disk_write_rate",
    "hour",
    "minute_of_hour",
    "load_1m",
)

LABEL_CLASSES: tuple[str, ...] = ("normal", "atencao", "critico")


def _label_row(cpu: float, mem: float, settings: Settings) -> str:
    if cpu > settings.thresh_critical_cpu or mem > settings.thresh_critical_mem:
        return "critico"
    if cpu < settings.thresh_normal_cpu and mem < settings.thresh_normal_mem:
        return "normal"
    return "atencao"


def build_features(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Aplica limpeza, ordena por timestamp, gera features derivadas e labels."""
    if df.empty:
        return df.copy()

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Conversao numerica defensiva
    numeric_cols = [
        "cpu_percent", "memory_percent", "memory_available_mb", "disk_percent",
        "disk_read_bytes", "disk_write_bytes", "net_bytes_sent", "net_bytes_recv",
        "load_1m", "load_5m", "load_15m",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Estimativa do intervalo de coleta (segundos) para calcular taxas
    if len(df) >= 2:
        deltas = df["timestamp"].diff().dt.total_seconds().dropna()
        sample_seconds = float(deltas.median()) if not deltas.empty else 5.0
    else:
        sample_seconds = 5.0
    sample_seconds = max(0.5, sample_seconds)

    # Janelas (em pontos) para medias moveis ~1min e ~5min
    window_1m = max(1, int(round(60 / sample_seconds)))
    window_5m = max(window_1m, int(round(300 / sample_seconds)))

    df["cpu_ma_1m"] = df["cpu_percent"].rolling(window_1m, min_periods=1).mean()
    df["cpu_ma_5m"] = df["cpu_percent"].rolling(window_5m, min_periods=1).mean()
    df["cpu_delta"] = df["cpu_percent"].diff().fillna(0.0)
    df["mem_delta"] = df["memory_percent"].diff().fillna(0.0)

    # Taxas (bytes por segundo) a partir dos contadores cumulativos
    for col, out in (
        ("net_bytes_sent", "net_bytes_sent_rate"),
        ("net_bytes_recv", "net_bytes_recv_rate"),
        ("disk_read_bytes", "disk_read_rate"),
        ("disk_write_bytes", "disk_write_rate"),
    ):
        if col in df.columns:
            delta = df[col].diff().fillna(0.0)
            # Negativos podem aparecer em reset/overflow de contador - clipamos a zero
            df[out] = (delta.clip(lower=0.0) / sample_seconds).astype(float)
        else:
            df[out] = 0.0

    df["hour"] = df["timestamp"].dt.hour.astype(int)
    df["minute_of_hour"] = df["timestamp"].dt.minute.astype(int)

    # Targets de regressao
    df["cpu_percent_t+1"] = df["cpu_percent"].shift(-1)
    df["cpu_next_60s_mean"] = (
        df["cpu_percent"].rolling(window_1m, min_periods=1).mean().shift(-window_1m)
    )

    # Rotulo de classificacao
    df["risk_label"] = df.apply(
        lambda row: _label_row(row["cpu_percent"], row["memory_percent"], settings),
        axis=1,
    )
    df["risk_label_idx"] = df["risk_label"].map({c: i for i, c in enumerate(LABEL_CLASSES)})

    # Remove ultimas linhas sem target valido (devido ao shift)
    df = df.dropna(subset=["cpu_percent_t+1"]).reset_index(drop=True)

    # Sanity checks finais
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=list(FEATURE_COLUMNS))
    return df


def run(output_dir: Path | None = None) -> Path:
    settings = get_settings()
    storage = CsvStorage(settings.raw_dir)
    raw = storage.read_all()
    logger.info("dataset bruto carregado", extra={"linhas": len(raw)})
    if raw.empty:
        raise SystemExit("Nenhum CSV em data/raw/. Rode o coletor ou seed_synthetic primeiro.")

    df = build_features(raw, settings)
    logger.info(
        "dataset processado pronto",
        extra={
            "linhas": len(df),
            "atencao": int((df["risk_label"] == "atencao").sum()),
            "critico": int((df["risk_label"] == "critico").sum()),
            "normal": int((df["risk_label"] == "normal").sum()),
        },
    )

    out_dir = ensure_dir(output_dir or settings.processed_dir)
    parquet_path = out_dir / "dataset.parquet"
    csv_path = out_dir / "dataset.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    logger.info(
        "artefatos salvos",
        extra={"parquet": str(parquet_path), "csv": str(csv_path)},
    )
    return parquet_path


def load_processed() -> pd.DataFrame:
    settings = get_settings()
    parquet = settings.processed_dir / "dataset.parquet"
    if not parquet.exists():
        raise FileNotFoundError(
            f"Dataset processado nao encontrado em {parquet}. "
            "Rode `python -m app.training.preprocess` antes."
        )
    return pd.read_parquet(parquet)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de pre-processamento")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    run(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
