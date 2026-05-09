from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from app.common.config import get_settings
from app.training.preprocess import FEATURE_COLUMNS, LABEL_CLASSES, build_features


def _sample_raw(n: int = 200) -> pd.DataFrame:
    base = datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC)
    rows = []
    for i in range(n):
        cpu = 20.0 + (i % 80)
        mem = 30.0 + (i % 60)
        rows.append(
            {
                "timestamp": base + timedelta(seconds=5 * i),
                "cpu_percent": float(cpu),
                "memory_percent": float(mem),
                "memory_available_mb": 4096.0 - mem * 30,
                "disk_percent": 50.0 + (i % 5),
                "disk_read_bytes": 1_000_000 + 1000 * i,
                "disk_write_bytes": 800_000 + 800 * i,
                "net_bytes_sent": 500_000 + 200 * i,
                "net_bytes_recv": 700_000 + 300 * i,
                "load_1m": 0.5 + (i % 3) * 0.1,
                "load_5m": 0.4,
                "load_15m": 0.3,
            }
        )
    return pd.DataFrame(rows)


def test_build_features_shapes_and_labels(isolated_dirs):
    raw = _sample_raw()
    settings = get_settings()
    out = build_features(raw, settings)

    for col in FEATURE_COLUMNS:
        assert col in out.columns, f"feature ausente: {col}"
    assert "cpu_percent_t+1" in out.columns
    assert "risk_label" in out.columns
    assert set(out["risk_label"].unique()).issubset(set(LABEL_CLASSES))
    assert out["risk_label_idx"].between(0, 2).all()
    assert not out[list(FEATURE_COLUMNS)].isna().any().any()


def test_build_features_label_thresholds(isolated_dirs):
    settings = get_settings()
    raw = pd.DataFrame(
        [
            {
                "timestamp": datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC),
                "cpu_percent": 10.0, "memory_percent": 20.0, "memory_available_mb": 3000.0,
                "disk_percent": 30.0, "disk_read_bytes": 0, "disk_write_bytes": 0,
                "net_bytes_sent": 0, "net_bytes_recv": 0,
                "load_1m": 0.1, "load_5m": 0.1, "load_15m": 0.1,
            },
            {
                "timestamp": datetime(2026, 5, 8, 12, 0, 5, tzinfo=UTC),
                "cpu_percent": 65.0, "memory_percent": 50.0, "memory_available_mb": 1500.0,
                "disk_percent": 50.0, "disk_read_bytes": 100, "disk_write_bytes": 50,
                "net_bytes_sent": 100, "net_bytes_recv": 200,
                "load_1m": 1.0, "load_5m": 0.9, "load_15m": 0.8,
            },
            {
                "timestamp": datetime(2026, 5, 8, 12, 0, 10, tzinfo=UTC),
                "cpu_percent": 95.0, "memory_percent": 88.0, "memory_available_mb": 200.0,
                "disk_percent": 70.0, "disk_read_bytes": 500, "disk_write_bytes": 300,
                "net_bytes_sent": 1000, "net_bytes_recv": 2000,
                "load_1m": 4.0, "load_5m": 3.5, "load_15m": 3.0,
            },
        ]
    )
    out = build_features(raw, settings)
    # Apos o shift(-1) a ultima linha some, sobram duas linhas com normal e atencao.
    labels = out["risk_label"].tolist()
    assert labels[0] == "normal"
    assert labels[1] == "atencao"
