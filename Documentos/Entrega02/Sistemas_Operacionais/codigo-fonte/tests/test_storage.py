from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.common.storage import CsvStorage


def test_csv_storage_round_trip(tmp_path):
    storage = CsvStorage(tmp_path / "raw")
    base = datetime(2026, 5, 8, 12, 0, 0, tzinfo=UTC)
    records = [
        {
            "timestamp": base + timedelta(seconds=5 * i),
            "cpu_percent": 10.0 + i,
            "memory_percent": 30.0 + i,
            "memory_available_mb": 1024.0,
            "disk_percent": 50.0,
            "disk_read_bytes": 1000 * i,
            "disk_write_bytes": 500 * i,
            "net_bytes_sent": 200 * i,
            "net_bytes_recv": 300 * i,
            "load_1m": 0.5,
            "load_5m": 0.4,
            "load_15m": 0.3,
        }
        for i in range(3)
    ]
    storage.append_many(records)
    df = storage.read_all()
    assert len(df) == 3
    assert df["cpu_percent"].tolist() == [10.0, 11.0, 12.0]
    assert df["timestamp"].is_monotonic_increasing


def test_csv_storage_rotates_per_day(tmp_path):
    storage = CsvStorage(tmp_path / "raw")
    storage.append_many(
        [
            {"timestamp": datetime(2026, 5, 8, 23, 59, 0, tzinfo=UTC), "cpu_percent": 1.0},
            {"timestamp": datetime(2026, 5, 9, 0, 0, 30, tzinfo=UTC), "cpu_percent": 2.0},
        ]
    )
    files = storage.list_files()
    assert len(files) == 2
    assert {f.name for f in files} == {"metrics_20260508.csv", "metrics_20260509.csv"}
