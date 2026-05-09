from __future__ import annotations

from app.collector.collect_metrics import collect_once, run
from app.common.config import get_settings
from app.common.storage import CsvStorage


def test_collect_once_has_required_fields():
    rec = collect_once()
    for key in (
        "timestamp",
        "cpu_percent",
        "memory_percent",
        "memory_available_mb",
        "disk_percent",
        "disk_read_bytes",
        "disk_write_bytes",
        "net_bytes_sent",
        "net_bytes_recv",
    ):
        assert key in rec, f"campo ausente: {key}"
    assert 0.0 <= rec["cpu_percent"] <= 100.0
    assert 0.0 <= rec["memory_percent"] <= 100.0


def test_collector_runs_a_few_iterations(isolated_dirs):
    iterations = run(interval_seconds=0.1, max_iterations=3)
    assert iterations == 3
    settings = get_settings()
    df = CsvStorage(settings.raw_dir).read_all()
    assert len(df) == 3
