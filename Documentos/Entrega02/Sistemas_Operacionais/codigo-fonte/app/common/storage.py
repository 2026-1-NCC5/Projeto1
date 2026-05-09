"""Persistencia de metricas em CSV rotacionado por dia.

Interface intencionalmente pequena para permitir trocar por SQLite/Postgres
sem alterar o coletor.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Protocol

import pandas as pd

from app.common.logging import get_logger
from app.common.paths import ensure_dir

logger = get_logger(__name__)


# Schema canonico das metricas. Ordem afeta a saida CSV.
METRIC_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "cpu_percent",
    "memory_percent",
    "memory_available_mb",
    "disk_percent",
    "disk_read_bytes",
    "disk_write_bytes",
    "net_bytes_sent",
    "net_bytes_recv",
    "load_1m",
    "load_5m",
    "load_15m",
)


class Storage(Protocol):
    def append(self, record: Mapping[str, object]) -> None: ...
    def append_many(self, records: Iterable[Mapping[str, object]]) -> None: ...
    def read_all(self) -> pd.DataFrame: ...


class CsvStorage:
    """Storage simples baseado em CSVs rotacionados por dia (UTC)."""

    def __init__(self, raw_dir: Path, columns: tuple[str, ...] = METRIC_COLUMNS) -> None:
        self.raw_dir = ensure_dir(raw_dir)
        self.columns = columns
        self._lock = Lock()

    # -- escrita --------------------------------------------------------------

    def _file_for(self, ts: datetime) -> Path:
        ts_utc = ts.astimezone(UTC) if ts.tzinfo else ts.replace(tzinfo=UTC)
        return self.raw_dir / f"metrics_{ts_utc.strftime('%Y%m%d')}.csv"

    def _write_row(self, fp: Path, row: dict) -> None:
        write_header = not fp.exists()
        with fp.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.columns, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def append(self, record: Mapping[str, object]) -> None:
        ts = self._coerce_ts(record.get("timestamp"))
        row = dict(record)
        row["timestamp"] = ts.isoformat()
        with self._lock:
            self._write_row(self._file_for(ts), row)

    def append_many(self, records: Iterable[Mapping[str, object]]) -> None:
        # Agrupa por arquivo (dia) para minimizar abertura de file handles.
        by_file: dict[Path, list[dict]] = {}
        for record in records:
            ts = self._coerce_ts(record.get("timestamp"))
            row = dict(record)
            row["timestamp"] = ts.isoformat()
            by_file.setdefault(self._file_for(ts), []).append(row)

        with self._lock:
            for fp, rows in by_file.items():
                write_header = not fp.exists()
                with fp.open("a", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=self.columns, extrasaction="ignore")
                    if write_header:
                        writer.writeheader()
                    writer.writerows(rows)

    @staticmethod
    def _coerce_ts(value: object) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        return datetime.now(tz=UTC)

    # -- leitura --------------------------------------------------------------

    def list_files(self) -> list[Path]:
        return sorted(self.raw_dir.glob("metrics_*.csv"))

    def read_all(self) -> pd.DataFrame:
        files = self.list_files()
        if not files:
            return pd.DataFrame(columns=list(self.columns))
        frames = [pd.read_csv(fp) for fp in files]
        df = pd.concat(frames, ignore_index=True)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return df

    def read_recent(self, minutes: int) -> pd.DataFrame:
        df = self.read_all()
        if df.empty:
            return df
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=minutes)
        return df[df["timestamp"] >= cutoff].reset_index(drop=True)
