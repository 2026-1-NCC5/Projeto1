"""Gera um dataset sintetico plausivel (~12h) para destravar o treino.

Util quando voce quer rodar a pipeline de IA sem esperar a coleta real.

Execucao: `python scripts/seed_synthetic.py --hours 12 --interval 5`
"""

from __future__ import annotations

import argparse
import math
import random
from datetime import UTC, datetime, timedelta

import numpy as np

from app.common.config import get_settings
from app.common.logging import get_logger
from app.common.storage import CsvStorage

logger = get_logger(__name__)


def generate(hours: float, interval_seconds: float, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    n = int((hours * 3600) / interval_seconds)
    start = datetime.now(tz=UTC) - timedelta(hours=hours)

    # Padroes diarios + bursts aleatorios para forcar normal/atencao/critico.
    records: list[dict] = []
    net_sent = 1_000_000_000
    net_recv = 2_000_000_000
    disk_read = 500_000_000
    disk_write = 300_000_000
    burst_until: datetime | None = None
    burst_cpu = 0.0
    burst_mem = 0.0

    for i in range(n):
        ts = start + timedelta(seconds=i * interval_seconds)
        # Componente diurna (senoide com periodo 24h)
        seconds_of_day = ts.hour * 3600 + ts.minute * 60 + ts.second
        diurnal = 0.5 + 0.5 * math.sin((seconds_of_day / 86400.0) * 2 * math.pi - math.pi / 2)

        # Iniciar / encerrar bursts
        if burst_until is None or ts > burst_until:
            if rng.random() < 0.02:  # ~2% de chance por amostra
                burst_until = ts + timedelta(seconds=rng.randint(60, 600))
                burst_cpu = rng.uniform(35.0, 60.0)
                burst_mem = rng.uniform(15.0, 35.0)
            else:
                burst_until = None
                burst_cpu = 0.0
                burst_mem = 0.0

        cpu_base = 18 + 22 * diurnal + np_rng.normal(0, 4) + burst_cpu
        mem_base = 38 + 18 * diurnal + np_rng.normal(0, 3) + burst_mem
        cpu = float(np.clip(cpu_base, 1.0, 99.5))
        mem = float(np.clip(mem_base, 10.0, 98.0))

        disk = float(np.clip(45 + 5 * diurnal + np_rng.normal(0, 1.5), 10.0, 95.0))

        # Contadores cumulativos crescentes (com jitter)
        delta_disk_read = max(0, int(np_rng.normal(2_000_000, 800_000)))
        delta_disk_write = max(0, int(np_rng.normal(1_500_000, 600_000)))
        delta_net_sent = max(0, int(np_rng.normal(150_000, 80_000) + (5_000_000 if burst_until else 0)))
        delta_net_recv = max(0, int(np_rng.normal(220_000, 110_000) + (8_000_000 if burst_until else 0)))
        disk_read += delta_disk_read
        disk_write += delta_disk_write
        net_sent += delta_net_sent
        net_recv += delta_net_recv

        records.append(
            {
                "timestamp": ts,
                "cpu_percent": round(cpu, 2),
                "memory_percent": round(mem, 2),
                "memory_available_mb": round(max(64.0, 4096.0 * (1 - mem / 100)), 2),
                "disk_percent": round(disk, 2),
                "disk_read_bytes": disk_read,
                "disk_write_bytes": disk_write,
                "net_bytes_sent": net_sent,
                "net_bytes_recv": net_recv,
                "load_1m": round(max(0.0, cpu / 25 + np_rng.normal(0, 0.1)), 3),
                "load_5m": round(max(0.0, cpu / 30 + np_rng.normal(0, 0.08)), 3),
                "load_15m": round(max(0.0, cpu / 35 + np_rng.normal(0, 0.06)), 3),
            }
        )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera dataset sintetico para treino rapido")
    parser.add_argument("--hours", type=float, default=12.0)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    settings = get_settings()
    storage = CsvStorage(settings.raw_dir)

    logger.info(
        "gerando dataset sintetico",
        extra={"hours": args.hours, "interval_s": args.interval, "raw_dir": str(storage.raw_dir)},
    )
    records = generate(args.hours, args.interval, seed=args.seed)
    storage.append_many(records)
    logger.info("dataset sintetico salvo", extra={"linhas": len(records)})


if __name__ == "__main__":
    main()
