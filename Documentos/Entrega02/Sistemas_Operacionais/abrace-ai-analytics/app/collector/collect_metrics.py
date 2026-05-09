"""Coletor periodico de metricas do host (CPU, memoria, disco, rede, load).

Execucao: `python -m app.collector.collect_metrics`
"""

from __future__ import annotations

import argparse
import signal
import time
from datetime import UTC, datetime
from typing import Any

import psutil

from app.common.config import get_settings
from app.common.logging import get_logger
from app.common.storage import CsvStorage

logger = get_logger(__name__)


def _safe_loadavg() -> tuple[float | None, float | None, float | None]:
    """getloadavg nao existe no Windows; retorna None nesses casos."""
    try:
        l1, l5, l15 = psutil.getloadavg()
        return float(l1), float(l5), float(l15)
    except (AttributeError, OSError):
        return None, None, None


def collect_once() -> dict[str, Any]:
    """Coleta um snapshot pontual de metricas do host."""
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    disk_io = psutil.disk_io_counters()
    net = psutil.net_io_counters()
    l1, l5, l15 = _safe_loadavg()

    return {
        "timestamp": datetime.now(tz=UTC),
        "cpu_percent": float(cpu),
        "memory_percent": float(mem.percent),
        "memory_available_mb": round(mem.available / (1024 * 1024), 2),
        "disk_percent": float(disk.percent),
        "disk_read_bytes": int(disk_io.read_bytes) if disk_io else 0,
        "disk_write_bytes": int(disk_io.write_bytes) if disk_io else 0,
        "net_bytes_sent": int(net.bytes_sent) if net else 0,
        "net_bytes_recv": int(net.bytes_recv) if net else 0,
        "load_1m": l1,
        "load_5m": l5,
        "load_15m": l15,
    }


class _Stopper:
    """Captura SIGINT/SIGTERM para encerrar o loop graciosamente."""

    def __init__(self) -> None:
        self.should_stop = False
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum: int, _frame: object) -> None:  # noqa: D401
        logger.info("sinal recebido, encerrando coletor", extra={"signal": signum})
        self.should_stop = True


def run(interval_seconds: float | None = None, max_iterations: int | None = None) -> int:
    """Loop principal do coletor.

    `max_iterations` permite uso em testes / scripts curtos. Retorna numero
    de coletas realizadas.
    """
    settings = get_settings()
    storage = CsvStorage(settings.raw_dir)
    interval = float(interval_seconds if interval_seconds is not None else settings.collection_interval_seconds)

    # Primeira chamada de cpu_percent serve so para "armar" o calculo.
    psutil.cpu_percent(interval=None)
    time.sleep(0.1)

    stopper = _Stopper()
    iterations = 0
    logger.info(
        "coletor iniciado",
        extra={"interval_seconds": interval, "raw_dir": str(storage.raw_dir)},
    )

    while not stopper.should_stop:
        start = time.monotonic()
        try:
            record = collect_once()
            storage.append(record)
            iterations += 1
            logger.info(
                "coletado",
                extra={
                    "n": iterations,
                    "cpu": record["cpu_percent"],
                    "mem": record["memory_percent"],
                    "disk": record["disk_percent"],
                },
            )
        except Exception:  # noqa: BLE001 - logamos e seguimos
            logger.exception("falha ao coletar metricas")

        if max_iterations is not None and iterations >= max_iterations:
            break

        elapsed = time.monotonic() - start
        sleep_for = max(0.0, interval - elapsed)
        # Sleep em pequenos pedacos para responder a sinais rapidamente
        deadline = time.monotonic() + sleep_for
        while not stopper.should_stop and time.monotonic() < deadline:
            time.sleep(min(0.5, deadline - time.monotonic()))

    logger.info("coletor finalizado", extra={"total_iteracoes": iterations})
    return iterations


def main() -> None:
    parser = argparse.ArgumentParser(description="Coletor de metricas do host")
    parser.add_argument("--interval", type=float, default=None, help="Intervalo entre coletas em segundos")
    parser.add_argument("--max", type=int, default=None, help="Numero maximo de coletas (default: infinito)")
    args = parser.parse_args()
    run(interval_seconds=args.interval, max_iterations=args.max)


if __name__ == "__main__":
    main()
