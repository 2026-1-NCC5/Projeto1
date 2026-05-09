"""Gerador controlado de carga para enriquecer o dataset com idle / cpu / mem / net.

Uso: `python -m app.collector.load_generator --pattern mixed --duration 600`
"""

from __future__ import annotations

import argparse
import math
import os
import random
import threading
import time
from collections.abc import Iterable

from app.common.logging import get_logger

logger = get_logger(__name__)


def _stress_cpu(stop: threading.Event, intensity: float = 0.8) -> None:
    """Loop que ocupa um core consumindo `intensity` da CPU (0.0..1.0)."""
    intensity = max(0.05, min(1.0, intensity))
    period = 0.05
    busy = period * intensity
    idle = period - busy
    while not stop.is_set():
        end = time.monotonic() + busy
        # Trabalho real para nao ser otimizado
        x = 0.0001
        while time.monotonic() < end:
            x = math.sqrt(x + 1.000003)
        if idle > 0:
            stop.wait(idle)


def _stress_memory(stop: threading.Event, target_mb: int = 200) -> None:
    """Aloca progressivamente um buffer de bytes ate `target_mb` e mantem."""
    block = bytearray(0)
    chunk = 10 * 1024 * 1024  # 10 MB
    while not stop.is_set() and len(block) < target_mb * 1024 * 1024:
        block.extend(b"\x00" * chunk)
        stop.wait(0.5)
    logger.info("memoria alocada", extra={"mb": len(block) // (1024 * 1024)})
    while not stop.is_set():
        stop.wait(1.0)
    del block


def _stress_net_localhost(stop: threading.Event) -> None:
    """Trafego artificial em localhost via socket (nao sai da maquina)."""
    import socket

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.listen(1)

    def _serve() -> None:
        try:
            conn, _addr = server.accept()
            with conn:
                while not stop.is_set():
                    data = conn.recv(65536)
                    if not data:
                        break
        except OSError:
            pass

    t = threading.Thread(target=_serve, daemon=True)
    t.start()

    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", port))
        payload = os.urandom(64 * 1024)
        while not stop.is_set():
            try:
                client.sendall(payload)
            except OSError:
                break
            stop.wait(0.05)
        client.close()
    finally:
        try:
            server.close()
        except OSError:
            pass


def _idle(stop: threading.Event) -> None:
    while not stop.is_set():
        stop.wait(1.0)


PATTERNS: dict[str, list[tuple[str, float]]] = {
    # nome do padrao -> sequencia de (estagio, fracao_de_tempo)
    "idle": [("idle", 1.0)],
    "cpu": [("cpu", 1.0)],
    "mem": [("mem", 1.0)],
    "net": [("net", 1.0)],
    "mixed": [("idle", 0.25), ("cpu", 0.30), ("mem", 0.20), ("net", 0.15), ("idle", 0.10)],
    "burst": [("idle", 0.5), ("cpu", 0.5)],
}


def _run_stage(stage: str, duration: float, cpu_threads: int, mem_mb: int) -> None:
    if duration <= 0:
        return
    stop = threading.Event()
    threads: list[threading.Thread] = []

    if stage == "idle":
        threads.append(threading.Thread(target=_idle, args=(stop,), daemon=True))
    elif stage == "cpu":
        for _ in range(cpu_threads):
            threads.append(
                threading.Thread(
                    target=_stress_cpu,
                    args=(stop,),
                    kwargs={"intensity": random.uniform(0.5, 0.95)},
                    daemon=True,
                )
            )
    elif stage == "mem":
        threads.append(threading.Thread(target=_stress_memory, args=(stop, mem_mb), daemon=True))
    elif stage == "net":
        threads.append(threading.Thread(target=_stress_net_localhost, args=(stop,), daemon=True))
    else:
        logger.warning("estagio desconhecido", extra={"stage": stage})
        return

    logger.info("estagio iniciado", extra={"stage": stage, "duration_s": duration})
    for t in threads:
        t.start()

    time.sleep(duration)
    stop.set()
    for t in threads:
        t.join(timeout=2.0)
    logger.info("estagio finalizado", extra={"stage": stage})


def run(pattern: str, duration_seconds: float, cpu_threads: int = 0, mem_mb: int = 200) -> None:
    sequence: Iterable[tuple[str, float]] = PATTERNS.get(pattern, PATTERNS["mixed"])
    if cpu_threads <= 0:
        cpu_threads = max(1, (os.cpu_count() or 2) - 1)

    logger.info(
        "load_generator iniciado",
        extra={"pattern": pattern, "duration_s": duration_seconds, "cpu_threads": cpu_threads},
    )
    for stage, fraction in sequence:
        _run_stage(stage, duration_seconds * fraction, cpu_threads, mem_mb)
    logger.info("load_generator finalizado")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gerador controlado de carga")
    parser.add_argument("--pattern", choices=sorted(PATTERNS), default="mixed")
    parser.add_argument("--duration", type=float, default=600.0, help="Duracao total em segundos")
    parser.add_argument("--cpu-threads", type=int, default=0, help="Threads de CPU (0 = automatico)")
    parser.add_argument("--mem-mb", type=int, default=200, help="MB de memoria alocados no estagio mem")
    args = parser.parse_args()
    run(args.pattern, args.duration, args.cpu_threads, args.mem_mb)


if __name__ == "__main__":
    main()
