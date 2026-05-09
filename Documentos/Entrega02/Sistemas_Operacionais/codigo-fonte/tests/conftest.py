"""Fixtures compartilhadas: aponta DATA_DIR/REPORT_DIR para um tmpdir isolado."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.common.config import get_settings


@pytest.fixture()
def isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "data" / "models"))
    monkeypatch.setenv("REPORT_DIR", str(tmp_path / "reports"))
    # Reset do singleton de Settings
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()
    # Garante limpeza de envs (monkeypatch ja faz, mas seguranca extra)
    for var in ("DATA_DIR", "MODEL_DIR", "REPORT_DIR"):
        os.environ.pop(var, None)
