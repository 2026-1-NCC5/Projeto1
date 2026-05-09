"""Configuracao centralizada via variaveis de ambiente (pydantic-settings)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.common.paths import PROJECT_ROOT, resolve


class Settings(BaseSettings):
    """Configuracoes carregadas de variaveis de ambiente / arquivo .env."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Coletor
    collection_interval_seconds: float = Field(default=5.0, ge=0.5, le=300.0)

    # Diretorios (resolvidos no field_validator abaixo)
    data_dir: Path = Field(default=Path("./data"))
    model_dir: Path = Field(default=Path("./data/models"))
    report_dir: Path = Field(default=Path("./reports"))

    # Dashboard
    dashboard_port: int = Field(default=8501, ge=1, le=65535)
    dashboard_host: str = Field(default="0.0.0.0")
    dashboard_window_min: int = Field(default=60, ge=1, le=24 * 60)
    dashboard_refresh_seconds: int = Field(default=10, ge=0, le=600)

    # Limiares de classificacao (normal / atencao / critico)
    thresh_normal_cpu: float = Field(default=50.0, ge=0.0, le=100.0)
    thresh_normal_mem: float = Field(default=60.0, ge=0.0, le=100.0)
    thresh_critical_cpu: float = Field(default=80.0, ge=0.0, le=100.0)
    thresh_critical_mem: float = Field(default=80.0, ge=0.0, le=100.0)

    # Modo do container e logging
    run_mode: str = Field(default="collector")
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="text")

    @field_validator("data_dir", "model_dir", "report_dir", mode="after")
    @classmethod
    def _resolve_paths(cls, value: Path) -> Path:
        return resolve(value)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def figures_dir(self) -> Path:
        return self.report_dir / "figures"

    @property
    def metrics_dir(self) -> Path:
        return self.report_dir / "metrics"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton de Settings. Use `get_settings.cache_clear()` em testes."""
    return Settings()
