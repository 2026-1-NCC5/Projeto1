"""Resolve diretorios base do projeto de forma reproduzivel."""

from __future__ import annotations

from pathlib import Path

# Raiz do projeto = pai do pacote `app`
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def resolve(path_like: str | Path) -> Path:
    """Resolve um caminho relativo contra a raiz do projeto.

    Caminhos absolutos sao retornados sem modificacao.
    """
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def ensure_dir(path: Path) -> Path:
    """Cria o diretorio (e pais) caso nao exista e retorna o path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
