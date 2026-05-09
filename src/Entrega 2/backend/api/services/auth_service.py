import hashlib
import secrets
from datetime import datetime
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from api.core.config import settings
from bd.models.sessao_auth import SessaoAuth
from bd.models.usuario import Usuario


def _hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_email_domains() -> set[str]:
    parts = [p.strip().lower() for p in settings.AUTH_EMAIL_DOMAINS.split(",")]
    return {p for p in parts if p}


def normalizar_email(email: str) -> str:
    return email.strip().lower()


def email_institucional_valido(email_norm: str) -> bool:
    if "@" not in email_norm:
        return False
    _, _, dom = email_norm.partition("@")
    dom = dom.strip().lower()
    return dom in parse_email_domains()


def normalizar_nome_comparacao(nome: str) -> str:
    return " ".join(nome.strip().casefold().split())


def criar_sessao_auth(db: Session, usuario_id: int) -> tuple[str, SessaoAuth]:
    raw = secrets.token_urlsafe(32)
    row = SessaoAuth(
        token_hash=_hash_token(raw),
        usuario_id=usuario_id,
        criado_em=datetime.utcnow(),
        ultimo_acesso_em=datetime.utcnow(),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return raw, row


def obter_usuario_por_token(
    db: Session,
    raw_token: Optional[str],
    renovar_acesso: bool,
) -> Optional[Usuario]:
    if not raw_token:
        return None
    h = _hash_token(raw_token)
    row = (
        db.query(SessaoAuth)
        .filter(
            SessaoAuth.token_hash == h,
            SessaoAuth.revogada_em.is_(None),
        )
        .first()
    )
    if row is None:
        return None
    now = datetime.utcnow()
    limite_s = settings.AUTH_SESSION_IDLE_MINUTES * 60
    if (now - row.ultimo_acesso_em).total_seconds() > limite_s:
        return None
    if renovar_acesso:
        row.ultimo_acesso_em = now
        db.add(row)
        db.commit()
    usuario = db.query(Usuario).filter(Usuario.id == row.usuario_id).first()
    if usuario is None or not usuario.ativo:
        return None
    if (usuario.perfil or "").lower() != "admin":
        return None
    return usuario


def revogar_sessao_por_token(db: Session, raw_token: Optional[str]) -> None:
    if not raw_token:
        return
    h = _hash_token(raw_token)
    row = (
        db.query(SessaoAuth)
        .filter(
            SessaoAuth.token_hash == h,
            SessaoAuth.revogada_em.is_(None),
        )
        .first()
    )
    if row is None:
        return
    row.revogada_em = datetime.utcnow()
    db.add(row)
    db.commit()


def buscar_usuario_login(
    db: Session,
    nome: str,
    email_norm: str,
) -> Optional[Usuario]:
    """Localiza admin por e-mail (case-insensitive) e valida o nome informado."""
    u = (
        db.query(Usuario)
        .filter(
            func.lower(Usuario.email) == email_norm,
            Usuario.ativo.is_(True),
        )
        .first()
    )
    if u is None:
        return None
    if (u.perfil or "").lower() != "admin":
        return None
    if normalizar_nome_comparacao(u.nome) != normalizar_nome_comparacao(nome):
        return None
    return u
