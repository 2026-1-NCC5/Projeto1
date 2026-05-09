from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base

if TYPE_CHECKING:
    from .usuario import Usuario


class SessaoAuth(Base):
    """Sessão de login admin (cookie opaco); distinta da sessão de triagem (`sessoes`)."""

    __tablename__ = "sessoes_auth"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    usuario_id: Mapped[int] = mapped_column(
        ForeignKey("usuarios.id", ondelete="CASCADE"), nullable=False
    )
    criado_em: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ultimo_acesso_em: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    revogada_em: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    usuario: Mapped["Usuario"] = relationship("Usuario")
