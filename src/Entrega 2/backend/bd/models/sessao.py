from datetime import datetime
from typing import List, Optional
from sqlalchemy import String, DateTime, Numeric, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .database import Base


class Sessao(Base):
    __tablename__ = "sessoes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    grupo_id: Mapped[int] = mapped_column(
        ForeignKey("grupos.id", ondelete="CASCADE"), nullable=False
    )
    usuario_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("usuarios.id", ondelete="SET NULL"), nullable=True
    )
    inicio: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    fim: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="ativa")
    total_kg: Mapped[float] = mapped_column(Numeric(8, 2), default=0)
    total_itens: Mapped[int] = mapped_column(default=0)
    fonte_resultado_final: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    decisao_final_em: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    decisao_final_por_usuario_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("usuarios.id", ondelete="SET NULL"), nullable=True
    )
    total_kg_final: Mapped[Optional[float]] = mapped_column(Numeric(8, 2), nullable=True)
    total_itens_final: Mapped[Optional[int]] = mapped_column(nullable=True)
    criado_em: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    grupo: Mapped["Grupo"] = relationship("Grupo", back_populates="sessoes")
    usuario: Mapped[Optional["Usuario"]] = relationship(
        "Usuario",
        back_populates="sessoes",
        foreign_keys=[usuario_id],
    )
    deteccoes: Mapped[List["Deteccao"]] = relationship(
        "Deteccao", back_populates="sessao", cascade="all, delete-orphan"
    )
