from datetime import datetime
from typing import List
from sqlalchemy import String, Boolean, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .database import Base


class Usuario(Base):
    __tablename__ = "usuarios"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    nome: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    perfil: Mapped[str] = mapped_column(String(20), default="operador")
    ativo: Mapped[bool] = mapped_column(Boolean, default=True)
    criado_em: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    sessoes: Mapped[List["Sessao"]] = relationship(
        "Sessao",
        back_populates="usuario",
        foreign_keys="[Sessao.usuario_id]",
        cascade="all, delete-orphan",
    )
