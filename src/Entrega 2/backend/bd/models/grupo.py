from datetime import datetime
from typing import List
from sqlalchemy import String, Boolean, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .database import Base


class Grupo(Base):
    __tablename__ = "grupos"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    nome: Mapped[str] = mapped_column(String(100), nullable=False)
    descricao: Mapped[str] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pendente")
    criado_em: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    atualizado_em: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    alunos: Mapped[List["Aluno"]] = relationship(
        "Aluno", back_populates="grupo", cascade="all, delete-orphan"
    )
    sessoes: Mapped[List["Sessao"]] = relationship(
        "Sessao", back_populates="grupo", cascade="all, delete-orphan"
    )
    itens_declarados: Mapped[List["ItemDeclarado"]] = relationship(
        "ItemDeclarado", back_populates="grupo", cascade="all, delete-orphan"
    )
