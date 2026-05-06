from datetime import datetime
from typing import List
from sqlalchemy import String, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .database import Base


class GrupoAlimento(Base):
    __tablename__ = "grupos_alimentos"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    nome: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    descricao: Mapped[str] = mapped_column(String, nullable=True)
    cor: Mapped[str] = mapped_column(String(7), nullable=True)
    icone: Mapped[str] = mapped_column(String(10), nullable=True)
    criado_em: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    alimentos: Mapped[List["Alimento"]] = relationship(
        "Alimento", back_populates="grupo_alimento"
    )
