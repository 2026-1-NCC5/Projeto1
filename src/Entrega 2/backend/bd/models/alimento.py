from datetime import datetime
from typing import List
from sqlalchemy import String, Boolean, DateTime, Numeric, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .database import Base


class Alimento(Base):
    __tablename__ = "alimentos"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    nome: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    grupo_alimento_id: Mapped[int] = mapped_column(
        ForeignKey("grupos_alimentos.id", ondelete="SET NULL"), nullable=True
    )
    peso_padrao_kg: Mapped[float] = mapped_column(Numeric(5, 2), nullable=False)
    unidade: Mapped[str] = mapped_column(String(20), default="kg")
    classe_yolo: Mapped[str] = mapped_column(String(50), nullable=True)
    descricao: Mapped[str] = mapped_column(String, nullable=True)
    ativo: Mapped[bool] = mapped_column(Boolean, default=True)
    criado_em: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    grupo_alimento: Mapped["GrupoAlimento"] = relationship(
        "GrupoAlimento", back_populates="alimentos"
    )
    deteccoes: Mapped[List["Deteccao"]] = relationship(
        "Deteccao",
        foreign_keys="Deteccao.alimento_id",
        back_populates="alimento",
    )
    deteccoes_original: Mapped[List["Deteccao"]] = relationship(
        "Deteccao",
        foreign_keys="Deteccao.alimento_id_original",
        back_populates="alimento_original",
    )
