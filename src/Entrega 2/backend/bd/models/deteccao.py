from datetime import datetime
from typing import Optional
from sqlalchemy import String, Boolean, DateTime, Numeric, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .database import Base


class Deteccao(Base):
    __tablename__ = "deteccoes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    sessao_id: Mapped[int] = mapped_column(
        ForeignKey("sessoes.id", ondelete="CASCADE"), nullable=False
    )
    alimento_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("alimentos.id", ondelete="SET NULL"), nullable=True
    )
    alimento_id_original: Mapped[Optional[int]] = mapped_column(
        ForeignKey("alimentos.id", ondelete="SET NULL"), nullable=True
    )
    peso_kg: Mapped[float] = mapped_column(Numeric(5, 2), nullable=False)
    quantidade: Mapped[int] = mapped_column(default=1)
    confianca: Mapped[float] = mapped_column(Numeric(3, 2), nullable=True)
    imagem_path: Mapped[str] = mapped_column(String(500), nullable=True)
    corrigido_manualmente: Mapped[bool] = mapped_column(Boolean, default=False)
    # Origem da classificação final: YOLO, GEMINI, MANUAL ou DESCONHECIDO
    fonte: Mapped[str] = mapped_column(String(20), default="YOLO")
    # Validação cruzada com o Gemini (campos opcionais — preenchidos só quando há retorno)
    gemini_concorda: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    gemini_classe: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    gemini_justificativa: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    # Gemini respondeu mas peso ilegível ou alerta explícito — exige correção antes de finalizar sessão
    revisao_manual_pendente: Mapped[bool] = mapped_column(Boolean, default=False)
    criado_em: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    sessao: Mapped["Sessao"] = relationship("Sessao", back_populates="deteccoes")
    alimento: Mapped[Optional["Alimento"]] = relationship(
        "Alimento", foreign_keys=[alimento_id], back_populates="deteccoes"
    )
    alimento_original: Mapped[Optional["Alimento"]] = relationship(
        "Alimento",
        foreign_keys=[alimento_id_original],
        back_populates="deteccoes_original",
    )
