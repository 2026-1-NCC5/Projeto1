from datetime import datetime
from sqlalchemy import String, DateTime, Numeric, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .database import Base


class ItemDeclarado(Base):
    __tablename__ = "itens_declarados"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    grupo_id: Mapped[int] = mapped_column(
        ForeignKey("grupos.id", ondelete="CASCADE"), nullable=False
    )
    nome_alimento: Mapped[str] = mapped_column(String(100), nullable=False)
    marca: Mapped[str] = mapped_column(String(100), nullable=True)
    quantidade: Mapped[int] = mapped_column(default=1)
    peso_declarado_kg: Mapped[float] = mapped_column(Numeric(8, 2), default=0)
    observacao: Mapped[str] = mapped_column(String, nullable=True)
    criado_em: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    grupo: Mapped["Grupo"] = relationship("Grupo", back_populates="itens_declarados")
