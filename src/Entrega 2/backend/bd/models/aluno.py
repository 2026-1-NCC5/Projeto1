from datetime import datetime
from sqlalchemy import String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .database import Base


class Aluno(Base):
    __tablename__ = "alunos"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    nome: Mapped[str] = mapped_column(String(100), nullable=False)
    ra: Mapped[str] = mapped_column(String(20), nullable=False, unique=True)
    grupo_id: Mapped[int] = mapped_column(
        ForeignKey("grupos.id", ondelete="CASCADE"), nullable=False
    )
    ativo: Mapped[bool] = mapped_column(Boolean, default=True)
    criado_em: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    grupo: Mapped["Grupo"] = relationship("Grupo", back_populates="alunos")
