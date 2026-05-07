"""adiciona campos gemini em deteccoes

Revision ID: c1f3e9a4b201
Revises: 8172a49d6648
Create Date: 2026-05-07 16:25:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c1f3e9a4b201'
down_revision: Union[str, Sequence[str], None] = '8172a49d6648'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Adiciona colunas de validação Gemini na tabela deteccoes."""
    with op.batch_alter_table("deteccoes") as batch_op:
        batch_op.add_column(
            sa.Column("fonte", sa.String(length=20), nullable=False, server_default="YOLO")
        )
        batch_op.add_column(sa.Column("gemini_concorda", sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column("gemini_classe", sa.String(length=50), nullable=True))
        batch_op.add_column(sa.Column("gemini_justificativa", sa.String(length=500), nullable=True))


def downgrade() -> None:
    """Remove colunas de validação Gemini."""
    with op.batch_alter_table("deteccoes") as batch_op:
        batch_op.drop_column("gemini_justificativa")
        batch_op.drop_column("gemini_classe")
        batch_op.drop_column("gemini_concorda")
        batch_op.drop_column("fonte")
