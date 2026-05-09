"""adiciona metadados de decisao final em sessoes

Revision ID: ab12c3d4e5f6
Revises: c1f3e9a4b201
Create Date: 2026-05-08 00:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "ab12c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "c1f3e9a4b201"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Adiciona campos de decisao final para consolidacao oficial da sessao."""
    with op.batch_alter_table("sessoes") as batch_op:
        batch_op.add_column(sa.Column("fonte_resultado_final", sa.String(length=20), nullable=True))
        batch_op.add_column(sa.Column("decisao_final_em", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("decisao_final_por_usuario_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("total_kg_final", sa.Numeric(8, 2), nullable=True))
        batch_op.add_column(sa.Column("total_itens_final", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_sessoes_decisao_final_por_usuario_id",
            "usuarios",
            ["decisao_final_por_usuario_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    """Remove campos de decisao final da sessao."""
    with op.batch_alter_table("sessoes") as batch_op:
        batch_op.drop_constraint("fk_sessoes_decisao_final_por_usuario_id", type_="foreignkey")
        batch_op.drop_column("total_itens_final")
        batch_op.drop_column("total_kg_final")
        batch_op.drop_column("decisao_final_por_usuario_id")
        batch_op.drop_column("decisao_final_em")
        batch_op.drop_column("fonte_resultado_final")
