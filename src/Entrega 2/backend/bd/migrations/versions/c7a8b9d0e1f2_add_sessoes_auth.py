"""add sessoes_auth for admin cookie sessions

Revision ID: c7a8b9d0e1f2
Revises: f0e1d2c3b4a5
Create Date: 2026-05-08 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "c7a8b9d0e1f2"
down_revision: Union[str, None] = "f0e1d2c3b4a5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "sessoes_auth",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column("usuario_id", sa.Integer(), nullable=False),
        sa.Column("criado_em", sa.DateTime(), nullable=False),
        sa.Column("ultimo_acesso_em", sa.DateTime(), nullable=False),
        sa.Column("revogada_em", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["usuario_id"], ["usuarios.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash"),
    )


def downgrade() -> None:
    op.drop_table("sessoes_auth")
