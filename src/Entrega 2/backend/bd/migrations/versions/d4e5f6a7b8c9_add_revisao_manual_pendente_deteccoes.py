"""adiciona revisao_manual_pendente em deteccoes

Revision ID: d4e5f6a7b8c9
Revises: c7a8b9d0e1f2
Create Date: 2026-05-09 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, Sequence[str], None] = "c7a8b9d0e1f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("deteccoes") as batch_op:
        batch_op.add_column(
            sa.Column(
                "revisao_manual_pendente",
                sa.Boolean(),
                nullable=False,
                server_default="0",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("deteccoes") as batch_op:
        batch_op.drop_column("revisao_manual_pendente")
