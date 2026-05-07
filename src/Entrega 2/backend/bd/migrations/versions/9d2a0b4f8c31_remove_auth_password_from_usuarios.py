"""remove auth password from usuarios

Revision ID: 9d2a0b4f8c31
Revises: c1f3e9a4b201
Create Date: 2026-05-07 17:12:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9d2a0b4f8c31"
down_revision: Union[str, Sequence[str], None] = "c1f3e9a4b201"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Remove a coluna de senha porque a PoC não usa autenticação."""
    with op.batch_alter_table("usuarios") as batch_op:
        batch_op.drop_column("senha_hash")


def downgrade() -> None:
    """Recria a coluna de senha apenas para rollback estrutural."""
    with op.batch_alter_table("usuarios") as batch_op:
        batch_op.add_column(
            sa.Column(
                "senha_hash",
                sa.String(length=255),
                nullable=False,
                server_default="",
            )
        )
