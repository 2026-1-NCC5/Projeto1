"""merge heads after decisao final migration

Revision ID: f0e1d2c3b4a5
Revises: 9d2a0b4f8c31, ab12c3d4e5f6
Create Date: 2026-05-08 00:26:00.000000

"""
from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = "f0e1d2c3b4a5"
down_revision: Union[str, Sequence[str], None] = ("9d2a0b4f8c31", "ab12c3d4e5f6")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Merge migration heads sem alterações estruturais."""


def downgrade() -> None:
    """Sem ações de rollback para merge-only revision."""
