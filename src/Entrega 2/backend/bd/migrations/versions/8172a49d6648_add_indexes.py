"""add indexes

Revision ID: 8172a49d6648
Revises: 40469500dd70
Create Date: 2026-04-30 11:00:48.879237

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8172a49d6648'
down_revision: Union[str, Sequence[str], None] = '40469500dd70'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_index("idx_grupos_status", "grupos", ["status"])
    op.create_index("idx_alunos_grupo", "alunos", ["grupo_id"])
    op.create_index("idx_sessoes_grupo_status", "sessoes", ["grupo_id", "status"])
    op.create_index("idx_deteccoes_sessao", "deteccoes", ["sessao_id"])
    op.create_index("idx_deteccoes_alimento", "deteccoes", ["alimento_id"])
    op.create_index("idx_itens_declarados_grupo", "itens_declarados", ["grupo_id"])
    op.create_index("idx_alimentos_grupo", "alimentos", ["grupo_alimento_id"])
    op.create_index(
        "idx_alimentos_classe_yolo",
        "alimentos",
        ["classe_yolo"],
        sqlite_where=sa.text("classe_yolo IS NOT NULL"),
    )
    op.create_index("idx_deteccoes_criado_em", "deteccoes", ["criado_em"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("idx_deteccoes_criado_em", table_name="deteccoes")
    op.drop_index("idx_alimentos_classe_yolo", table_name="alimentos")
    op.drop_index("idx_alimentos_grupo", table_name="alimentos")
    op.drop_index("idx_itens_declarados_grupo", table_name="itens_declarados")
    op.drop_index("idx_deteccoes_alimento", table_name="deteccoes")
    op.drop_index("idx_deteccoes_sessao", table_name="deteccoes")
    op.drop_index("idx_sessoes_grupo_status", table_name="sessoes")
    op.drop_index("idx_alunos_grupo", table_name="alunos")
    op.drop_index("idx_grupos_status", table_name="grupos")
