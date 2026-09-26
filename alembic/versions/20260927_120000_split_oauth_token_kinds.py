"""Give a stored API key and a login separate oauth_tokens rows.

Revision ID: 0027
Revises: 0026
"""

import sqlalchemy as sa
from alembic import op

revision = "0027"
down_revision = "0026"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "oauth_tokens",
        sa.Column("kind", sa.String(16), nullable=False, server_default="api_key"),
    )
    # A row holding a refresh or id token is a login: the test the key loader used.
    op.execute(
        "UPDATE oauth_tokens SET kind = 'login' "
        "WHERE coalesce(encrypted_refresh_token, '') <> '' "
        "OR coalesce(encrypted_id_token, '') <> ''"
    )
    op.alter_column("oauth_tokens", "kind", server_default=None)
    op.drop_constraint("pk_oauth_tokens", "oauth_tokens", type_="primary")
    op.create_primary_key("pk_oauth_tokens", "oauth_tokens", ["provider", "kind"])


def downgrade() -> None:
    # One row per provider again: where both exist, the stored API key is kept
    # and the login dropped (it can be signed in again).
    op.execute(
        "DELETE FROM oauth_tokens AS login USING oauth_tokens AS stored "
        "WHERE login.provider = stored.provider "
        "AND login.kind = 'login' AND stored.kind = 'api_key'"
    )
    op.drop_constraint("pk_oauth_tokens", "oauth_tokens", type_="primary")
    op.create_primary_key("pk_oauth_tokens", "oauth_tokens", ["provider"])
    op.drop_column("oauth_tokens", "kind")
