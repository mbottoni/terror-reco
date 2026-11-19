"""lowercase user emails

Addresses were stored exactly as typed and compared case-sensitively, so the
same person could hold two accounts. The application now normalises on write
and on lookup; this brings existing rows into the same form so the existing
unique index on ``users.email`` enforces one account per address.

Revision ID: c4d1e7a9b2f3
Revises: 823c5251029c
Create Date: 2026-11-19 23:10:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c4d1e7a9b2f3"
down_revision: str | Sequence[str] | None = "823c5251029c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Trim and lowercase every address, refusing to merge accounts silently."""
    bind = op.get_bind()
    collisions = bind.execute(
        sa.text(
            "SELECT lower(trim(email)) AS e, count(*) AS n FROM users "
            "GROUP BY lower(trim(email)) HAVING count(*) > 1"
        )
    ).fetchall()
    if collisions:
        # Two accounts that differ only by case each have their own history,
        # feedback and watchlist. Picking a winner is a human decision.
        raise RuntimeError(
            f"{len(collisions)} email address(es) are registered more than once with "
            "different casing. Merge or remove the duplicates, then re-run the migration."
        )
    bind.execute(
        sa.text("UPDATE users SET email = lower(trim(email)) WHERE email <> lower(trim(email))")
    )


def downgrade() -> None:
    """The original casing is not recoverable; nothing to undo."""
