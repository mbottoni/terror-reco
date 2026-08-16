"""Alembic environment.

The database URL is taken from the application settings rather than
``alembic.ini`` so there is exactly one source of truth, and so that the
``postgres://`` -> ``postgresql+psycopg`` normalisation in ``app.db`` applies
to migrations too.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db import Base, _normalize_database_url  # noqa: E402
from app.settings import get_settings  # noqa: E402

# Importing the models registers every table on Base.metadata; without this
# autogenerate would see an empty schema and propose dropping everything.
import app.models  # noqa: E402,F401  isort:skip

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

DATABASE_URL = _normalize_database_url(get_settings().DATABASE_URL)
config.set_main_option("sqlalchemy.url", DATABASE_URL.replace("%", "%%"))

target_metadata = Base.metadata

# SQLite cannot ALTER most column properties, so alembic needs to rebuild the
# table instead. Harmless on PostgreSQL, essential for local dev on SQLite.
RENDER_AS_BATCH = DATABASE_URL.startswith("sqlite")


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting (``alembic upgrade head --sql``)."""
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        render_as_batch=RENDER_AS_BATCH,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect and run migrations against the live database."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_as_batch=RENDER_AS_BATCH,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
