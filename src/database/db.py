"""
Database connection and session management.
"""
import os
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from .models import Base

load_dotenv()


def _get_database_url() -> str:
    url = os.getenv("DATABASE_URL", "sqlite:///data/tt_matches.db")
    # Fix Render/Heroku postgres:// connections mapping to postgresql://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    # Résolution du chemin relatif pour SQLite
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        relative_path = url[len("sqlite:///"):]
        abs_path = Path(__file__).resolve().parents[2] / relative_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{abs_path}"
    return url


def _enable_wal_mode(dbapi_conn, _connection_record):
    """Active WAL pour de meilleures performances SQLite en écriture concurrente."""
    dbapi_conn.execute("PRAGMA journal_mode=WAL")
    dbapi_conn.execute("PRAGMA foreign_keys=ON")


DATABASE_URL = _get_database_url()
engine = create_engine(DATABASE_URL, echo=False)

if DATABASE_URL.startswith("sqlite"):
    event.listen(engine, "connect", _enable_wal_mode)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db() -> None:
    """Crée toutes les tables si elles n'existent pas."""
    Base.metadata.create_all(bind=engine)
    logger.info(f"Database initialized at {DATABASE_URL}")


@contextmanager
def get_session():
    """Context manager pour une session DB avec rollback automatique en cas d'erreur."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
