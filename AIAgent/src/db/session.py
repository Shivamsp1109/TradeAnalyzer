from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.common.config import get_config


config = get_config()
engine = create_engine(
    config.db.url,
    echo=config.db.echo_sql,
    pool_size=config.db.pool_size,
    max_overflow=config.db.max_overflow,
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
