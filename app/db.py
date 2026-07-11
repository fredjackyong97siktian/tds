from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .config import settings


transactional_engine = create_engine(
    settings.transactional_database_url,
    future=True,
    pool_pre_ping=True,
)
vector_engine = create_engine(
    settings.vector_database_url,
    future=True,
    pool_pre_ping=True,
)

TransactionalSessionLocal = sessionmaker(
    bind=transactional_engine,
    autoflush=False,
    autocommit=False,
    future=True,
)
VectorSessionLocal = sessionmaker(
    bind=vector_engine,
    autoflush=False,
    autocommit=False,
    future=True,
)


def get_transaction_db() -> Generator[Session, None, None]:
    db = TransactionalSessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_vector_db() -> Generator[Session, None, None]:
    db = VectorSessionLocal()
    try:
        yield db
    finally:
        db.close()
