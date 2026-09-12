from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .config import settings


transactional_engine = create_engine(
    settings.transactional_database_url,
    future=True,
    pool_pre_ping=True,
    # pool_pre_ping alone only catches a connection that fails FAST (a clean
    # reset) - a half-open connection (packets silently dropped, no reset,
    # common after a long idle period through a NAT/load balancer hop) makes
    # the ping itself block forever with no timeout, which froze every worker
    # sharing this engine for hours with no exception ever raised (confirmed
    # live: grouping_worker went dead silent from 05:13 to past 10:50 with
    # the process still up, no crash, no log line at all). These socket-level
    # timeouts plus recycling connections before they go stale close that gap.
    pool_recycle=1800,
    connect_args={
        "connect_timeout": 10,
        "read_timeout": 30,
        "write_timeout": 30,
    },
)
vector_engine = create_engine(
    settings.vector_database_url,
    future=True,
    pool_pre_ping=True,
    pool_recycle=1800,
    connect_args={"connect_timeout": 10},
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
