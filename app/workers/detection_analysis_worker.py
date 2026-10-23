"""Dedicated bounded detection queue; never block frame/video retrieval."""

import logging
import time

from .. import repositories
from ..db import TransactionalSessionLocal
from ..services import detection_analysis

logger = logging.getLogger(__name__)


def run_once():
    with TransactionalSessionLocal() as db:
        if not detection_analysis.ENABLED or repositories.is_worker_paused(db, "entrance_analysis"):
            return
        candidates = detection_analysis.pending(db)
    for session_id in candidates:
        with TransactionalSessionLocal() as db:
            if repositories.is_worker_paused(db, "entrance_analysis"):
                return
            detection_analysis.run_session(db, int(session_id))


def main():
    logging.basicConfig(level=logging.INFO)
    while True:
        try:
            run_once()
        except Exception:
            logger.exception("Detection Analysis worker loop failed")
        time.sleep(10)


if __name__ == "__main__":
    main()
