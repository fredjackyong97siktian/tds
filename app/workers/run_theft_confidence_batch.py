"""Standalone entrypoint to run theft-confidence scoring for one grouping batch.

Run as its own OS process (see theft_confidence_worker.py) rather than a thread
so a hang inside it (e.g. a stuck grouping-repair verification loop) can be
force-killed without taking down the worker or blocking every other batch -
Python threads cannot be forcibly cancelled once blocked in a native call, but
a subprocess can always be killed at the OS level.
"""

from __future__ import annotations

import logging
import sys

from ..db import TransactionalSessionLocal
from ..services import workflow_service


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logger = logging.getLogger("tds.run_theft_confidence_batch")

    if len(sys.argv) != 2:
        logger.error("Usage: python -m app.workers.run_theft_confidence_batch <batch_id>")
        return 2

    batch_id = int(sys.argv[1])
    db = TransactionalSessionLocal()
    try:
        result = workflow_service.run_theft_confidence_for_grouping_batch(db, batch_id=batch_id)
        logger.info(
            "Theft confidence completed batch_id=%s analyzed=%s promoted=%s",
            batch_id,
            result.get("analyzed_count"),
            result.get("promoted_count"),
        )
        return 0
    except Exception:
        logger.exception("Theft confidence crashed for batch_id=%s", batch_id)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
