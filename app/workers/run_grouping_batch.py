"""Standalone entrypoint to dispatch grouping analysis for one batch.

Run as its own OS process (see grouping_worker.py) rather than a thread so a
hang inside it can be force-killed without blocking every other batch -
Python threads cannot be forcibly cancelled once blocked in a native call,
but a subprocess can always be killed at the OS level. Mirrors
run_theft_confidence_batch.py, which fixed the identical issue in the
theft-confidence worker.
"""

from __future__ import annotations

import logging
import sys

from .. import repositories
from ..db import TransactionalSessionLocal
from ..services import workflow_service


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logger = logging.getLogger("tds.run_grouping_batch")

    if len(sys.argv) != 2:
        logger.error("Usage: python -m app.workers.run_grouping_batch <batch_id>")
        return 2

    batch_id = int(sys.argv[1])
    db = TransactionalSessionLocal()
    try:
        try:
            job = workflow_service.build_grouping_analysis_job_from_batch(db, batch_id)
        except Exception as exc:
            # Building the job is a fast, synchronous, non-network step - a
            # failure here is a real data/logic error, not a hang, so it's
            # reported the same specific way the worker always has rather
            # than falling through to the generic crash handler.
            logger.exception("Could not build grouping job for batch_id=%s", batch_id)
            repositories.update_grouping_batch(db, batch_id, {"status": "issue", "issue_reason": str(exc)})
            return 1
        result = workflow_service.start_grouping_analysis_job(job)
        logger.info(
            "Grouping dispatch finished batch_id=%s location_id=%s status=%s runner_job_id=%s",
            batch_id,
            job.location_id,
            result.status,
            result.runner_job_id,
        )
        return 0
    except Exception:
        logger.exception("Grouping dispatch crashed for batch_id=%s", batch_id)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
