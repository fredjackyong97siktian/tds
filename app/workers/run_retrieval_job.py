"""Standalone entrypoint to run one video/frame retrieval job.

Run as its own OS process (see video_retrieval_worker.py) rather than a
thread so a hang inside it (a stalled ffmpeg/RTSP call talking to the NVR is
exactly the kind of thing that can block forever) can be force-killed
without permanently occupying a worker slot. Mirrors run_grouping_batch.py
and run_theft_confidence_batch.py, which fixed the identical issue in their
respective workers.
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
    logger = logging.getLogger("tds.run_retrieval_job")

    if len(sys.argv) != 3 or sys.argv[1] not in ("video", "frame"):
        logger.error("Usage: python -m app.workers.run_retrieval_job <video|frame> <asset_id>")
        return 2

    kind = sys.argv[1]
    asset_id = int(sys.argv[2])
    db = TransactionalSessionLocal()
    try:
        try:
            if kind == "video":
                job = workflow_service.build_retrieval_job_from_video_asset(db, asset_id)
            else:
                job = workflow_service.build_retrieval_job_from_trigger_frame_asset(db, asset_id)
        except Exception as exc:
            # Building the job is a fast, synchronous, non-network step (just
            # DB reads) - a failure here is a real data/logic error, not a
            # hang, so it's reported the same specific way the worker always
            # has rather than falling through to the generic crash handler.
            logger.exception("Could not build %s retrieval job for asset_id=%s", kind, asset_id)
            if kind == "video":
                repositories.update_video_asset_status(db, asset_id, "issue")
                repositories.create_script_run(
                    db,
                    session_id=None,
                    trigger_id=None,
                    script_name="retrieve_video",
                    model_name="worker_build_job",
                    status="failed",
                    command="worker_build_job",
                    stdout_log="",
                    stderr_log=str(exc),
                )
            else:
                repositories.update_trigger_frame_asset_status(db, asset_id, "issue", error=str(exc))
                repositories.create_script_run(
                    db,
                    session_id=None,
                    trigger_id=None,
                    script_name="retrieve_video",
                    model_name="worker_build_frame_job",
                    status="failed",
                    command="worker_build_frame_job",
                    stdout_log="",
                    stderr_log=str(exc),
                )
            return 1

        if kind == "video":
            workflow_service.start_video_retrieval_job(job)
        else:
            workflow_service.start_trigger_frame_asset_retrieval_job(job)
        logger.info("Retrieval job completed for %s_id=%s", kind, asset_id)
        return 0
    except Exception:
        logger.exception("Retrieval job crashed for %s_id=%s", kind, asset_id)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
