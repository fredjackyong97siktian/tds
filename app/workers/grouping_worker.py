from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock

from .. import repositories
from ..config import settings
from ..db import TransactionalSessionLocal
from ..services import workflow_service


logger = logging.getLogger("tds.grouping_worker")


@dataclass
class RunningJob:
    future: Future[workflow_service.ScriptExecutionResult]
    batch_id: int
    location_id: int


class GroupingWorker:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max(1, settings.grouping_max_global_workers))
        self._running: dict[int, RunningJob] = {}
        self._lock = Lock()

    def run_forever(self) -> None:
        poll_seconds = max(1, settings.grouping_poll_seconds)
        logger.info(
            "Grouping worker started with poll=%ss max_global=%s",
            poll_seconds,
            settings.grouping_max_global_workers,
        )
        while True:
            try:
                self._reap_finished_jobs()
                self._fill_available_slots()
            except Exception:
                logger.exception("Grouping worker loop failed")
            time.sleep(poll_seconds)

    def _reap_finished_jobs(self) -> None:
        finished_ids: list[int] = []
        with self._lock:
            items = list(self._running.items())
        for batch_id, job in items:
            if not job.future.done():
                continue
            try:
                result = job.future.result()
                logger.info(
                    "Grouping dispatch finished batch_id=%s location_id=%s status=%s runner_job_id=%s",
                    job.batch_id,
                    job.location_id,
                    result.status,
                    result.runner_job_id,
                )
            except Exception:
                logger.exception("Grouping dispatch crashed for batch_id=%s", job.batch_id)
            finished_ids.append(batch_id)
        if not finished_ids:
            return
        with self._lock:
            for batch_id in finished_ids:
                self._running.pop(batch_id, None)

    def _fill_available_slots(self) -> None:
        with self._lock:
            running_jobs = list(self._running.values())
        available_slots = max(0, settings.grouping_max_global_workers - len(running_jobs))
        if available_slots <= 0:
            return

        db = TransactionalSessionLocal()
        try:
            reconciled = workflow_service.reconcile_running_remote_analysis_script_runs(db)
            for item in reconciled:
                logger.info(
                    "Reconciled remote script_run_id=%s runner_job_id=%s script=%s runpod_status=%s status=%s",
                    item["script_run_id"],
                    item["runner_job_id"],
                    item["script_name"],
                    item["runpod_status"],
                    item["status"],
                )
            if repositories.is_worker_paused(db, "grouping"):
                return
            if repositories.has_active_remote_analysis_script_run(db, script_names=["grouping"]):
                return
            workflow_service.prepare_due_grouping_batches(db)
            candidates = repositories.list_pending_grouping_batches(
                db,
                limit=max(settings.grouping_max_global_workers * 10, 20),
            )
            for candidate in candidates:
                if available_slots <= 0:
                    break
                batch_id = int(candidate["id"])
                if not repositories.claim_grouping_batch_for_dispatch(db, batch_id):
                    continue
                try:
                    job = workflow_service.build_grouping_analysis_job_from_batch(db, batch_id)
                    future = self._executor.submit(workflow_service.start_grouping_analysis_job, job)
                except Exception as exc:
                    logger.exception("Could not build grouping job for batch_id=%s", batch_id)
                    repositories.update_grouping_batch(
                        db,
                        batch_id,
                        {
                            "status": "issue",
                            "issue_reason": str(exc),
                        },
                    )
                    continue
                with self._lock:
                    self._running[batch_id] = RunningJob(
                        future=future,
                        batch_id=batch_id,
                        location_id=job.location_id,
                    )
                available_slots -= 1
                logger.info("Claimed grouping batch_id=%s location_id=%s", batch_id, job.location_id)
        finally:
            db.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    worker = GroupingWorker()
    worker.run_forever()


if __name__ == "__main__":
    main()
