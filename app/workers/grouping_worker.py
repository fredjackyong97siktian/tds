from __future__ import annotations

import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from threading import Lock

from .. import repositories
from ..config import settings
from ..db import TransactionalSessionLocal
from ..services import workflow_service


logger = logging.getLogger("tds.grouping_worker")


@dataclass
class RunningJob:
    process: subprocess.Popen
    batch_id: int
    location_id: int
    started_at: float = field(default_factory=time.monotonic)


class GroupingWorker:
    # Batches used to run as a submitted callable on a ThreadPoolExecutor. A
    # hang inside one batch's dispatch occupied the only thread forever - with
    # max_global_workers=1 (the default), that didn't just block new batches
    # from running, it stopped the scheduler from even looking for due
    # windows at all (see the old available_slots<=0 early-return, which sat
    # before prepare_due_grouping_batches ever ran) - confirmed live: worker
    # logs went dead silent for hours while the container still showed "up".
    # Running each batch as its own OS subprocess means a hung one can always
    # be force-killed after a timeout, same fix already applied to the
    # theft-confidence worker.
    def __init__(self) -> None:
        self._max_workers = max(1, settings.grouping_max_global_workers)
        self._stale_seconds = max(60, settings.grouping_stale_process_seconds)
        self._running: dict[int, RunningJob] = {}
        self._lock = Lock()

    def run_forever(self) -> None:
        poll_seconds = max(1, settings.grouping_poll_seconds)
        logger.info(
            "Grouping worker started with poll=%ss max_global=%s stale_timeout=%ss",
            poll_seconds,
            self._max_workers,
            self._stale_seconds,
        )
        self._recover_orphaned_batches()
        while True:
            try:
                self._reap_finished_jobs()
                self._kill_stale_jobs()
                self._recover_orphaned_batches()
                self._fill_available_slots()
            except Exception:
                logger.exception("Grouping worker loop failed")
            time.sleep(poll_seconds)

    def _recover_orphaned_batches(self) -> None:
        # Any batch sitting at 'dispatching'/'running' that ISN'T in
        # self._running cannot possibly have a live process behind it in
        # this instance - either this process just started (self._running
        # starts empty, so nothing here yet is genuinely in flight), or
        # _kill_stale_jobs already popped it after killing its process.
        # Called every poll, not just once at startup or once right after a
        # kill, because a single recovery attempt can itself fail - most
        # notably when the killed process was itself mid theft-confidence
        # analysis for this same batch: a SIGKILL never runs its own
        # release_lock() cleanup, so force_recover_killed_grouping_batch's
        # own lock-acquisition guard sees that orphaned lock as "still held"
        # and refuses to proceed. Confirmed live: batch 180 got stuck exactly
        # this way, with nothing left to ever retry it since recovery used to
        # only ever be attempted once. Retrying every poll gives MySQL's own
        # dead-connection detection time to release the orphaned lock, and
        # keeps trying until it does.
        with self._lock:
            currently_tracked = set(self._running.keys())
        db = TransactionalSessionLocal()
        try:
            orphaned = repositories.list_running_grouping_batches(db)
            for row in orphaned:
                batch_id = int(row["id"])
                if batch_id in currently_tracked:
                    continue
                logger.warning(
                    "Recovering orphaned grouping batch_id=%s (status=%s) - no process in this instance owns it",
                    batch_id,
                    row.get("status"),
                )
                try:
                    workflow_service.force_recover_killed_grouping_batch(db, batch_id=batch_id)
                except Exception:
                    logger.exception("Could not recover orphaned grouping batch_id=%s", batch_id)
        finally:
            db.close()

    def _reap_finished_jobs(self) -> None:
        finished_ids: list[int] = []
        with self._lock:
            items = list(self._running.items())
        for batch_id, job in items:
            exit_code = job.process.poll()
            if exit_code is None:
                continue
            if exit_code == 0:
                logger.info("Grouping dispatch process finished batch_id=%s location_id=%s", job.batch_id, job.location_id)
            else:
                logger.error(
                    "Grouping dispatch process exited non-zero batch_id=%s location_id=%s exit_code=%s",
                    job.batch_id,
                    job.location_id,
                    exit_code,
                )
            finished_ids.append(batch_id)
        if not finished_ids:
            return
        with self._lock:
            for batch_id in finished_ids:
                self._running.pop(batch_id, None)

    def _kill_stale_jobs(self) -> None:
        now = time.monotonic()
        with self._lock:
            items = list(self._running.items())
        for batch_id, job in items:
            if now - job.started_at < self._stale_seconds:
                continue
            logger.error(
                "Grouping dispatch process stuck for over %ss, killing batch_id=%s location_id=%s pid=%s",
                self._stale_seconds,
                job.batch_id,
                job.location_id,
                job.process.pid,
            )
            job.process.kill()
            job.process.wait()
            with self._lock:
                self._running.pop(batch_id, None)
            # Recovery itself (resetting the batch back to 'pending') is left
            # to _recover_orphaned_batches, called right after this in the
            # same poll - now that the batch is popped from self._running,
            # that pass picks it up naturally. Doing it here too would
            # double-process the same batch in one poll (reset it, then
            # immediately reset it again) whenever the first attempt
            # succeeds before the second one runs.

    def _fill_available_slots(self) -> None:
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
                logger.warning("Grouping worker is paused - skipping batch preparation and dispatch this poll")
                return
            blocking_runs = repositories.list_active_remote_analysis_script_runs(db, script_names=["grouping"])
            if blocking_runs:
                # This used to be a silent early return - a single leftover
                # 'running' script_run row (with a runner_job_id, however that
                # got set) here blocks EVERY future batch from ever being
                # created or dispatched, with zero log trace, until something
                # else clears that row. Confirmed live: this cost hours of
                # missed grouping windows with nothing in the log to point at.
                logger.warning(
                    "Grouping dispatch blocked by %s active remote script_run row(s) still 'running': %s",
                    len(blocking_runs),
                    [
                        {"id": row["id"], "runner_job_id": row["runner_job_id"], "started_at": str(row["started_at"])}
                        for row in blocking_runs
                    ],
                )
                return
            # Preparing (creating) due batch rows is cheap DB work, independent
            # of whether a dispatch slot is currently free - always run this so
            # a busy/stale slot doesn't also stop new windows from even getting
            # their batch created, on top of not being dispatched yet.
            workflow_service.prepare_due_grouping_batches(db)

            with self._lock:
                running_jobs = list(self._running.values())
            available_slots = max(0, self._max_workers - len(running_jobs))
            if available_slots <= 0:
                return

            candidates = repositories.list_pending_grouping_batches(
                db,
                limit=max(self._max_workers * 10, 20),
            )
            for candidate in candidates:
                if available_slots <= 0:
                    break
                batch_id = int(candidate["id"])
                if batch_id in self._running:
                    continue
                if not repositories.claim_grouping_batch_for_dispatch(db, batch_id):
                    continue
                location_id = int(candidate["location_id"])
                process = subprocess.Popen(
                    [sys.executable, "-m", "app.workers.run_grouping_batch", str(batch_id)]
                )
                with self._lock:
                    self._running[batch_id] = RunningJob(
                        process=process,
                        batch_id=batch_id,
                        location_id=location_id,
                    )
                available_slots -= 1
                logger.info("Claimed grouping batch_id=%s location_id=%s pid=%s", batch_id, location_id, process.pid)
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
