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


logger = logging.getLogger("tds.theft_confidence_worker")


@dataclass
class RunningJob:
    process: subprocess.Popen
    batch_id: int
    location_id: int
    started_at: float = field(default_factory=time.monotonic)


class TheftConfidenceWorker:
    # Batches used to run as a submitted callable on a ThreadPoolExecutor. A
    # hang inside one batch (e.g. a stuck grouping-repair verification loop)
    # occupied that thread forever - Python cannot forcibly cancel a thread
    # blocked in a native/HTTP call, so with max_workers=1 (the default) one
    # stuck batch permanently starved every other batch, including their own
    # repair passes. Running each batch as its own OS subprocess instead means
    # a hung one can always be force-killed at the OS level after a timeout,
    # freeing the slot for the next batch no matter what it was stuck on.
    def __init__(self) -> None:
        self._max_workers = max(1, settings.theft_confidence_max_global_workers)
        self._stale_seconds = max(60, settings.theft_confidence_stale_process_seconds)
        self._running: dict[int, RunningJob] = {}
        self._lock = Lock()

    def run_forever(self) -> None:
        poll_seconds = max(1, settings.theft_confidence_poll_seconds)
        logger.info(
            "Theft confidence worker started with poll=%ss max_global=%s stale_timeout=%ss",
            poll_seconds,
            self._max_workers,
            self._stale_seconds,
        )
        while True:
            try:
                self._reap_finished_jobs()
                self._kill_stale_jobs()
                self._fill_available_slots()
            except Exception:
                logger.exception("Theft confidence worker loop failed")
            time.sleep(poll_seconds)

    def _reap_finished_jobs(self) -> None:
        finished_ids: list[int] = []
        with self._lock:
            items = list(self._running.items())
        for batch_id, job in items:
            exit_code = job.process.poll()
            if exit_code is None:
                continue
            if exit_code == 0:
                logger.info("Theft confidence process finished batch_id=%s location_id=%s", job.batch_id, job.location_id)
            else:
                logger.error(
                    "Theft confidence process exited non-zero batch_id=%s location_id=%s exit_code=%s",
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
                "Theft confidence process stuck for over %ss, killing batch_id=%s location_id=%s pid=%s",
                self._stale_seconds,
                job.batch_id,
                job.location_id,
                job.process.pid,
            )
            job.process.kill()
            job.process.wait()
            with self._lock:
                self._running.pop(batch_id, None)

    def _fill_available_slots(self) -> None:
        with self._lock:
            running_jobs = list(self._running.values())
        available_slots = max(0, self._max_workers - len(running_jobs))
        if available_slots <= 0:
            return

        db = TransactionalSessionLocal()
        try:
            if repositories.is_worker_paused(db, "theft_confidence_analysis"):
                return
            candidates = repositories.list_pending_theft_confidence_batches(
                db,
                limit=max(self._max_workers * 10, 20),
            )
            for candidate in candidates:
                if available_slots <= 0:
                    break
                batch_id = int(candidate["id"])
                if batch_id in self._running:
                    continue
                location_id = int(candidate["location_id"])
                process = subprocess.Popen(
                    [sys.executable, "-m", "app.workers.run_theft_confidence_batch", str(batch_id)]
                )
                with self._lock:
                    self._running[batch_id] = RunningJob(
                        process=process,
                        batch_id=batch_id,
                        location_id=location_id,
                    )
                available_slots -= 1
                logger.info("Claimed theft confidence batch_id=%s location_id=%s pid=%s", batch_id, location_id, process.pid)
        finally:
            db.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    worker = TheftConfidenceWorker()
    worker.run_forever()


if __name__ == "__main__":
    main()
