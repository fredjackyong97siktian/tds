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


logger = logging.getLogger("tds.theft_confidence_worker")


@dataclass
class RunningJob:
    future: Future[dict]
    batch_id: int
    location_id: int


class TheftConfidenceWorker:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max(1, settings.theft_confidence_max_global_workers))
        self._running: dict[int, RunningJob] = {}
        self._lock = Lock()

    def run_forever(self) -> None:
        poll_seconds = max(1, settings.theft_confidence_poll_seconds)
        logger.info(
            "Theft confidence worker started with poll=%ss max_global=%s",
            poll_seconds,
            settings.theft_confidence_max_global_workers,
        )
        while True:
            try:
                self._reap_finished_jobs()
                self._fill_available_slots()
            except Exception:
                logger.exception("Theft confidence worker loop failed")
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
                    "Theft confidence completed batch_id=%s location_id=%s analyzed=%s promoted=%s",
                    job.batch_id,
                    job.location_id,
                    result.get("analyzed_count"),
                    result.get("promoted_count"),
                )
            except Exception:
                logger.exception("Theft confidence crashed for batch_id=%s", job.batch_id)
            finished_ids.append(batch_id)
        if not finished_ids:
            return
        with self._lock:
            for batch_id in finished_ids:
                self._running.pop(batch_id, None)

    def _fill_available_slots(self) -> None:
        with self._lock:
            running_jobs = list(self._running.values())
        available_slots = max(0, settings.theft_confidence_max_global_workers - len(running_jobs))
        if available_slots <= 0:
            return

        db = TransactionalSessionLocal()
        try:
            if repositories.is_worker_paused(db, "theft_confidence_analysis"):
                return
            candidates = repositories.list_pending_theft_confidence_batches(
                db,
                limit=max(settings.theft_confidence_max_global_workers * 10, 20),
            )
            for candidate in candidates:
                if available_slots <= 0:
                    break
                batch_id = int(candidate["id"])
                if batch_id in self._running:
                    continue
                location_id = int(candidate["location_id"])
                future = self._executor.submit(_run_batch, batch_id)
                with self._lock:
                    self._running[batch_id] = RunningJob(
                        future=future,
                        batch_id=batch_id,
                        location_id=location_id,
                    )
                available_slots -= 1
                logger.info("Claimed theft confidence batch_id=%s location_id=%s", batch_id, location_id)
        finally:
            db.close()


def _run_batch(batch_id: int) -> dict:
    db = TransactionalSessionLocal()
    try:
        return workflow_service.run_theft_confidence_for_grouping_batch(db, batch_id=batch_id)
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
