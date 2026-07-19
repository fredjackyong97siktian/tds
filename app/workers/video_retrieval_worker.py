from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock

from ..config import settings
from ..db import TransactionalSessionLocal
from .. import repositories
from ..services import workflow_service


logger = logging.getLogger("tds.video_retrieval_worker")


@dataclass
class RunningJob:
    future: Future[None]
    location_id: int
    video_asset_id: int


class VideoRetrievalWorker:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max(1, settings.retrieval_max_global_workers))
        self._running: dict[int, RunningJob] = {}
        self._lock = Lock()

    def run_forever(self) -> None:
        poll_seconds = max(1, settings.retrieval_poll_seconds)
        logger.info(
            "Video retrieval worker started with poll=%ss max_global=%s max_per_location=%s",
            poll_seconds,
            settings.retrieval_max_global_workers,
            settings.retrieval_max_per_location,
        )
        while True:
            try:
                self._reap_finished_jobs()
                self._fill_available_slots()
            except Exception:
                logger.exception("Video retrieval worker loop failed")
            time.sleep(poll_seconds)

    def _reap_finished_jobs(self) -> None:
        finished_ids: list[int] = []
        with self._lock:
            items = list(self._running.items())
        for video_asset_id, job in items:
            if not job.future.done():
                continue
            try:
                job.future.result()
                logger.info("Retrieval job completed for video_asset_id=%s location_id=%s", job.video_asset_id, job.location_id)
            except Exception:
                logger.exception("Retrieval job crashed for video_asset_id=%s", job.video_asset_id)
            finished_ids.append(video_asset_id)
        if not finished_ids:
            return
        with self._lock:
            for video_asset_id in finished_ids:
                self._running.pop(video_asset_id, None)

    def _fill_available_slots(self) -> None:
        with self._lock:
            running_jobs = list(self._running.values())
        available_slots = max(0, settings.retrieval_max_global_workers - len(running_jobs))
        if available_slots <= 0:
            return

        running_by_location: dict[int, int] = {}
        for job in running_jobs:
            running_by_location[job.location_id] = running_by_location.get(job.location_id, 0) + 1

        db = TransactionalSessionLocal()
        try:
            # Include already-running rows from DB so a restarted worker does not double-book a location.
            for row in repositories.list_running_video_asset_retrievals(db):
                location_id = row.get("location_id")
                if location_id is None:
                    continue
                running_by_location[int(location_id)] = max(
                    running_by_location.get(int(location_id), 0),
                    1,
                )

            candidates = repositories.list_pending_video_asset_retrievals(
                db,
                limit=max(settings.retrieval_max_global_workers * 10, 20),
            )
            for candidate in candidates:
                if available_slots <= 0:
                    break
                location_id = candidate.get("location_id")
                if location_id is None:
                    continue
                location_id = int(location_id)
                if running_by_location.get(location_id, 0) >= max(1, settings.retrieval_max_per_location):
                    continue
                video_asset_id = int(candidate["id"])
                claimed = repositories.claim_video_asset_for_retrieval(db, video_asset_id)
                if not claimed:
                    continue
                try:
                    job = workflow_service.build_retrieval_job_from_video_asset(db, video_asset_id)
                except Exception as exc:
                    logger.exception("Could not build retrieval job for video_asset_id=%s", video_asset_id)
                    repositories.update_video_asset_status(db, video_asset_id, "issue")
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
                    continue

                future = self._executor.submit(workflow_service.start_video_retrieval_job, job)
                with self._lock:
                    self._running[video_asset_id] = RunningJob(
                        future=future,
                        location_id=location_id,
                        video_asset_id=video_asset_id,
                    )
                running_by_location[location_id] = running_by_location.get(location_id, 0) + 1
                available_slots -= 1
                logger.info("Claimed retrieval job video_asset_id=%s location_id=%s", video_asset_id, location_id)
        finally:
            db.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    worker = VideoRetrievalWorker()
    worker.run_forever()


if __name__ == "__main__":
    main()
