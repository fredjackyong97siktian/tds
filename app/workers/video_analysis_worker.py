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


logger = logging.getLogger("tds.video_analysis_worker")


@dataclass
class RunningJob:
    future: Future[workflow_service.ScriptExecutionResult]
    location_id: int
    video_asset_id: int


class VideoAnalysisWorker:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max(1, settings.analysis_max_global_workers))
        self._running: dict[int, RunningJob] = {}
        self._lock = Lock()
        self._next_dispatch_after = 0.0

    def run_forever(self) -> None:
        poll_seconds = max(1, settings.analysis_poll_seconds)
        logger.info(
            "Video analysis worker started with poll=%ss max_global=%s cooldown=%ss",
            poll_seconds,
            settings.analysis_max_global_workers,
            settings.analysis_cooldown_seconds,
        )
        while True:
            try:
                self._reap_finished_jobs()
                self._fill_available_slots()
            except Exception:
                logger.exception("Video analysis worker loop failed")
            time.sleep(poll_seconds)

    def _reap_finished_jobs(self) -> None:
        finished_ids: list[int] = []
        cooldown_seconds = max(0, settings.analysis_cooldown_seconds)
        with self._lock:
            items = list(self._running.items())
        for video_asset_id, job in items:
            if not job.future.done():
                continue
            try:
                result = job.future.result()
                logger.info(
                    "Analysis dispatch finished for video_asset_id=%s location_id=%s status=%s runner_job_id=%s",
                    job.video_asset_id,
                    job.location_id,
                    result.status,
                    result.runner_job_id,
                )
            except Exception:
                logger.exception("Analysis dispatch crashed for video_asset_id=%s", job.video_asset_id)
            finished_ids.append(video_asset_id)
        if not finished_ids:
            return
        with self._lock:
            for video_asset_id in finished_ids:
                self._running.pop(video_asset_id, None)
            self._next_dispatch_after = time.time() + cooldown_seconds

    def _fill_available_slots(self) -> None:
        now = time.time()
        with self._lock:
            running_jobs = list(self._running.values())
            next_dispatch_after = self._next_dispatch_after
        if now < next_dispatch_after:
            return

        available_slots = max(0, settings.analysis_max_global_workers - len(running_jobs))
        if available_slots <= 0:
            return

        db = TransactionalSessionLocal()
        try:
            if repositories.is_worker_paused(db, "analysis"):
                return
            if repositories.has_active_remote_analysis_script_run(db):
                return
            if repositories.list_running_video_asset_analyses(db):
                return

            candidates = repositories.list_pending_video_asset_analyses(
                db,
                limit=max(settings.analysis_max_global_workers * 10, 20),
            )
            for candidate in candidates:
                if available_slots <= 0:
                    break
                video_asset_id = int(candidate["id"])
                location_id = int(candidate["location_id"])
                claimed = repositories.claim_video_asset_for_analysis(db, video_asset_id)
                if not claimed:
                    continue
                try:
                    job = workflow_service.build_entrance_analysis_job_from_video_asset(db, video_asset_id)
                except Exception as exc:
                    logger.exception("Could not build analysis job for video_asset_id=%s", video_asset_id)
                    repositories.update_video_asset_status(db, video_asset_id, "issue")
                    repositories.create_script_run(
                        db,
                        session_id=None,
                        trigger_id=None,
                        script_name="entry",
                        model_name="worker_build_job",
                        status="failed",
                        command="worker_build_job",
                        stdout_log="",
                        stderr_log=str(exc),
                    )
                    continue

                future = self._executor.submit(workflow_service.start_entrance_analysis_job, job)
                with self._lock:
                    self._running[video_asset_id] = RunningJob(
                        future=future,
                        location_id=location_id,
                        video_asset_id=video_asset_id,
                    )
                available_slots -= 1
                logger.info("Claimed analysis dispatch video_asset_id=%s location_id=%s", video_asset_id, location_id)
                break
        finally:
            db.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    worker = VideoAnalysisWorker()
    worker.run_forever()


if __name__ == "__main__":
    main()
