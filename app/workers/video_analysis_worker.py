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
from ..services.workflow_service import ScriptExecutionResult


logger = logging.getLogger("tds.video_analysis_worker")


@dataclass
class RunningJob:
    future: Future[ScriptExecutionResult]
    location_id: int
    video_asset_id: int


class VideoAnalysisWorker:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max(1, settings.analysis_max_global_workers))
        self._running: dict[int, RunningJob] = {}
        self._lock = Lock()

    def run_forever(self) -> None:
        poll_seconds = max(1, settings.analysis_poll_seconds)
        logger.info(
            "Video analysis worker started with poll=%ss max_global=%s max_per_location=%s",
            poll_seconds,
            settings.analysis_max_global_workers,
            settings.analysis_max_per_location,
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
        finished_count = 0
        with self._lock:
            items = list(self._running.items())
        for video_asset_id, job in items:
            if not job.future.done():
                continue
            try:
                result = job.future.result()
                if result.status == "success":
                    logger.info(
                        "Analysis job completed successfully for video_asset_id=%s location_id=%s",
                        job.video_asset_id,
                        job.location_id,
                    )
                else:
                    logger.warning(
                        "Analysis job completed with issue for video_asset_id=%s location_id=%s stderr=%s",
                        job.video_asset_id,
                        job.location_id,
                        (result.stderr or "")[:500],
                    )
            except Exception:
                logger.exception("Analysis job crashed for video_asset_id=%s", job.video_asset_id)
            finished_ids.append(video_asset_id)
            finished_count += 1
        if not finished_ids:
            return
        with self._lock:
            for video_asset_id in finished_ids:
                self._running.pop(video_asset_id, None)
        cooldown_seconds = max(0, settings.analysis_post_job_sleep_seconds)
        if finished_count > 0 and cooldown_seconds > 0:
            logger.info(
                "Analysis worker cooldown started for %ss after %s finished job(s)",
                cooldown_seconds,
                finished_count,
            )
            time.sleep(cooldown_seconds)

    def _fill_available_slots(self) -> None:
        with self._lock:
            running_jobs = list(self._running.values())
        available_slots = max(0, settings.analysis_max_global_workers - len(running_jobs))
        if available_slots <= 0:
            return

        running_by_location: dict[int, int] = {}
        for job in running_jobs:
            running_by_location[job.location_id] = running_by_location.get(job.location_id, 0) + 1

        db = TransactionalSessionLocal()
        try:
            if repositories.is_worker_paused(db, "analysis"):
                return
            for row in repositories.list_running_video_asset_analyses(db):
                location_id = row.get("location_id")
                if location_id is None:
                    continue
                running_by_location[int(location_id)] = max(running_by_location.get(int(location_id), 0), 1)

            candidates = repositories.list_pending_video_asset_analyses(
                db,
                limit=max(settings.analysis_max_global_workers * 20, 50),
            )
            for candidate in candidates:
                if available_slots <= 0:
                    break
                location_id = candidate.get("location_id")
                if location_id is None:
                    continue
                location_id = int(location_id)
                if running_by_location.get(location_id, 0) >= max(1, settings.analysis_max_per_location):
                    continue
                video_asset_id = int(candidate["id"])
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
                        trigger_id=int(candidate["trigger_id"]) if candidate.get("trigger_id") is not None else None,
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
                running_by_location[location_id] = running_by_location.get(location_id, 0) + 1
                available_slots -= 1
                logger.info("Claimed analysis job video_asset_id=%s location_id=%s", video_asset_id, location_id)
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
