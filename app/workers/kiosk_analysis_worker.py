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


logger = logging.getLogger("tds.kiosk_analysis_worker")


@dataclass
class RunningJob:
    future: Future[workflow_service.ScriptExecutionResult]
    location_id: int
    session_id: int
    video_asset_ids: list[int]


class KioskAnalysisWorker:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max(1, settings.kiosk_analysis_max_global_workers))
        self._running: dict[int, RunningJob] = {}
        self._lock = Lock()
        self._next_dispatch_after = 0.0

    def run_forever(self) -> None:
        poll_seconds = max(1, settings.kiosk_analysis_poll_seconds)
        logger.info(
            "Kiosk analysis worker started with poll=%ss max_global=%s cooldown=%ss",
            poll_seconds,
            settings.kiosk_analysis_max_global_workers,
            settings.kiosk_analysis_cooldown_seconds,
        )
        while True:
            try:
                self._reap_finished_jobs()
                self._fill_available_slots()
            except Exception:
                logger.exception("Kiosk analysis worker loop failed")
            time.sleep(poll_seconds)

    def _reap_finished_jobs(self) -> None:
        finished_ids: list[int] = []
        cooldown_seconds = max(0, settings.kiosk_analysis_cooldown_seconds)
        with self._lock:
            items = list(self._running.items())
        for session_id, job in items:
            if not job.future.done():
                continue
            try:
                result = job.future.result()
                logger.info(
                    "Kiosk analysis dispatch finished for session_id=%s video_asset_ids=%s location_id=%s status=%s runner_job_id=%s",
                    job.session_id,
                    job.video_asset_ids,
                    job.location_id,
                    result.status,
                    result.runner_job_id,
                )
            except Exception:
                logger.exception("Kiosk analysis dispatch crashed for session_id=%s video_asset_ids=%s", job.session_id, job.video_asset_ids)
            finished_ids.append(session_id)
        if not finished_ids:
            return
        with self._lock:
            for session_id in finished_ids:
                self._running.pop(session_id, None)
            self._next_dispatch_after = time.time() + cooldown_seconds

    def _fill_available_slots(self) -> None:
        now = time.time()
        with self._lock:
            running_jobs = list(self._running.values())
            next_dispatch_after = self._next_dispatch_after
        if now < next_dispatch_after:
            return

        available_slots = max(0, settings.kiosk_analysis_max_global_workers - len(running_jobs))
        if available_slots <= 0:
            return

        db = TransactionalSessionLocal()
        try:
            reconciled = workflow_service.reconcile_running_remote_analysis_script_runs(db)
            for item in reconciled:
                logger.info(
                    "Reconciled remote kiosk analysis script_run_id=%s runner_job_id=%s script=%s runpod_status=%s status=%s",
                    item["script_run_id"],
                    item["runner_job_id"],
                    item["script_name"],
                    item["runpod_status"],
                    item["status"],
                )
            orphaned_ids = repositories.reset_orphaned_processing_kiosk_video_assets(db)
            for video_asset_id in orphaned_ids:
                logger.warning(
                    "Reset orphaned kiosk video_asset_id=%s stuck at status='processing' with no running script_run behind it",
                    video_asset_id,
                )
            if repositories.is_worker_paused(db, "kiosk_analysis"):
                return
            if repositories.has_active_remote_analysis_script_run(db, script_names=["kiosk"]):
                return
            if repositories.list_running_video_asset_analyses(db, sections=["kiosk"]):
                return

            candidates = repositories.list_pending_kiosk_video_asset_analyses(
                db,
                limit=max(settings.kiosk_analysis_max_global_workers * 20, 20),
            )
            if not candidates:
                return

            # Every ready kiosk video for the SAME session gets dispatched
            # together as one RunPod job (one call, one Gemini pass covering
            # all of them) instead of one job per video - a session can have
            # more than one kiosk video, one per non-overlapping paid-
            # transaction window. Candidates are already ordered by
            # captured_start_time, so the first session_id encountered is the
            # oldest one waiting; batch only that session's videos this cycle.
            first_session_id = int(candidates[0]["session_id"])
            session_candidates = [c for c in candidates if int(c["session_id"]) == first_session_id]
            location_id = int(session_candidates[0]["location_id"])

            claimed_video_asset_ids: list[int] = []
            for candidate in session_candidates:
                video_asset_id = int(candidate["id"])
                if repositories.claim_video_asset_for_analysis(db, video_asset_id):
                    claimed_video_asset_ids.append(video_asset_id)
            if not claimed_video_asset_ids:
                return

            try:
                job = workflow_service.build_kiosk_analysis_job_for_videos(db, claimed_video_asset_ids)
                future = self._executor.submit(workflow_service.start_kiosk_analysis_job, job)
            except Exception as exc:
                logger.exception(
                    "Could not build kiosk analysis job for session_id=%s video_asset_ids=%s",
                    first_session_id,
                    claimed_video_asset_ids,
                )
                for video_asset_id in claimed_video_asset_ids:
                    repositories.update_video_asset_status(db, video_asset_id, "issue")
                repositories.create_script_run(
                    db,
                    session_id=first_session_id,
                    trigger_id=None,
                    script_name="kiosk",
                    model_name="worker_build_job",
                    status="failed",
                    command="worker_build_job",
                    stdout_log="",
                    stderr_log=str(exc),
                )
                return

            with self._lock:
                self._running[first_session_id] = RunningJob(
                    future=future,
                    location_id=location_id,
                    session_id=first_session_id,
                    video_asset_ids=claimed_video_asset_ids,
                )
            logger.info(
                "Claimed kiosk analysis dispatch session_id=%s video_asset_ids=%s location_id=%s",
                first_session_id,
                claimed_video_asset_ids,
                location_id,
            )
        finally:
            db.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    worker = KioskAnalysisWorker()
    worker.run_forever()


if __name__ == "__main__":
    main()
