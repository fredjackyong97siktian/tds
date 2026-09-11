from __future__ import annotations

import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from threading import Lock

from ..config import settings
from ..db import TransactionalSessionLocal
from .. import repositories
from ..services import workflow_service


logger = logging.getLogger("tds.video_retrieval_worker")


@dataclass
class RunningJob:
    process: subprocess.Popen
    location_id: int
    asset_id: int
    asset_kind: str
    started_at: float = field(default_factory=time.monotonic)


class VideoRetrievalWorker:
    # Exactly one ffmpeg process runs at a time, full stop - frame retrieval
    # (quick per-trigger snapshots feeding grouping) and video retrieval (full
    # clips for L1 entrance/kiosk analysis) share this single slot rather than
    # each getting their own budget. Video candidates are checked first each
    # cycle (see _fill_available_slots) so an entrance/kiosk analysis waiting
    # on its video doesn't sit behind a pile of new-trigger frame jobs.
    #
    # This used to submit jobs to a ThreadPoolExecutor - a stalled ffmpeg/RTSP
    # call (exactly the kind of thing an NVR under load can produce) occupied
    # that thread forever, with no way to force it out. There WAS already a
    # DB-level staleness reset (reset_stale_trigger_frame_asset_retrievals/
    # reset_stale_video_asset_retrievals), but it lived inside the same
    # "if available_slots <= 0: return" early-out as everything else, so it
    # never ran precisely when every slot was stuck and a reset was needed
    # most. Running each job as its own OS subprocess means it can always be
    # force-killed after a timeout, same fix already applied to the grouping
    # and theft-confidence workers.
    def __init__(self) -> None:
        self._max_workers = max(1, settings.retrieval_max_global_workers)
        self._stale_seconds = max(60, settings.retrieval_stale_seconds)
        self._running: dict[str, RunningJob] = {}
        self._lock = Lock()

    def run_forever(self) -> None:
        poll_seconds = max(1, settings.retrieval_poll_seconds)
        logger.info(
            "Video retrieval worker started with poll=%ss max_global=%s max_per_location=%s stale_timeout=%ss",
            poll_seconds,
            self._max_workers,
            settings.retrieval_max_per_location,
            self._stale_seconds,
        )
        while True:
            try:
                self._reap_finished_jobs()
                self._kill_stale_jobs()
                self._fill_available_slots()
            except Exception:
                logger.exception("Video retrieval worker loop failed")
            time.sleep(poll_seconds)

    def _reap_finished_jobs(self) -> None:
        finished_ids: list[str] = []
        with self._lock:
            items = list(self._running.items())
        for running_key, job in items:
            exit_code = job.process.poll()
            if exit_code is None:
                continue
            if exit_code == 0:
                logger.info("Retrieval job completed for %s_id=%s location_id=%s", job.asset_kind, job.asset_id, job.location_id)
            else:
                logger.error(
                    "Retrieval job process exited non-zero for %s_id=%s location_id=%s exit_code=%s",
                    job.asset_kind,
                    job.asset_id,
                    job.location_id,
                    exit_code,
                )
            finished_ids.append(running_key)
        if not finished_ids:
            return
        with self._lock:
            for running_key in finished_ids:
                self._running.pop(running_key, None)

    def _kill_stale_jobs(self) -> None:
        now = time.monotonic()
        with self._lock:
            items = list(self._running.items())
        for running_key, job in items:
            if now - job.started_at < self._stale_seconds:
                continue
            logger.error(
                "Retrieval job stuck for over %ss, killing %s_id=%s location_id=%s pid=%s",
                self._stale_seconds,
                job.asset_kind,
                job.asset_id,
                job.location_id,
                job.process.pid,
            )
            job.process.kill()
            job.process.wait()
            with self._lock:
                self._running.pop(running_key, None)

    def _fill_available_slots(self) -> None:
        db = TransactionalSessionLocal()
        try:
            if repositories.is_worker_paused(db, "retrieval"):
                return
            # Always run this, regardless of slot availability - it's exactly
            # when every slot looks busy that a stuck DB row most needs
            # resetting, and a subprocess kill above only clears this
            # worker's own bookkeeping, not a stale row some earlier crashed
            # process left behind.
            stale_count = repositories.reset_stale_trigger_frame_asset_retrievals(
                db,
                settings.retrieval_stale_seconds,
            )
            if stale_count:
                logger.warning("Reset %s stale trigger frame retrieval job(s)", stale_count)
            stale_video_count = repositories.reset_stale_video_asset_retrievals(
                db,
                settings.retrieval_stale_seconds,
            )
            if stale_video_count:
                logger.warning("Reset %s stale video asset retrieval job(s)", stale_video_count)

            with self._lock:
                running_jobs = list(self._running.values())
            available_slots = max(0, self._max_workers - len(running_jobs))
            if available_slots <= 0:
                return

            running_by_location: dict[int, int] = {}
            for job in running_jobs:
                running_by_location[job.location_id] = running_by_location.get(job.location_id, 0) + 1

            # Include already-running rows from DB so a restarted worker does not double-book a location.
            for row in repositories.list_running_video_asset_retrievals(db):
                location_id = row.get("location_id")
                if location_id is None:
                    continue
                running_by_location[int(location_id)] = max(
                    running_by_location.get(int(location_id), 0),
                    1,
                )
            for row in repositories.list_running_trigger_frame_asset_retrievals(db):
                location_id = row.get("location_id")
                if location_id is None:
                    continue
                running_by_location[int(location_id)] = max(
                    running_by_location.get(int(location_id), 0),
                    1,
                )

            # Video candidates (entrance/kiosk L1 analysis) are claimed before frame
            # candidates so an already-queued analysis video doesn't wait behind a
            # burst of new-trigger frame jobs for the one available slot.
            video_candidates = repositories.list_pending_video_asset_retrievals(
                db,
                limit=max(self._max_workers * 10, 20),
            )
            for candidate in video_candidates:
                if available_slots <= 0:
                    break
                location_id = candidate.get("location_id")
                if location_id is None:
                    continue
                location_id = int(location_id)
                if running_by_location.get(location_id, 0) >= max(1, settings.retrieval_max_per_location):
                    continue
                video_asset_id = int(candidate["id"])
                running_key = f"video:{video_asset_id}"
                if running_key in self._running:
                    continue
                claimed = repositories.claim_video_asset_for_retrieval(db, video_asset_id)
                if not claimed:
                    continue

                process = subprocess.Popen(
                    [sys.executable, "-m", "app.workers.run_retrieval_job", "video", str(video_asset_id)]
                )
                with self._lock:
                    self._running[running_key] = RunningJob(
                        process=process,
                        location_id=location_id,
                        asset_id=video_asset_id,
                        asset_kind="video_asset",
                    )
                running_by_location[location_id] = running_by_location.get(location_id, 0) + 1
                available_slots -= 1
                logger.info("Claimed retrieval job video_asset_id=%s location_id=%s pid=%s", video_asset_id, location_id, process.pid)

            frame_candidates = repositories.list_pending_trigger_frame_asset_retrievals(
                db,
                limit=max(self._max_workers * 10, 20),
            )
            for candidate in frame_candidates:
                if available_slots <= 0:
                    break
                location_id = candidate.get("location_id")
                if location_id is None:
                    continue
                location_id = int(location_id)
                if running_by_location.get(location_id, 0) >= max(1, settings.retrieval_max_per_location):
                    continue
                frame_asset_id = int(candidate["id"])
                running_key = f"frame:{frame_asset_id}"
                if running_key in self._running:
                    continue
                claimed = repositories.claim_trigger_frame_asset_for_retrieval(db, frame_asset_id)
                if not claimed:
                    continue

                process = subprocess.Popen(
                    [sys.executable, "-m", "app.workers.run_retrieval_job", "frame", str(frame_asset_id)]
                )
                with self._lock:
                    self._running[running_key] = RunningJob(
                        process=process,
                        location_id=location_id,
                        asset_id=frame_asset_id,
                        asset_kind="frame_asset",
                    )
                running_by_location[location_id] = running_by_location.get(location_id, 0) + 1
                available_slots -= 1
                logger.info("Claimed frame retrieval job frame_asset_id=%s location_id=%s pid=%s", frame_asset_id, location_id, process.pid)
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
