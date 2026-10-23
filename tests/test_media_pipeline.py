from __future__ import annotations

import copy
import os
import tempfile
import types
import unittest
import importlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from app.services import confidence_media as media
from app.services import media_retention as retention

UTC = timezone.utc


class RetentionPolicyTests(unittest.TestCase):
    def test_theft_group_protection_includes_all_exits_and_running_grouping_inputs(self):
        def rows(db, sql, params=None):
            if "select s.id" in sql:
                return [{"id": 7, "location_id": 2, "entry_trigger_id": 1, "exit_trigger_id": 2, "grouping_id": 3}]
            if "entry_item" in sql:
                return [{"trigger_id": 1}, {"trigger_id": 2}, {"trigger_id": 3}]
            if "select video_asset_id" in sql:
                return [{"video_asset_id": 9}]
            if "where b.status" in sql:
                return [{"trigger_id": 4}]
            return []
        with patch.object(retention, "_rows", side_effect=rows), patch.object(retention.repositories, "_column_exists", return_value=False):
            sessions, triggers, videos = retention.protected_owners(Mock(), datetime(2026, 9, 17))
        self.assertEqual(sessions, {7})
        self.assertEqual(triggers, {1, 2, 3, 4})
        self.assertEqual(videos, {9})

    def test_daily_worker_does_not_repeat_a_successful_day(self):
        from app.workers import cleaner_worker
        db = MagicMock()
        db.get_bind.return_value.connect.return_value.__enter__.return_value.execute.return_value.scalar.return_value = 1
        with patch.object(cleaner_worker, "TransactionalSessionLocal", return_value=db), patch.object(cleaner_worker.repositories, "get_app_setting", return_value='{"date":"2026-10-17"}'), patch.object(cleaner_worker, "run_cleanup") as run:
            cleaner_worker.tick(datetime(2026, 10, 16, 20, tzinfo=UTC))
            run.assert_not_called()
            db.close.assert_called_once()

    def test_calendar_month_boundaries(self):
        self.assertEqual(retention.month_ago(datetime(2024, 3, 31, tzinfo=UTC)), datetime(2024, 2, 29, tzinfo=UTC))
        self.assertEqual(retention.month_ago(datetime(2026, 1, 31, tzinfo=UTC)), datetime(2025, 12, 31, tzinfo=UTC))

    def test_schedule_uses_malaysia_not_server_timezone(self):
        self.assertIsNone(retention.scheduled_date(datetime(2026, 10, 16, 19, 59, tzinfo=UTC)))
        self.assertEqual(retention.scheduled_date(datetime(2026, 10, 16, 20, tzinfo=UTC)), "2026-10-17")

    def test_deletion_targets_are_scoped(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(retention.settings, "video_storage_dir", Path(directory)), patch.object(retention, "is_spaces_configured", return_value=True), patch.object(retention, "generate_public_object_url", return_value="https://bucket.example/"), patch.object(retention.settings, "spaces_key_prefix", "tds"):
            local, remote = retention.media_references([
                f"{directory}/locations/trigger_1/frame.jpg", "/etc/passwd",
                "https://bucket.example/tds/location_1/trigger_1/frame.jpg?signature=x",
                "https://unrelated.example/tds/location_1/frame.jpg", "spaces://another-app/frame.jpg",
                "spaces://tds/../frame.jpg", "spaces://tds/location_1/manifest.json",
            ])
            self.assertEqual(local, {(Path(directory) / "locations/trigger_1/frame.jpg").resolve()})
            self.assertEqual(remote, {"tds/location_1/trigger_1/frame.jpg"})

    def run_sample_cleanup(self, dry_run):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        old = datetime(2026, 8, 1, tzinfo=UTC)
        now = datetime(2026, 10, 17, tzinfo=UTC)
        paths = [root / "locations/location_1" / f"trigger_{value}/entrance/raw/video.mp4" for value in (10, 20, 30)]
        for index, path in enumerate(paths):
            path.parent.mkdir(parents=True)
            path.write_bytes(b"test video")
            stamp = (old if index < 2 else now).timestamp()
            os.utime(path, (stamp, stamp))
        videos = [{"id": index + 1, "trigger_id": trigger, "created_at": old.replace(tzinfo=None),
                   "status": "ready", "file_path": str(path), "video_url": "", "metadata": {}}
                  for index, (trigger, path) in enumerate(zip((10, 20, 30), paths))]
        frames = [{"id": 1, "trigger_id": 10, "image_url": "https://bucket.example/tds/location_1/trigger_10/entrance/frames/a.jpg"},
                  {"id": 2, "trigger_id": 20, "image_url": "https://bucket.example/tds/location_1/trigger_20/entrance/frames/b.jpg"}]
        def rows(db, sql, params=None):
            if "select * from tds_video_asset" in sql:
                return copy.deepcopy(videos)
            if "select f.*" in sql:
                return frames if params["id"] == 0 else []
            return []
        db = Mock()
        db.execute.return_value.mappings.return_value.all.return_value = []
        client = Mock()
        client.head_object.return_value = {"LastModified": old}
        with patch.object(retention.settings, "video_storage_dir", root), patch.object(retention, "protected_owners", return_value=(set(), {20}, {2})), patch.object(retention, "_rows", side_effect=rows), patch.object(retention, "is_spaces_configured", return_value=True), patch.object(retention, "_spaces_client", return_value=client), patch.object(retention, "generate_public_object_url", return_value="https://bucket.example/"):
            result = retention.run_cleanup(db, now=now, dry_run=dry_run)
        self.assertTrue(paths[1].exists(), "Theft evidence must survive")
        self.assertTrue(paths[2].exists(), "Newly written files must survive even on old DB rows")
        self.assertEqual(paths[0].exists(), dry_run)
        self.assertEqual(result["videos_expired"], 1)
        self.assertEqual(result["frames_expired"], 1)
        if dry_run:
            client.delete_object.assert_not_called()
            self.assertTrue(all(str(call.args[0]).lstrip().lower().startswith("select") for call in db.execute.call_args_list))
        else:
            self.assertEqual(client.delete_object.call_count, 1)
            self.assertNotIn("trigger_20", str(client.delete_object.call_args))
        return result

    def test_cleanup_preserves_theft_and_recent_media(self):
        self.assertEqual(self.run_sample_cleanup(False)["errors"], 0)

    def test_dry_run_never_deletes_files_or_updates_rows(self):
        self.assertEqual(self.run_sample_cleanup(True)["errors"], 0)

    def test_shared_theft_object_is_not_removed_through_another_video_row(self):
        old = datetime(2026, 8, 1)
        videos = [{"id": key, "trigger_id": None, "created_at": old, "status": "ready",
                   "file_path": "spaces://tds/shared.mp4", "video_url": "", "metadata": {}} for key in (1, 2)]
        def rows(db, sql, params=None):
            return videos if "select * from tds_video_asset" in sql else []
        with tempfile.TemporaryDirectory() as directory, patch.object(retention.settings, "video_storage_dir", Path(directory)), patch.object(retention, "protected_owners", return_value=(set(), set(), {2})), patch.object(retention, "_rows", side_effect=rows), patch.object(retention, "is_spaces_configured", return_value=True), patch.object(retention, "_spaces_client") as client, patch.object(retention, "generate_public_object_url", return_value="https://bucket.example/"):
            stats = retention.run_cleanup(Mock(), now=datetime(2026, 10, 17, tzinfo=UTC))
            self.assertEqual(stats["videos_expired"], 0)
            client.return_value.delete_object.assert_not_called()

    def test_cloud_delete_failure_keeps_the_media_row_retryable(self):
        video = {"id": 1, "trigger_id": 10, "created_at": datetime(2026, 8, 1), "status": "ready",
                 "file_path": "spaces://tds/old.mp4", "metadata": {}, "video_url": ""}
        def rows(db, sql, params=None):
            return [video] if "select * from tds_video_asset" in sql else []
        client = Mock()
        client.head_object.return_value = {"LastModified": datetime(2026, 8, 1, tzinfo=UTC)}
        client.delete_object.side_effect = RuntimeError("storage unavailable")
        with tempfile.TemporaryDirectory() as directory, patch.object(retention.settings, "video_storage_dir", Path(directory)), patch.object(retention, "protected_owners", return_value=(set(), set(), set())), patch.object(retention, "_rows", side_effect=rows), patch.object(retention, "is_spaces_configured", return_value=True), patch.object(retention, "_spaces_client", return_value=client), patch.object(retention, "generate_public_object_url", return_value="https://bucket.example/"), self.assertLogs(retention.logger, level="ERROR"):
            stats = retention.run_cleanup(Mock(), now=datetime(2026, 10, 17, tzinfo=UTC))
            self.assertEqual(stats["videos_expired"], 0)
            self.assertEqual(stats["errors"], 1)


class ConfidenceMediaTests(unittest.TestCase):
    def setUp(self):
        self.detection_patch = patch("app.services.detection_analysis.ENABLED", False)
        self.detection_patch.start()
        self.addCleanup(self.detection_patch.stop)
        self.db = Mock()
        self.start = datetime(2026, 10, 17, 10)
        self.triggers = {1: {"id": 1, "location_id": 2, "trigger_time": self.start},
                         2: {"id": 2, "location_id": 2, "trigger_time": self.start + timedelta(minutes=5)},
                         3: {"id": 3, "location_id": 2, "trigger_time": self.start + timedelta(minutes=7)}}

    def test_required_windows_include_all_exits_and_kiosk_without_transactions(self):
        windows = media.required_windows(self.triggers, [1], [2, 3])
        self.assertEqual([window["role"] for window in windows], ["entry", "exit", "exit", "kiosk"])
        self.assertEqual(windows[0]["start"], self.start - timedelta(seconds=40))
        self.assertEqual(windows[0]["end"], self.start + timedelta(seconds=10))
        self.assertEqual(windows[-1]["end"], self.start + timedelta(minutes=7))

    def test_incomplete_and_backwards_groups_are_rejected(self):
        with self.assertRaises(ValueError):
            media.required_windows(self.triggers, [1], [])
        with self.assertRaises(ValueError):
            media.required_windows(self.triggers, [2], [1])

    def test_queue_safe_group_reuses_ready_assets_without_retrieving_again(self):
        row = {"id": 8, "batch_id": 3, "location_id": 2, "need_deep_analysis": 0,
               "factor_payload": {"entry_trigger_ids": [1], "exit_trigger_ids": [2]}}
        saves = []
        with patch.object(media, "_save", side_effect=lambda db, result_id, plan: saves.append(copy.deepcopy(plan))), patch.object(media.repositories, "get_trigger", side_effect=lambda db, key: self.triggers[key]), patch.object(media.repositories, "find_video_asset_by_window", side_effect=[{"id": 10, "status": "ready"}, {"id": 11, "status": "ready"}, {"id": 12, "status": "ready"}]), patch.object(media.repositories, "create_video_asset") as create:
            self.assertEqual(media._queue(self.db, row), [10, 11, 12])
            create.assert_not_called()
            self.assertEqual(saves[-1]["status"], "pending")
            self.assertEqual(len(saves[-1]["links"]), 3)

    def sample_row(self, deep=False):
        return {"id": 8, "batch_id": 3, "location_id": 2, "need_deep_analysis": deep,
                "factor_payload": {"entry_trigger_ids": [1], "exit_trigger_ids": [2],
                                   "media_retrieval": {"status": "pending", "links": [
                                       {"video_asset_id": key, "section": role, "start": self.start.isoformat(), "end": (self.start + timedelta(minutes=5)).isoformat()}
                                       for key, role in ((10, "entry"), (11, "exit"), (12, "kiosk"))]}}}

    def finish(self, *, failed=False, deep=False, paused=False):
        from app.services import session_analysis
        row = self.sample_row(deep)
        assets = [{"id": key, "status": "issue" if failed and key == 12 else "ready",
                   "retrieved_at": None if failed and key == 12 else self.start,
                   "metadata": {"retrieval_source": "confidence_evidence", "analysis_eligible": False}} for key in (10, 11, 12)]
        workflow = types.ModuleType("app.services.workflow_service")
        workflow._ensure_session_for_confidence_group = Mock(return_value={"id": 7})
        saved = []
        import app.services
        with patch.object(session_analysis, "PAUSED", paused), patch.object(session_analysis, "no_transaction_check", return_value=None) as no_purchase, patch.dict("sys.modules", {"app.services.workflow_service": workflow}), patch.object(app.services, "workflow_service", workflow, create=True), patch.object(media.repositories, "get_video_asset", side_effect=assets), patch.object(media.repositories, "create_session_video_asset_link") as link, patch.object(media.repositories, "update_session_grouping_link"), patch.object(media.repositories, "update_video_asset_status") as update, patch.object(media, "_save", side_effect=lambda db, result_id, plan: saved.append(copy.deepcopy(plan))):
            media._finish(self.db, row)
            if failed:
                workflow._ensure_session_for_confidence_group.assert_not_called()
                link.assert_not_called()
                self.assertEqual(saved[-1]["status"], "waiting")
                self.assertEqual(saved[-1]["issue_video_asset_ids"], [12])
            elif paused or not deep:
                workflow._ensure_session_for_confidence_group.assert_not_called()
                no_purchase.assert_not_called()
                link.assert_not_called()
                update.assert_not_called()
                self.assertEqual(saved[-1]["status"], "complete")
                self.assertEqual(saved[-1]["session_analysis_status"], "paused" if paused else "not_required")
            else:
                workflow._ensure_session_for_confidence_group.assert_called_once()
                self.assertEqual(link.call_count, 3)
                self.assertEqual(saved[-1]["status"], "complete")
                self.assertEqual(saved[-1]["session_id"], 7)
                self.assertTrue(all(call.kwargs["metadata"]["analysis_eligible"] == deep for call in update.call_args_list))

    def test_failed_kiosk_blocks_session_creation(self):
        self.finish(failed=True)

    def test_safe_group_keeps_videos_without_creating_session(self):
        self.finish(deep=False)

    def test_paused_flagged_group_keeps_videos_without_session_or_analysis(self):
        self.finish(deep=True, paused=True)

    def test_flagged_group_enables_analysis_after_retrieval(self):
        self.finish(deep=True)

    def test_completed_batch_resume_returns_summary_without_reanalysis(self):
        workflow = importlib.import_module("app.services.workflow_service")
        batch = {"id": 3, "location_id": 2, "result_payload": {"groups": [{"group_id": 1, "entry": [1], "exit": [2]}]}}
        with patch.object(workflow.repositories, "get_grouping_batch", return_value=batch), patch.object(workflow, "_load_filter_factor_settings", return_value={}), patch.object(workflow.repositories, "list_scored_group_keys_for_batch", return_value={"1"}), patch.object(workflow.settings, "grouping_repair_enabled", False):
            result = workflow._run_theft_confidence_for_grouping_batch_locked(self.db, batch_id=3)
            self.assertEqual(result["analyzed_count"], 0)
            self.assertEqual(result["session_count"], 0)
            self.assertEqual(result["queued_video_asset_ids"], [])


if __name__ == "__main__":
    unittest.main()
