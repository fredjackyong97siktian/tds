import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

from app.services import workflow_service as workflow


class RetrievalLockTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.now = 0.0
        self.conn = Mock()
        self.stack.enter_context(patch.object(workflow.transactional_engine, "connect", return_value=self.conn))
        self.stack.enter_context(patch.object(workflow.time, "monotonic", side_effect=lambda: self.now))
        self.sleep = self.stack.enter_context(patch.object(workflow.time, "sleep", side_effect=self.advance))

    def advance(self, seconds):
        self.now += seconds

    def availability(self, predicate):
        def execute(*args, **kwargs):
            result = Mock()
            result.mappings.return_value.first.return_value = {"acquired": int(predicate())}
            return result
        self.conn.execute.side_effect = execute

    def test_idle_slot_is_acquired_immediately_on_the_connection_that_is_returned(self):
        self.availability(lambda: True)
        self.assertIs(workflow._acquire_retrieval_ffmpeg_lock(wait_seconds=180), self.conn)
        self.sleep.assert_not_called()
        self.conn.close.assert_not_called()

    def test_busy_retrieval_can_take_more_than_mysql_read_timeout_without_long_sql_wait(self):
        self.availability(lambda: self.now >= 61)
        self.assertIs(workflow._acquire_retrieval_ffmpeg_lock(wait_seconds=180), self.conn)
        self.assertEqual(self.now, 61)
        self.assertTrue(all(call.args[1]["wait_seconds"] == 0 for call in self.conn.execute.call_args_list))
        self.conn.close.assert_not_called()

    def test_timeout_closes_connection_and_never_claims_the_busy_slot(self):
        self.availability(lambda: False)
        self.assertIsNone(workflow._acquire_retrieval_ffmpeg_lock(wait_seconds=180))
        self.assertEqual(self.now, 180)
        self.conn.close.assert_called_once()

    def test_zero_wait_only_checks_once(self):
        self.availability(lambda: False)
        self.assertIsNone(workflow._acquire_retrieval_ffmpeg_lock(wait_seconds=0))
        self.conn.execute.assert_called_once()
        self.sleep.assert_not_called()

    def test_database_failure_closes_connection(self):
        self.conn.execute.side_effect = RuntimeError("database unavailable")
        with self.assertRaisesRegex(RuntimeError, "database unavailable"):
            workflow._acquire_retrieval_ffmpeg_lock(wait_seconds=180)
        self.conn.close.assert_called_once()

    def test_null_lock_result_is_an_error_not_an_idle_slot(self):
        self.conn.execute.return_value.mappings.return_value.first.return_value = {"acquired": None}
        with self.assertRaisesRegex(RuntimeError, "could not check"):
            workflow._acquire_retrieval_ffmpeg_lock(wait_seconds=180)
        self.conn.close.assert_called_once()

    def test_release_uses_same_connection_and_closes_it_even_on_error(self):
        self.conn.execute.side_effect = RuntimeError("database unavailable")
        with self.assertRaises(RuntimeError):
            workflow._release_retrieval_ffmpeg_lock(self.conn)
        self.assertIn("release_lock", str(self.conn.execute.call_args.args[0]))
        self.conn.close.assert_called_once()

    def test_frame_worker_cannot_bypass_the_shared_slot(self):
        with patch.object(workflow, "_acquire_retrieval_ffmpeg_lock", return_value=None), \
             patch.object(workflow, "_run_trigger_frame_asset_retrieval_job") as run:
            with self.assertRaisesRegex(RuntimeError, "exclusive ffmpeg/NVR"):
                workflow.start_trigger_frame_asset_retrieval_job(Mock())
        run.assert_not_called()

    def test_frame_worker_releases_slot_even_if_retrieval_fails(self):
        job, lock = Mock(), Mock()
        with patch.object(workflow, "_acquire_retrieval_ffmpeg_lock", return_value=lock), \
             patch.object(workflow, "_release_retrieval_ffmpeg_lock") as release, \
             patch.object(workflow, "_run_trigger_frame_asset_retrieval_job", side_effect=RuntimeError("capture failed")) as run:
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                workflow.start_trigger_frame_asset_retrieval_job(job)
        run.assert_called_once_with(job)
        release.assert_called_once_with(lock)

    def test_video_worker_cannot_bypass_the_shared_slot(self):
        with patch.object(workflow, "_acquire_retrieval_ffmpeg_lock", return_value=None), \
             patch.object(workflow, "_run_video_retrieval_job") as run:
            with self.assertRaisesRegex(RuntimeError, "exclusive ffmpeg/NVR"):
                workflow.start_video_retrieval_job(Mock())
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
