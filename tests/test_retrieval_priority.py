from contextlib import ExitStack
import unittest
from unittest.mock import Mock, patch

from app.workers import video_retrieval_worker as retrieval


class RetrievalPriorityTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.db = Mock()
        self.stack.enter_context(patch.object(retrieval, "TransactionalSessionLocal", return_value=self.db))
        self.stack.enter_context(patch.object(retrieval.confidence_media, "finish_pending"))
        self.stack.enter_context(patch.object(retrieval.settings, "retrieval_max_per_location", 1))
        self.paused = self.stack.enter_context(patch.object(retrieval.repositories, "is_worker_paused", return_value=False))
        self.calls = {}
        for name, value in {
            "reset_stale_trigger_frame_asset_retrievals": 0,
            "reset_stale_video_asset_retrievals": 0,
            "list_running_video_asset_retrievals": [],
            "list_running_trigger_frame_asset_retrievals": [],
            "list_pending_video_asset_retrievals": [{"id": 40, "location_id": 2}],
            "list_pending_trigger_frame_asset_retrievals": [{"id": 90, "location_id": 2}],
            "claim_video_asset_for_retrieval": True,
            "claim_trigger_frame_asset_for_retrieval": True,
        }.items():
            self.calls[name] = self.stack.enter_context(patch.object(retrieval.repositories, name, return_value=value))
        self.launch = self.stack.enter_context(patch.object(retrieval.subprocess, "Popen", return_value=Mock(pid=1)))
        self.worker = retrieval.VideoRetrievalWorker()
        self.worker._max_workers = 1

    def test_frame_is_claimed_before_a_video_at_same_location(self):
        self.worker._fill_available_slots()
        self.assertEqual(self.launch.call_args.args[0][-2:], ["frame", "90"])
        self.calls["claim_video_asset_for_retrieval"].assert_not_called()
        self.assertIn("frame:90", self.worker._running)

    def test_video_runs_when_no_frames_are_waiting(self):
        self.calls["list_pending_trigger_frame_asset_retrievals"].return_value = []
        self.worker._fill_available_slots()
        self.assertEqual(self.launch.call_args.args[0][-2:], ["video", "40"])

    def test_video_at_another_location_can_use_remaining_slot(self):
        self.worker._max_workers = 2
        self.calls["list_pending_video_asset_retrievals"].return_value = [{"id": 40, "location_id": 3}]
        self.worker._fill_available_slots()
        self.assertEqual([call.args[0][-2:] for call in self.launch.call_args_list], [["frame", "90"], ["video", "40"]])

    def test_two_slots_do_not_double_book_frame_and_video_at_same_location(self):
        self.worker._max_workers = 2
        self.worker._fill_available_slots()
        self.launch.assert_called_once()
        self.calls["claim_video_asset_for_retrieval"].assert_not_called()

    def test_running_video_is_not_interrupted_or_double_booked(self):
        process = Mock()
        self.worker._running["video:40"] = retrieval.RunningJob(process, 2, 40, "video_asset")
        self.worker._fill_available_slots()
        self.launch.assert_not_called()
        process.kill.assert_not_called()
        self.calls["reset_stale_trigger_frame_asset_retrievals"].assert_called_once()

    def test_database_running_job_reserves_location_after_worker_restart(self):
        self.calls["list_running_video_asset_retrievals"].return_value = [{"id": 40, "location_id": 2}]
        self.worker._fill_available_slots()
        self.launch.assert_not_called()

    def test_failed_frame_claim_does_not_block_other_work(self):
        self.calls["claim_trigger_frame_asset_for_retrieval"].return_value = False
        self.worker._fill_available_slots()
        self.assertEqual(self.launch.call_args.args[0][-2:], ["video", "40"])

    def test_paused_worker_starts_nothing(self):
        self.paused.return_value = True
        self.worker._fill_available_slots()
        self.launch.assert_not_called()
        self.db.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
