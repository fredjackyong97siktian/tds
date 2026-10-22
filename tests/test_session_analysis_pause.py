import unittest
from unittest.mock import Mock, patch

from app.services import jev_pipeline, session_analysis, workflow_service
from app.workers.video_analysis_worker import VideoAnalysisWorker


class SessionAnalysisPauseTests(unittest.TestCase):
    def test_pause_is_enabled_by_default(self):
        self.assertTrue(session_analysis.PAUSED)

    def test_paused_worker_does_not_claim_any_analysis(self):
        worker = Mock()
        with patch.object(session_analysis, "PAUSED", True), patch("app.workers.video_analysis_worker.TransactionalSessionLocal") as connect:
            VideoAnalysisWorker._fill_available_slots(worker)
        connect.assert_not_called()

    def test_paused_no_purchase_check_creates_no_script_run_or_api_call(self):
        with patch.object(session_analysis, "PAUSED", True), patch.object(jev_pipeline, "logged_call") as api, patch.object(workflow_service.repositories, "create_script_run_started") as create:
            with self.assertRaisesRegex(ValueError, "paused"):
                jev_pipeline.evaluate_carry(Mock(), batch_id=1, group_key="1", location_id=2,
                                           group={"entry": [1], "exit": [2]}, transactions=[])
        api.assert_not_called()
        create.assert_not_called()

    def test_manual_entry_and_kiosk_analysis_are_also_paused(self):
        db = Mock()
        with patch.object(session_analysis, "PAUSED", True):
            with self.assertRaisesRegex(ValueError, "paused"):
                workflow_service.run_entry_for_trigger(db, trigger_id=1, video_path="entry.mp4")
            with self.assertRaisesRegex(ValueError, "paused"):
                workflow_service.run_kiosk_for_session(db, session_id=1, video_paths=["kiosk.mp4"])
        self.assertEqual(db.mock_calls, [])

    def test_resumed_session_check_requires_a_hit_and_no_paid_receipts(self):
        payload = {"entry_trigger_ids": [1], "exit_trigger_ids": [2], "transactions": []}
        row = {"batch_id": 1, "group_key": "1", "location_id": 2, "need_deep_analysis": False}
        with patch.object(session_analysis, "PAUSED", False), patch.object(jev_pipeline, "evaluate_carry", return_value={"status": "reasonable"}) as evaluate:
            self.assertIsNone(session_analysis.no_transaction_check(Mock(), row=row, payload=payload, session_id=7))
            evaluate.assert_not_called()
            row["need_deep_analysis"] = True
            self.assertIsNone(session_analysis.no_transaction_check(Mock(), row=row, payload={**payload, "transactions": [{"receipt_number": "paid"}]}, session_id=7))
            evaluate.assert_not_called()
            session_analysis.no_transaction_check(Mock(), row=row, payload=payload, session_id=7)
            self.assertEqual(evaluate.call_args.kwargs["session_id"], 7)
            evaluate.assert_called_once()

    def test_existing_session_check_is_not_sent_again(self):
        previous = {"status": "reasonable", "script_run_id": 9}
        with patch.object(session_analysis, "PAUSED", False), patch.object(jev_pipeline, "evaluate_carry") as evaluate:
            result = session_analysis.no_transaction_check(Mock(), row={"need_deep_analysis": True},
                payload={"transactions": [], "no_transaction_group": previous}, session_id=7)
        self.assertEqual(result, previous)
        evaluate.assert_not_called()
