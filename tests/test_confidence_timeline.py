import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException

from app.routers import filtering
from app.services import confidence_timeline as timeline


class ConfidenceTimelineTests(unittest.TestCase):
    def setUp(self):
        self.row = {"id": 10, "batch_id": 5, "group_key": "1", "location_id": 2, "session_id": 7,
                    "factor_payload": {"entry_trigger_ids": [11], "exit_trigger_ids": [12],
                        "transaction_assignment": {"decisions": [{"receipt_number": "paid-a", "script_run_id": 4,
                            "kiosk_evidence": {"script_run_id": 3}}]}}}
        self.context = {"row": self.row, "trigger_ids": {11, 12}, "run_ids": {3, 4}, "receipts": {"paid-a"}}

    def run_row(self, **values):
        return {"script_run_id": 1, "status": "success", "script_name": "retrieve_video",
                "model_name": "dahua_rtsp_playback:trigger_frame_asset", "runner_payload": {}, **values}

    def test_frame_and_video_retrieval_follow_group_triggers(self):
        self.assertEqual(timeline._scope(self.run_row(trigger_id=11), **self.context), "Trigger evidence")
        self.assertIsNone(timeline._scope(self.run_row(trigger_id=99), **self.context))

    def test_batch_wide_grouping_is_labelled_shared(self):
        run = self.run_row(script_name="grouping", model_name="openrouter-jev-1.14_grouping_adjacent",
                           runner_payload={"batch_id": 5})
        self.assertEqual(timeline._scope(run, **self.context), "Shared grouping batch")
        run["runner_payload"]["batch_id"] = 6
        self.assertIsNone(timeline._scope(run, **self.context))

    def test_legacy_batch_root_without_payload_is_included(self):
        run = self.run_row(script_name="grouping", batch_root=True, runner_payload=None)
        self.assertEqual(timeline._scope(run, **self.context), "Shared grouping batch")

    def test_receipt_comparison_can_use_another_primary_trigger(self):
        run = self.run_row(script_name="grouping", model_name="openrouter-jev-1.14_transaction_assignment",
                           trigger_id=99, runner_payload={"receipt_number": "paid-a", "location_id": 2})
        self.assertEqual(timeline._scope(run, **self.context), "Receipt assignment")
        run["runner_payload"]["location_id"] = 3
        self.assertIsNone(timeline._scope(run, **self.context))

    def test_an_unrelated_receipt_in_same_batch_is_not_a_grouping_step(self):
        run = self.run_row(script_name="grouping", model_name="openrouter-jev-1.14_transaction_assignment",
                           trigger_id=11, runner_payload={"receipt_number": "other", "location_id": 2, "batch_id": 5})
        self.assertIsNone(timeline._scope(run, **self.context))

    def test_carry_is_scoped_to_group_not_merely_batch(self):
        run = self.run_row(script_name="carry_confidence", runner_payload={"batch_id": 5, "group_key": "2"})
        self.assertIsNone(timeline._scope(run, **self.context))
        run["runner_payload"]["group_key"] = "1"
        self.assertEqual(timeline._scope(run, **self.context), "Group analysis")

    def test_sensor_presence_is_not_shared_with_unrelated_groups(self):
        run = self.run_row(script_name="grouping", model_name="openrouter-jev-1.14_person_presence",
                           trigger_id=99, runner_payload={"batch_id": 5})
        self.assertIsNone(timeline._scope(run, **self.context))
        run["trigger_id"] = 12
        self.assertEqual(timeline._scope(run, **self.context), "Trigger evidence")

    def test_saved_retry_and_child_evidence_are_linked(self):
        self.assertEqual(timeline._recorded_runs(self.row["factor_payload"]), {3, 4})
        self.assertEqual(timeline._scope(self.run_row(script_run_id=4), **self.context), "Recorded for this group")
        self.assertEqual(timeline._scope(self.run_row(runner_payload={"parent_script_run_id": 4}), **self.context), "Linked evidence")

    def test_session_runs_are_included(self):
        self.assertEqual(timeline._scope(self.run_row(script_name="entry", session_id=7), **self.context), "Session")

    def test_failed_and_running_records_are_never_shown(self):
        for status in ("failed", "running", "pending"):
            self.assertIsNone(timeline._scope(self.run_row(script_run_id=4, status=status), **self.context))

    def test_query_is_chronological_success_only_and_excludes_logs(self):
        db = Mock()
        db.execute.return_value.mappings.return_value.all.return_value = [self.run_row(trigger_id=11)]
        with patch.object(timeline.repositories, "list_filter_confidence_results", return_value=[self.row]), \
             patch.object(timeline.repositories, "_script_run_cost_select", return_value="cost_amount, cost_currency, cost_source"):
            rows = timeline.list_successful_runs(db, 10)
        sql = str(db.execute.call_args.args[0])
        self.assertIn("status = 'success'", sql)
        self.assertIn("order by coalesce(started_at, finished_at) asc, id asc", sql)
        self.assertNotIn("stdout_log", sql)
        self.assertNotIn("stderr_log", sql)
        self.assertNotIn("runner_payload", rows[0])
        self.assertEqual(db.execute.call_args.args[1]["trigger_ids"], [11, 12])

    def test_missing_result_does_not_query_script_runs(self):
        db = Mock()
        with patch.object(timeline.repositories, "list_filter_confidence_results", return_value=[]):
            self.assertIsNone(timeline.list_successful_runs(db, 10))
        db.execute.assert_not_called()

    def test_missing_result_endpoint_returns_404(self):
        with patch.object(timeline, "list_successful_runs", return_value=None):
            with self.assertRaises(HTTPException) as error:
                filtering.confidence_script_runs(10, Mock())
        self.assertEqual(error.exception.status_code, 404)
