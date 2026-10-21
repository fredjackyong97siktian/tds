from contextlib import ExitStack, nullcontext
import unittest
import json
from unittest.mock import Mock, patch
from datetime import datetime

from fastapi import HTTPException
from sqlalchemy.exc import OperationalError
from app.routers import workflows

from app.services import transaction_assignment as assignment
from app.services import transaction_assignment_store as store
from app.services import workflow_service as workflow


class AssignmentRetryTests(unittest.TestCase):
    def setUp(self):
        self.db = Mock()
        self.run = {"model_name": "openrouter-jev-1.14_transaction_assignment",
                    "runner_payload": {"batch_id": 494, "location_id": 2,
                                       "receipt_number": "receipt-test",
                                       "payment_time": "2026-10-20T22:19:59"}}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(assignment, "is_enabled", return_value=True))
        self.stack.enter_context(patch.object(assignment.repositories, "get_script_run", return_value=self.run))
        self.stack.enter_context(patch.object(assignment.repositories, "get_grouping_batch",
                                             return_value={"location_id": 2, "status": "success"}))
        self.stack.enter_context(patch.object(assignment.repositories, "is_worker_paused", return_value=False))
        self.lock = self.stack.enter_context(patch.object(workflow, "_try_acquire_theft_confidence_lock", return_value=Mock()))
        self.release = self.stack.enter_context(patch.object(workflow, "_release_theft_confidence_lock"))
        self.stack.enter_context(patch.object(store, "location_lock", return_value=nullcontext()))
        self.load = self.stack.enter_context(patch.object(store, "load", return_value={}))
        self.save = self.stack.enter_context(patch.object(store, "save"))
        self.api = self.stack.enter_context(patch.object(assignment, "decide_with_jev"))

    def test_retry_queues_one_receipt_without_calling_api_or_rerunning_grouping(self):
        result = assignment.retry_from_script_run(self.db, 20804)
        self.assertTrue(result["queued"])
        self.api.assert_not_called()
        decision = self.save.call_args.kwargs["decision"]
        self.assertTrue(decision["retry_authorized"])
        self.assertEqual(self.save.call_args.kwargs["receipt"], "receipt-test")
        update, cleanup = self.db.execute.call_args_list
        self.assertIn("assignment_retry_requested", str(update.args[0]))
        self.assertIn("json_extract('true', '$')", str(update.args[0]))
        self.assertEqual(update.args[1], {"batch_id": 494, "receipt": "receipt-test"})
        self.assertIn("group_key='__error__'", str(cleanup.args[0]))
        self.release.assert_called_once()

    def test_already_assigned_receipt_preserves_ownership(self):
        self.load.return_value = {"receipt-test": {"status": "assigned", "entry_trigger_id": 9546}}
        assignment.retry_from_script_run(self.db, 20804)
        self.save.assert_not_called()
        self.api.assert_not_called()

    def test_running_batch_rejects_retry_without_resetting_receipt(self):
        self.lock.return_value = None
        with self.assertRaisesRegex(RuntimeError, "already running"):
            assignment.retry_from_script_run(self.db, 20804)
        self.save.assert_not_called()
        self.db.execute.assert_not_called()

    def test_write_failure_rolls_back_and_releases_lock(self):
        self.save.side_effect = RuntimeError("database unavailable")
        with self.assertRaisesRegex(RuntimeError, "database unavailable"):
            assignment.retry_from_script_run(self.db, 20804)
        self.db.rollback.assert_called_once()
        self.release.assert_called_once()

    def test_wrong_script_type_does_not_modify_database(self):
        self.run["model_name"] = "gemini_grouping_direct"
        with self.assertRaisesRegex(ValueError, "not Transaction Assignment"):
            assignment.retry_from_script_run(self.db, 20804)
        self.db.execute.assert_not_called()

    def test_retry_accepts_json_payload_and_keeps_script_reference(self):
        self.run["runner_payload"] = json.dumps(self.run["runner_payload"])
        assignment.retry_from_script_run(self.db, 20804)
        decision = self.save.call_args.kwargs["decision"]
        self.assertEqual(decision["script_run_id"], 20804)
        self.assertEqual(decision["batch_id"], 494)

    def test_retry_rejects_malformed_payload_without_writes(self):
        self.run["runner_payload"] = "not json"
        with self.assertRaisesRegex(ValueError, "invalid assignment payload"):
            assignment.retry_from_script_run(self.db, 20804)
        self.db.execute.assert_not_called()

    def test_stuck_list_reports_queued_receipt_without_calling_provider(self):
        row = {"location_id": 2, "receipt_number": "receipt-test", "status": "error",
               "evidence": json.dumps({"retry_authorized": True, "reason": "Queued"}),
               "runner_payload": json.dumps({"batch_id": 494}), "latest_script_run_id": 20804,
               "payment_time_utc": datetime(2026, 10, 20, 14, 19, 59)}
        with patch.object(store, "list_unresolved", return_value=[row]):
            result = assignment.list_stuck_assignments(self.db)
        self.assertEqual(result["items"][0]["status"], "queued")
        self.assertEqual(result["items"][0]["script_run_id"], 20804)
        self.assertEqual(result["items"][0]["batch_id"], 494)
        self.assertNotIn("evidence", result["items"][0])
        self.api.assert_not_called()

    def test_stuck_list_handles_missing_or_malformed_legacy_payload(self):
        row = {"location_id": 2, "receipt_number": "receipt-test", "status": "error",
               "evidence": "invalid", "runner_payload": [], "script_run_id": None}
        with patch.object(store, "list_unresolved", return_value=[row]):
            result = assignment.list_stuck_assignments(self.db)
        self.assertIsNone(result["items"][0]["script_run_id"])

    def test_retry_route_returns_driver_diagnostic_not_sql_or_plain_text(self):
        error = OperationalError("private SQL", {"private": "parameters"},
                                 Exception(3819, "Check constraint 'test_constraint' is violated."))
        with patch.object(assignment, "retry_from_script_run", side_effect=error), patch.object(workflows.logger, "exception"):
            with self.assertRaises(HTTPException) as raised:
                workflows.retry_transaction_assignment(20804, self.db)
        self.assertEqual(raised.exception.status_code, 500)
        self.assertIn("3819", raised.exception.detail)
        self.assertIn("test_constraint", raised.exception.detail)
        self.assertNotIn("private", raised.exception.detail)

    def test_retry_route_unexpected_failure_returns_actionable_error(self):
        with patch.object(assignment, "retry_from_script_run", side_effect=KeyError("missing")), patch.object(workflows.logger, "exception"):
            with self.assertRaises(HTTPException) as raised:
                workflows.retry_transaction_assignment(20804, self.db)
        self.assertEqual(raised.exception.status_code, 500)
        self.assertIn("server logs", raised.exception.detail)

    def test_unresolved_store_query_is_bounded_and_uses_existing_run_reference_first(self):
        with patch.object(assignment.repositories, "_fetch_all_dicts", return_value=[]):
            store.list_unresolved(self.db, limit=10000)
        sql, params = self.db.execute.call_args.args
        self.assertEqual(params["limit"], 50)
        self.assertIn("status in ('error', 'uncertain')", str(sql))
        self.assertIn("coalesce(\n          a.script_run_id", str(sql))

    def test_scored_keys_do_not_skip_legacy_numeric_retry_flags(self):
        self.db.execute.return_value.all.return_value = []
        assignment.repositories.list_scored_group_keys_for_batch(self.db, 494)
        sql = str(self.db.execute.call_args.args[0])
        self.assertIn("not in ('true', '1')", sql)
        self.assertIn("group_key <> '__error__'", sql)


if __name__ == "__main__":
    unittest.main()
