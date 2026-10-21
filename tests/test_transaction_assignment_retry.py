from contextlib import ExitStack, nullcontext
import unittest
from unittest.mock import Mock, patch

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


if __name__ == "__main__":
    unittest.main()
