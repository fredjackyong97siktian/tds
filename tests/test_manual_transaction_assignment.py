from contextlib import ExitStack, nullcontext
from datetime import datetime
import unittest
from unittest.mock import Mock, patch

from app.services import transaction_assignment as assignment
from app.services import transaction_assignment_store as store
from app.services import workflow_service as workflow
from app.services import jev_pipeline, identifier_capture


class ManualAssignmentTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.db = Mock()
        self.snapshot = {"receipt_number": "receipt", "status": "uncertain", "machine_name": "Left",
                         "image_side": "right", "mapping": "test"}
        self.result = {"id": 10, "batch_id": 1, "location_id": 2, "factor_payload": {
            "entry_trigger_ids": [9769], "exit_trigger_ids": [9771],
            "transaction_assignment": {"entry_trigger_id": 9769, "decisions": [self.snapshot]}}}
        self.previous = {"status": "uncertain", "payment_time_utc": datetime(2026, 10, 21, 2, 3),
                         "evidence": {"status": "uncertain", "script_run_id": 22,
                                      "jev_decision": {"choice": "entry_9769", "confidence": 0.25}}}
        self.candidates = [{"entry_trigger_id": entry, "exit_trigger_ids": [entry + 2], "batch_id": 1,
                            "entry_time": datetime(2026, 10, 21, 10), "exit_time": datetime(2026, 10, 21, 10, 5)}
                           for entry in (9769, 9770)]
        self.stack.enter_context(patch.object(assignment.repositories, "get_filter_confidence_result", return_value=self.result))
        self.stack.enter_context(patch.object(assignment.repositories, "_fetch_all_dicts", return_value=[{"batch_id": 1}]))
        self.stack.enter_context(patch.object(store, "load", return_value={"receipt": self.previous}))
        self.stack.enter_context(patch.object(store, "location_lock", return_value=nullcontext()))
        self.save = self.stack.enter_context(patch.object(store, "save"))
        self.collect = self.stack.enter_context(patch.object(assignment, "collect_groups", return_value=self.candidates))
        self.lock = self.stack.enter_context(patch.object(workflow, "_try_acquire_theft_confidence_lock", return_value=Mock()))
        self.release = self.stack.enter_context(patch.object(workflow, "_release_theft_confidence_lock"))
        self.stack.enter_context(patch.object(assignment, "is_enabled", return_value=True))
        self.stack.enter_context(patch.object(assignment.repositories, "is_worker_paused", return_value=False))
        self.ai = self.stack.enter_context(patch.object(jev_pipeline, "logged_call"))
        self.capture = self.stack.enter_context(patch.object(identifier_capture, "capture_sequence"))

    def resolve(self, entry=9769):
        return assignment.resolve_manually(self.db, confidence_result_id=10, receipt="receipt", entry_trigger_id=entry)

    def test_manual_owner_is_saved_without_capture_or_ai_and_preserves_jev_answer(self):
        response = self.resolve()
        decision = self.save.call_args.kwargs["decision"]
        self.assertEqual(response["status"], "assigned")
        self.assertEqual(decision["entry_trigger_id"], 9769)
        self.assertEqual(decision["method"], "manual")
        self.assertEqual(decision["jev_decision"]["confidence"], 0.25)
        self.assertEqual(decision["manual_resolution"]["confidence_result_id"], 10)
        self.ai.assert_not_called()
        self.capture.assert_not_called()
        self.release.assert_called_once()
        self.assertIn("location_id=:location_id", str(self.db.execute.call_args.args[0]))

    def test_manual_none_resolves_receipt_without_an_owner(self):
        response = self.resolve(None)
        self.assertEqual(response["status"], "none")
        self.assertIsNone(self.save.call_args.kwargs["decision"]["entry_trigger_id"])

    def test_ungrouped_or_inactive_identity_cannot_be_selected(self):
        for entry in (9999, 9769):
            with self.subTest(entry=entry):
                if entry == 9769:
                    self.candidates[0]["exit_trigger_ids"] = []
                with self.assertRaisesRegex(ValueError, "existing completed group"):
                    self.resolve(entry)
        self.save.assert_not_called()

    def test_receipt_must_appear_in_this_results_audit(self):
        self.snapshot["receipt_number"] = "other-receipt"
        with self.assertRaisesRegex(ValueError, "does not belong"):
            self.resolve()
        self.save.assert_not_called()

    def test_concurrent_confirmed_owner_cannot_be_stolen(self):
        self.previous.update(status="assigned", entry_trigger_id=9770)
        with self.assertRaisesRegex(RuntimeError, "already has a confirmed owner"):
            self.resolve()
        self.save.assert_not_called()

    def test_paused_pipeline_allows_free_manual_save_but_does_not_resume_it(self):
        with patch.object(assignment, "is_enabled", return_value=False):
            response = self.resolve()
        self.assertIn("Resume", response["message"])
        self.save.assert_called_once()
        self.ai.assert_not_called()

    def test_another_affected_running_batch_blocks_manual_update(self):
        self.candidates[1]["batch_id"] = 2
        self.lock.side_effect = [Mock(), None]
        with self.assertRaisesRegex(RuntimeError, "Another affected"):
            self.resolve()
        self.save.assert_not_called()
        self.release.assert_called_once()

    def test_manual_decision_uses_same_context_hash_as_regular_assignment(self):
        import hashlib
        import json
        self.resolve()
        expected = {"version": assignment.POLICY_VERSION, "paid_at": datetime(2026, 10, 21, 10, 3),
                    "groups": [{key: group[key] for key in ("entry_trigger_id", "exit_trigger_ids", "entry_time", "exit_time")}
                               for group in self.candidates],
                    "mapping": {key: self.snapshot[key] for key in ("machine_name", "image_side", "mapping")}}
        self.assertEqual(self.save.call_args.kwargs["context_hash"],
                         hashlib.sha256(json.dumps(expected, default=str, sort_keys=True).encode()).hexdigest())


if __name__ == "__main__":
    unittest.main()
