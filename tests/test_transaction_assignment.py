from contextlib import ExitStack, nullcontext
from datetime import datetime, timedelta
import unittest
from unittest.mock import Mock, patch

from app.services import transaction_assignment as assignment
from app.services import transaction_assignment_store as store
from app.services import identifier_capture, jev_pipeline, workflow_service as workflow


def time(hour, minute=0, second=0):
    return datetime(2026, 10, 21, hour, minute, second)


def group(entry_id, entry, exit):
    return {"entry_trigger_id": entry_id, "entry_time": entry, "exit_time": exit,
            "exit_trigger_ids": [entry_id + 1] if exit else [], "batch_id": 1, "group_key": str(entry_id)}


def receipt(number, paid_at):
    return {"receipt_number": str(number), "transaction_id": str(number), "status": "paid",
            "payment_attempt_at": paid_at - timedelta(hours=8), "transaction_time": paid_at,
            "raw_payload": {"machineName": "Left"}, "details": [], "total_items": 2}


class AssignmentTests(unittest.TestCase):
    def setUp(self):
        enabled = patch.object(assignment, "is_enabled", return_value=True)
        enabled.start()
        self.addCleanup(enabled.stop)

    def test_user_scenarios_are_decided_per_payment_not_receipt_count(self):
        a = group(1, time(10), time(10, 5))
        for b_exit, expected in [(time(10, 2), [1]), (time(10, 6), [1, 3]), (time(9, 58), [1])]:
            with self.subTest(exit=b_exit):
                groups = [a, group(3, time(9, 57), b_exit)]
                for minute in (3, 4, 5):
                    self.assertEqual([value["entry_trigger_id"] for value in assignment.active_groups(groups, time(10, minute))], expected)

    def test_one_receipt_can_be_ambiguous_and_open_visits_are_active(self):
        groups = [group(1, time(10), time(10, 5)), group(3, time(9, 57), None), group(5, time(10, 4), time(10, 6))]
        self.assertEqual([value["entry_trigger_id"] for value in assignment.active_groups(groups, time(10, 3))], [1, 3])

    def test_payment_attempt_is_utc_and_preferred_over_created_or_formatted_time(self):
        row = receipt(1, time(10, 3))
        row.update(created_at=time(2), transaction_time=time(9))
        actual, source = assignment.payment_time(row)
        self.assertEqual(actual, time(10, 3))
        self.assertEqual(source, "payment_attempt_at")
        row["payment_attempt_at"] = None
        self.assertEqual(assignment.payment_time(row), (time(9), "transaction_time_fallback"))

    def test_edumetro_mapping_is_not_assumed_for_other_locations(self):
        self.assertEqual(assignment.machine_side("Edumetro, Subang", receipt(1, time(10)))["image_side"], "right")
        row = receipt(1, time(10))
        row["raw_payload"]["machineName"] = "Right"
        self.assertEqual(assignment.machine_side("Edumetro", row)["image_side"], "left")
        self.assertIsNone(assignment.machine_side("Another store", row)["image_side"])

    def assign(self, rows, groups, *, previous=None, decision=None):
        self.calls = {}
        with ExitStack() as stack:
            for target, name, value in [
                (store, "location_lock", nullcontext()), (store, "load", previous or {}), (store, "save", None),
                (assignment, "collect_groups", groups), (workflow.repositories, "get_location_endpoint", {"name": "Edumetro"}),
                (workflow.repositories, "list_paid_transactions_for_assignment", rows),
                (assignment, "decide_with_jev", decision or {"status": "assigned", "entry_trigger_id": 1, "method": "jev"}),
            ]:
                self.calls[name] = stack.enter_context(patch.object(target, name, return_value=value))
            return assignment.assign(Mock(), batch_id=1, location_id=2, group={"entry": [1], "exit": [2]},
                                     start_time=time(10), end_time=time(10, 5))

    def test_multiple_receipts_sole_active_owner_are_all_assigned_without_jev(self):
        assigned, report = self.assign([receipt(n, time(10, n)) for n in (3, 4, 5)], [group(1, time(10), time(10, 5))])
        self.assertEqual(len(assigned), 3)
        self.assertEqual(report["selection"], "all")
        self.calls["decide_with_jev"].assert_not_called()
        self.assertEqual(self.calls["save"].call_count, 3)
        self.assertEqual(self.calls["save"].call_args.kwargs["payment_time_utc"], time(2, 5))
        self.assertTrue(all(row["receipt_attribution"] == "confirmed_group_ownership" for row in assigned))

    def test_single_receipt_with_two_active_groups_uses_jev(self):
        assigned, report = self.assign([receipt(1, time(10, 3))], [group(1, time(10), time(10, 5)), group(3, time(9, 57), time(10, 6))],
                                       decision={"status": "assigned", "entry_trigger_id": 3, "method": "jev"})
        self.assertEqual(assigned, [])
        self.assertEqual(report["selection"], "none")
        self.calls["decide_with_jev"].assert_called_once()
        self.assertEqual(report["decisions"][0]["entry_trigger_id"], 3)

    def test_none_uncertain_and_api_errors_never_assign_window_receipts(self):
        for status in ("none", "uncertain", "error"):
            with self.subTest(status=status):
                assigned, report = self.assign([receipt(1, time(10, 3))], [group(1, time(10), time(10, 5)), group(3, time(9), None)],
                                               decision={"status": status, "entry_trigger_id": None, "method": "jev"})
                self.assertEqual(assigned, [])
                self.assertEqual(report["unresolved_count"], 0 if status == "none" else 1)

    def test_existing_receipt_owner_cannot_be_silently_stolen_by_rerun(self):
        previous = {"1": {"context_hash": "older", "status": "assigned", "evidence": {
            "status": "assigned", "method": "jev", "entry_trigger_id": 3, "receipt_number": "1"}}}
        assigned, report = self.assign([receipt(1, time(10, 3))], [group(1, time(10), time(10, 5))], previous=previous)
        self.assertEqual(assigned, [])
        self.assertEqual(report["status"], "needs_review")
        self.calls["save"].assert_not_called()
        self.calls["decide_with_jev"].assert_not_called()

    def test_repeated_identical_context_reuses_decision_without_api_cost(self):
        rows, groups = [receipt(1, time(10, 3))], [group(1, time(10), time(10, 5)), group(3, time(9), None)]
        self.assign(rows, groups)
        saved = self.calls["save"].call_args.kwargs
        self.assign(rows, groups, previous={"1": {"context_hash": saved["context_hash"], "status": "assigned", "evidence": saved["decision"]}})
        self.calls["decide_with_jev"].assert_not_called()
        self.calls["save"].assert_not_called()

    def test_failed_receipt_is_not_automatically_resent_even_if_context_changes(self):
        previous = {"1": {"context_hash": "older", "status": "error", "evidence": {
            "status": "error", "entry_trigger_id": None, "method": "jev", "reason": "Provider failed"}}}
        assigned, report = self.assign([receipt(1, time(10, 3))], [group(1, time(10), time(10, 5)), group(3, time(9), None)], previous=previous)
        self.assertEqual(assigned, [])
        self.assertEqual(report["status"], "needs_review")
        self.assertTrue(report["decisions"][0]["cached"])
        self.calls["decide_with_jev"].assert_not_called()
        self.calls["save"].assert_not_called()

    def test_same_visit_in_another_batch_does_not_invalidate_receipt_cache(self):
        rows, groups = [receipt(1, time(10, 3))], [group(1, time(10), time(10, 5))]
        self.assign(rows, groups)
        saved = self.calls["save"].call_args.kwargs
        groups[0].update(batch_id=100, group_key="new-key")
        self.assign(rows, groups, previous={"1": {"context_hash": saved["context_hash"], "status": "assigned", "evidence": saved["decision"]}})
        self.calls["save"].assert_not_called()

    def test_failed_jev_answer_is_logged_and_does_not_guess_receipt_owner(self):
        with patch.object(workflow.repositories, "create_script_run_started", return_value=8), \
             patch.object(workflow.repositories, "finish_script_run") as finish, \
             patch.object(workflow.repositories, "merge_script_run_stdout_fields", return_value="{}"), \
             patch.object(identifier_capture, "capture_sequence", side_effect=RuntimeError("Camera unavailable")):
            result = assignment.decide_with_jev(Mock(), location_id=2, batch_id=1, receipt="a", row=receipt("a", time(10)),
                                               paid_at=time(10), candidates=[group(1, time(9), time(11))], mapping={})
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["entry_trigger_id"])
        self.assertIn("Camera unavailable", result["reason"])
        self.assertEqual(finish.call_args.kwargs["status"], "failed")

    def test_no_window_receipts_routes_to_no_purchase_without_ai_assignment(self):
        rows, report = self.assign([], [group(1, time(10), time(10, 5))])
        self.assertEqual(rows, [])
        self.assertEqual(report["candidate_count"], 0)
        self.assertEqual(report["selection"], "none")
        self.calls["decide_with_jev"].assert_not_called()

    def test_paid_query_uses_payment_utc_bounds_and_paid_filter(self):
        db = Mock()
        with patch.object(workflow.repositories, "_column_exists", return_value=True), \
             patch.object(workflow.repositories, "_fetch_all_dicts", return_value=[]):
            workflow.repositories.list_paid_transactions_for_assignment(db, location_id=2, start_time=time(10), end_time=time(10, 5))
        sql, params = db.execute.call_args.args
        self.assertIn("paymentAttemptAt", str(sql))
        self.assertEqual(params["start_utc"], time(2))
        self.assertEqual(params["end_utc"], time(2, 5))
        self.assertEqual(params["paid"], workflow.settings.paid_transaction_status_value)
        self.assertEqual(params["location_id"], 2)

    def test_assignment_script_run_is_filterable_without_new_script_name_constraint(self):
        filters, params = workflow.repositories._script_run_list_filters(script_name="transaction_assignment", script_type="jev", model_name=None)
        self.assertEqual(params["script_name"], "grouping")
        self.assertEqual(params["presence_suffix"], "_transaction_assignment")
        self.assertTrue(any("openrouter-jev" in value for value in filters))

    def test_unknown_receipt_identity_stops_scoring(self):
        row = receipt(1, time(10, 3))
        row.update(receipt_number=None, transaction_id=None)
        with self.assertRaisesRegex(ValueError, "receipt identity"):
            self.assign([row], [group(1, time(10), time(10, 5))])

    def test_current_batch_open_entries_block_false_sole_owner(self):
        with patch.object(store, "list_overlapping_groups", return_value=([], [])), \
             patch.object(workflow.repositories, "get_grouping_batch", return_value={"result_payload": {"groups": [], "open_entries": [3]}}), \
             patch.object(workflow.repositories, "get_trigger", side_effect=lambda db, tid: {"trigger_time": {1: time(10), 2: time(10, 5), 3: time(9)}[tid]}):
            groups = assignment.collect_groups(Mock(), location_id=2, start=time(10), end=time(10, 5), batch_id=1, current_group={"entry": [1], "exit": [2]})
        self.assertEqual(len(assignment.active_groups(groups, time(10, 3))), 2)

    def test_session_retry_never_reintroduces_rejected_paid_receipts(self):
        with patch.object(workflow, "_is_kiosk_analysis_disabled", return_value=False), \
             patch.object(assignment, "for_session", return_value=[]), \
             patch.object(workflow.repositories, "delete_session_transactions") as delete, \
             patch.object(workflow, "_identify_kiosk_transaction_groups_for_session") as identify:
            result = workflow._kickoff_kiosk_pipeline_for_session(Mock(), session_id=7, location_id=2, transactions=[receipt(1, time(10))])
        self.assertEqual(result["reason"], "no_assigned_paid_transactions")
        delete.assert_called_once()
        identify.assert_not_called()

    def test_session_payload_is_checked_against_ledger_owner(self):
        payload = {"transaction_assignment": {"entry_trigger_id": 1, "assigned_receipts": ["a", "b"]},
                   "transactions": [{"receipt_number": "a"}, {"receipt_number": "b"}]}
        with patch.object(workflow.repositories, "get_session", return_value={"location_id": 2, "confidence_result_id": 8}), \
             patch.object(workflow.repositories, "get_filter_confidence_result", return_value={"factor_payload": payload}), \
             patch.object(store, "load", return_value={"a": {"entry_trigger_id": 1}, "b": {"entry_trigger_id": 3}}):
            self.assertEqual(assignment.for_session(Mock(), 7), [{"receipt_number": "a"}])

    def test_jev_receipt_choice_includes_all_candidates_none_uncertain_and_ten_payment_samples(self):
        options = {"entry_1", "entry_3", "none", "uncertain"}
        answer = {"answers": {"receipt_owner": {"type": "choice", "choice": "entry_3", "confidence": 0.9,
                  "probabilities": {key: 0.94 if key == "entry_3" else 0.02 for key in options}}}}
        with ExitStack() as stack:
            stack.enter_context(patch.object(workflow.repositories, "create_script_run_started", return_value=8))
            finish = stack.enter_context(patch.object(workflow.repositories, "finish_script_run"))
            stack.enter_context(patch.object(workflow.repositories, "merge_script_run_stdout_fields", return_value="{}"))
            capture = stack.enter_context(patch.object(identifier_capture, "capture_sequence", return_value={"frames": []}))
            stack.enter_context(patch.object(assignment, "_references", return_value=[]))
            stack.enter_context(patch.object(jev_pipeline, "encode_sequences", return_value=([{}, {}], ["https://example.test/image.jpg"])))
            call = stack.enter_context(patch.object(jev_pipeline, "logged_call", return_value=(answer, {})))
            result = assignment.decide_with_jev(Mock(), location_id=2, batch_id=1, receipt="a", row=receipt("a", time(10)), paid_at=time(10),
                                                candidates=[group(1, time(9), time(11)), group(3, time(9), time(11))],
                                                mapping={"machine_name": "Left", "image_side": "right"})
        self.assertEqual(result["entry_trigger_id"], 3)
        times = capture.call_args.kwargs["times"]
        self.assertEqual(len(times), 10)
        self.assertEqual(times[0], time(10) - timedelta(seconds=5))
        self.assertEqual(times[-1], time(10) + timedelta(seconds=4))
        self.assertEqual(set(call.call_args.args[2]["questions"]["receipt_owner"]["criteria"]), options)
        self.assertEqual(finish.call_args.kwargs["status"], "success")


class AssignmentPauseTests(unittest.TestCase):
    def test_default_is_paused_and_only_explicit_true_resumes(self):
        for value, expected in [(None, False), ("false", False), ("true", True), ("unexpected", False)]:
            with self.subTest(value=value), patch.object(assignment.repositories, "get_app_setting", return_value=value):
                self.assertEqual(assignment.is_enabled(Mock()), expected)

    def test_paused_assignment_does_not_query_receipts_or_send_api(self):
        with patch.object(assignment, "is_enabled", return_value=False), \
             patch.object(workflow.repositories, "list_paid_transactions_for_assignment") as query, \
             patch.object(assignment, "decide_with_jev") as call:
            with self.assertRaises(assignment.AssignmentPaused):
                assignment.assign(Mock(), batch_id=1, location_id=2, group={"entry": [1]}, start_time=time(10), end_time=time(11))
        query.assert_not_called()
        call.assert_not_called()

    def test_paused_confidence_batch_stays_queued_without_scoring_or_error_record(self):
        with patch.object(assignment, "is_enabled", return_value=False), \
             patch.object(workflow, "_try_acquire_theft_confidence_lock") as lock, \
             patch.object(workflow, "_run_theft_confidence_for_grouping_batch_locked") as score:
            result = workflow.run_theft_confidence_for_grouping_batch(Mock(), batch_id=1)
        self.assertEqual(result["reason"], "transaction_assignment_paused")
        score.assert_not_called()
        lock.assert_not_called()

    def test_direct_assignment_call_is_blocked_before_capture_or_script_creation(self):
        with patch.object(assignment, "is_enabled", return_value=False), \
             patch.object(workflow.repositories, "create_script_run_started") as create, \
             patch.object(identifier_capture, "capture_sequence") as capture:
            with self.assertRaises(assignment.AssignmentPaused):
                assignment.decide_with_jev(Mock(), location_id=2, batch_id=1, receipt="a", row={},
                    paid_at=time(10), candidates=[group(1, time(9), time(11))], mapping={})
        create.assert_not_called()
        capture.assert_not_called()

    def test_worker_does_not_spawn_or_requeue_while_paused(self):
        from app.workers.theft_confidence_worker import TheftConfidenceWorker, RunningJob
        from app.workers import theft_confidence_worker as module
        worker = TheftConfidenceWorker()
        process = Mock()
        process.poll.return_value = None
        worker._running[1] = RunningJob(process=process, batch_id=1, location_id=2)
        with patch.object(module, "TransactionalSessionLocal", return_value=Mock()), \
             patch.object(assignment, "is_enabled", return_value=False), \
             patch.object(module.repositories, "list_pending_theft_confidence_batches") as query, \
             patch.object(module.subprocess, "Popen") as spawn, self.assertLogs(module.logger, level="WARNING"):
            worker._fill_available_slots()
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=10)
        self.assertEqual(worker._running, {})
        query.assert_not_called()
        spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
