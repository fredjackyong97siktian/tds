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

    def test_open_identity_without_a_completed_group_is_not_active(self):
        groups = [group(1, time(10), time(10, 5)), group(3, time(9, 57), None), group(5, time(10, 4), time(10, 6))]
        self.assertEqual([value["entry_trigger_id"] for value in assignment.active_groups(groups, time(10, 3))], [1])

    def test_stale_unresolved_entries_do_not_count_as_active_forever(self):
        paid_at = time(10, 3)
        groups = [group(1, time(10), time(10, 5)),
                  *[group(100 + n, paid_at - timedelta(days=n + 1), None) for n in range(531)]]
        self.assertEqual([row["entry_trigger_id"] for row in assignment.active_groups(groups, paid_at)], [1])
        assigned, report = self.assign([receipt(1, paid_at)], groups)
        self.assertEqual(len(assigned), 1)
        self.assertEqual(report["decisions"][0]["method"], "time")
        self.assertEqual(len(report["decisions"][0]["active_groups"]), 1)
        self.calls["decide_with_jev"].assert_not_called()

    def test_recent_open_identities_are_excluded_too(self):
        paid_at = time(10, 3)
        groups = [group(1, paid_at - timedelta(minutes=60), None),
                  group(3, paid_at - timedelta(minutes=60, seconds=1), None),
                  group(5, paid_at - timedelta(minutes=30), None)]
        self.assertEqual(assignment.active_groups(groups, paid_at), [])

    def test_completed_long_visit_uses_its_real_exit_not_open_entry_age_limit(self):
        groups = [group(1, time(8), time(10, 5)), group(3, time(8), time(10, 2))]
        self.assertEqual([row["entry_trigger_id"] for row in assignment.active_groups(groups, time(10, 3))], [1])

    def test_all_three_scenarios_only_send_jev_for_actual_overlap(self):
        for exit_time, needs_jev in [(time(10, 2), False), (time(10, 6), True), (time(9, 58), False)]:
            with self.subTest(exit_time=exit_time):
                groups = [group(1, time(10), time(10, 5)), group(3, time(9, 57), exit_time)]
                assigned, report = self.assign([receipt(n, time(10, n)) for n in (3, 4, 5)], groups)
                self.assertEqual(len(assigned), 3)
                self.assertEqual(self.calls["decide_with_jev"].call_count, 3 if needs_jev else 0)
                self.assertTrue(all(row["method"] == ("jev" if needs_jev else "time") for row in report["decisions"]))

    def test_one_group_receipts_use_payment_bounds_for_collecting_candidates(self):
        self.assign([receipt(1, time(10, 3)), receipt(2, time(10, 4))], [group(1, time(10), time(10, 5))])
        kwargs = self.calls["collect_groups"].call_args.kwargs
        self.assertEqual(kwargs["start"], time(10, 3))
        self.assertEqual(kwargs["end"], time(10, 4))

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

    def test_full_machine_names_are_mapped_without_matching_unrelated_words(self):
        for name, expected in [("Edumetro-right-1", "left"), ("Edumetro-left-2", "right"),
                               ("EDUMETRO_RIGHT_1", "left"), ("Right1", "left"),
                               ("bright-1", None), ("left-right-1", None), ("", None)]:
            with self.subTest(name=name):
                row = receipt(1, time(10))
                row["raw_payload"]["machineName"] = name
                self.assertEqual(assignment.machine_side("Edumetro, Subang", row)["image_side"], expected)

    def test_overlapping_group_query_only_reads_completed_memberships(self):
        db = Mock()
        with patch.object(workflow.repositories, "_fetch_all_dicts", side_effect=[[], []]):
            store.list_overlapping_groups(db, location_id=2, start=time(10, 3), end=time(10, 5))
        closed_sql, closed_params = db.execute.call_args_list[0].args
        self.assertIn("exit_time >= :start_time", str(closed_sql))
        self.assertNotIn("or exit_time is null", str(closed_sql))
        self.assertEqual(closed_params["location_id"], 2)
        self.assertEqual(db.execute.call_count, 1)
        self.assertIn("gi.status='grouped'", str(closed_sql))

    def assign(self, rows, groups, *, previous=None, decision=None, recovered=None, save_side_effect=None):
        self.calls = {}
        with ExitStack() as stack:
            for target, name, value in [
                (store, "location_lock", nullcontext()), (store, "load", previous or {}), (store, "save", None),
                (store, "recover_decision", recovered),
                (assignment, "collect_groups", groups), (workflow.repositories, "get_location_endpoint", {"name": "Edumetro"}),
                (workflow.repositories, "list_paid_transactions_for_assignment", rows),
                (workflow.repositories, "finish_script_run", None),
                (workflow.repositories, "merge_script_run_stdout_fields", "{}"),
                (assignment, "decide_with_jev", decision or {"status": "assigned", "entry_trigger_id": 1, "method": "jev"}),
            ]:
                self.calls[name] = stack.enter_context(patch.object(target, name, return_value=value))
            self.calls["save"].side_effect = save_side_effect
            return assignment.assign(Mock(), batch_id=1, location_id=2, group={"entry": [1], "exit": [2]},
                                     start_time=time(10), end_time=time(10, 5))

    def test_explicit_retry_recovers_saved_answer_without_buying_it_again(self):
        previous = {"1": {"status": "error", "context_hash": "old",
                          "evidence": {"retry_authorized": True}}}
        recovered = {"status": "assigned", "entry_trigger_id": 1, "method": "jev",
                     "recorded_candidate_entry_ids": [1, 3], "script_run_id": 20804}
        assigned, _ = self.assign([receipt(1, time(10, 3))],
            [group(1, time(10), time(10, 5)), group(3, time(10), time(10, 6))],
            previous=previous, recovered=recovered)
        self.assertEqual(len(assigned), 1)
        self.calls["decide_with_jev"].assert_not_called()
        self.assertEqual(self.calls["save"].call_count, 1)

    def test_explicit_retry_without_saved_answer_reserves_before_one_api_call(self):
        previous = {"1": {"status": "error", "context_hash": "old",
                          "evidence": {"retry_authorized": True}}}
        assigned, _ = self.assign([receipt(1, time(10, 3))],
            [group(1, time(10), time(10, 5)), group(3, time(10), time(10, 6))], previous=previous)
        self.assertEqual(len(assigned), 1)
        self.calls["decide_with_jev"].assert_called_once()
        reservation = self.calls["save"].call_args_list[0].kwargs["decision"]
        self.assertTrue(reservation["attempt_reserved"])
        self.assertNotIn("retry_authorized", reservation)

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
                assigned, report = self.assign([receipt(1, time(10, 3))], [group(1, time(10), time(10, 5)), group(3, time(9, 30), time(10, 6))],
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
        rows, groups = [receipt(1, time(10, 3))], [group(1, time(10), time(10, 5)), group(3, time(9, 30), None)]
        self.assign(rows, groups)
        saved = self.calls["save"].call_args.kwargs
        self.assign(rows, groups, previous={"1": {"context_hash": saved["context_hash"], "status": "assigned", "evidence": saved["decision"]}})
        self.calls["decide_with_jev"].assert_not_called()
        self.calls["save"].assert_not_called()

    def test_paid_attempt_is_reserved_before_visual_work_and_finalized_after_receipt_save(self):
        events = []
        groups = [group(1, time(10), time(10, 5)), group(3, time(9, 57), time(10, 6))]
        def saved(db, **kwargs):
            events.append("reserve" if kwargs["decision"].get("attempt_reserved") else "receipt_committed")
        self.assign([receipt(1, time(10, 3))], groups, save_side_effect=saved,
                    decision={"status": "assigned", "entry_trigger_id": 1, "method": "jev", "script_run_id": 77})
        self.assertEqual(events, ["reserve", "receipt_committed"])
        self.assertEqual(self.calls["finish_script_run"].call_args.kwargs["status"], "success")
        self.assertTrue(self.calls["save"].call_args_list[0].kwargs["decision"]["attempt_reserved"])

    def test_failed_final_write_is_not_reported_success_and_retry_does_not_resend(self):
        groups = [group(1, time(10), time(10, 5)), group(3, time(9, 57), time(10, 6))]
        with self.assertRaisesRegex(RuntimeError, "write denied"):
            self.assign([receipt(1, time(10, 3))], groups,
                save_side_effect=[None, RuntimeError("write denied")],
                decision={"status": "assigned", "entry_trigger_id": 1, "method": "jev", "script_run_id": 77})
        self.assertEqual(self.calls["finish_script_run"].call_args.kwargs["status"], "failed")
        self.assertIn("persistence failed", self.calls["finish_script_run"].call_args.kwargs["stderr_log"])
        barrier = self.calls["save"].call_args_list[0].kwargs["decision"]
        self.assign([receipt(1, time(10, 3))], groups,
                    previous={"1": {"status": "error", "context_hash": "old", "evidence": barrier}})
        self.calls["decide_with_jev"].assert_not_called()
        self.calls["save"].assert_not_called()

    def test_reservation_write_failure_prevents_even_the_first_paid_call(self):
        with self.assertRaisesRegex(RuntimeError, "INSERT denied"):
            self.assign([receipt(1, time(10, 3))], [group(1, time(10), time(10, 5)), group(3, time(9, 57), None)],
                        save_side_effect=RuntimeError("INSERT denied"))
        self.calls["decide_with_jev"].assert_not_called()

    def test_prior_success_is_recovered_without_another_api_call(self):
        groups = [group(1, time(10), time(10, 5)), group(3, time(9, 57), time(10, 6))]
        for status in ("assigned", "none", "uncertain"):
            with self.subTest(status=status):
                recovered = {"status": status, "entry_trigger_id": 1 if status == "assigned" else None,
                             "method": "jev", "script_run_id": 77, "recovered_from_script_run": True,
                             "recorded_candidate_entry_ids": [1, 3]}
                assigned, report = self.assign([receipt(1, time(10, 3))], groups, recovered=recovered)
                self.assertEqual(len(assigned), 1 if status == "assigned" else 0)
                self.assertTrue(report["decisions"][0]["recovered_from_script_run"])
                self.calls["decide_with_jev"].assert_not_called()
                self.assertEqual(self.calls["save"].call_count, 1)

    def test_interrupted_attempt_with_checkpoint_recovers_instead_of_remaining_stuck(self):
        previous = {"1": {"status": "error", "context_hash": "old", "evidence": {"attempt_reserved": True, "status": "error"}}}
        recovered = {"status": "assigned", "entry_trigger_id": 1, "method": "jev", "script_run_id": 77,
                     "recovered_from_script_run": True, "recorded_candidate_entry_ids": [1, 3]}
        assigned, _report = self.assign([receipt(1, time(10, 3))],
            [group(1, time(10), time(10, 5)), group(3, time(9, 57), time(10, 6))], previous=previous, recovered=recovered)
        self.assertEqual(len(assigned), 1)
        self.calls["decide_with_jev"].assert_not_called()
        self.calls["save"].assert_called_once()

    def test_successful_none_and_uncertain_are_not_rebought_when_candidates_change(self):
        for status in ("none", "uncertain"):
            with self.subTest(status=status):
                previous = {"1": {"status": status, "context_hash": "old", "evidence": {
                    "status": status, "entry_trigger_id": None, "method": "jev"}}}
                assigned, report = self.assign([receipt(1, time(10, 3))],
                    [group(1, time(10), time(10, 5)), group(3, time(9, 57), None)], previous=previous)
                self.assertEqual(assigned, [])
                self.assertEqual(report["status"], "needs_review")
                self.calls["decide_with_jev"].assert_not_called()

    def test_recovery_only_accepts_valid_completed_answers_for_the_scoped_receipt(self):
        payload = {"candidate_entry_trigger_ids": [1, 3]}
        rows = [{"id": 77, "status": "success", "runner_payload": payload,
                 "stdout_log": '{"status":"assigned","entry_trigger_id":1}'}]
        db = Mock()
        with patch.object(workflow.repositories, "_fetch_all_dicts", return_value=rows):
            result = store.recover_decision(db, location_id=2, receipt="receipt-a")
        self.assertEqual(result["entry_trigger_id"], 1)
        self.assertEqual(result["script_run_id"], 77)
        self.assertEqual(db.execute.call_args.args[1], {"location_id": 2, "receipt": "receipt-a"})
        for result in ('{"status":"error"}', '{"status":"assigned","entry_trigger_id":99}', 'not json'):
            rows[0]["stdout_log"] = result
            with patch.object(workflow.repositories, "_fetch_all_dicts", return_value=rows):
                self.assertIsNone(store.recover_decision(db, location_id=2, receipt="receipt-a"))

    def test_api_succeeded_checkpoint_can_be_recovered_after_database_failure(self):
        row = {"id": 77, "status": "failed", "runner_payload": {"candidate_entry_trigger_ids": [1, 3]},
               "stdout_log": {"status": "assigned", "entry_trigger_id": 1, "api_succeeded": True}}
        with patch.object(workflow.repositories, "_fetch_all_dicts", return_value=[row]):
            self.assertIsNotNone(store.recover_decision(Mock(), location_id=2, receipt="a"))
        row["stdout_log"].pop("api_succeeded")
        with patch.object(workflow.repositories, "_fetch_all_dicts", return_value=[row]):
            self.assertIsNone(store.recover_decision(Mock(), location_id=2, receipt="a"))

    def test_failed_receipt_is_not_automatically_resent_even_if_context_changes(self):
        previous = {"1": {"context_hash": "older", "status": "error", "evidence": {
            "status": "error", "entry_trigger_id": None, "method": "jev", "reason": "Provider failed"}}}
        assigned, report = self.assign([receipt(1, time(10, 3))], [group(1, time(10), time(10, 5)), group(3, time(9, 30), time(10, 6))], previous=previous)
        self.assertEqual(assigned, [])
        self.assertEqual(report["status"], "needs_review")
        self.assertTrue(report["decisions"][0]["cached"])
        self.calls["decide_with_jev"].assert_not_called()
        self.calls["save"].assert_not_called()

    def test_failed_visual_receipt_can_be_resolved_by_time_without_resending(self):
        previous = {"1": {"context_hash": "old_531_candidates", "status": "error", "evidence": {
            "status": "error", "entry_trigger_id": None, "script_run_id": 90}}}
        assigned, report = self.assign([receipt(1, time(10, 3))], [group(1, time(10), time(10, 5))], previous=previous)
        self.assertEqual(len(assigned), 1)
        self.assertEqual(report["decisions"][0]["method"], "time")
        self.assertEqual(report["decisions"][0]["supersedes_failed_script_run_id"], 90)
        self.calls["decide_with_jev"].assert_not_called()
        self.calls["save"].assert_called_once()

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
        self.calls["collect_groups"].assert_not_called()
        self.calls["get_location_endpoint"].assert_not_called()

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

    def test_current_batch_open_entries_do_not_create_extra_transaction_candidates(self):
        with patch.object(store, "list_overlapping_groups", return_value=([], [])), \
             patch.object(workflow.repositories, "get_grouping_batch", return_value={"status": "success", "location_id": 2, "result_payload": {"groups": [{"entry": [1], "exit": [2]}], "open_entries": [3]}}), \
             patch.object(workflow.repositories, "get_trigger", side_effect=lambda db, tid: {"trigger_time": {1: time(10), 2: time(10, 5), 3: time(9, 30)}[tid]}):
            groups = assignment.collect_groups(Mock(), location_id=2, start=time(10), end=time(10, 5), batch_id=1, current_group={"entry": [1], "exit": [2]})
        self.assertEqual([row["entry_trigger_id"] for row in assignment.active_groups(groups, time(10, 3))], [1])

    def test_old_open_entries_in_current_batch_snapshot_are_also_excluded(self):
        with patch.object(store, "list_overlapping_groups", return_value=([], [])), \
             patch.object(workflow.repositories, "get_grouping_batch", return_value={"status": "success", "location_id": 2, "result_payload": {"groups": [{"entry": [1], "exit": [2]}], "open_entries": [3]}}), \
             patch.object(workflow.repositories, "get_trigger", side_effect=lambda db, tid: {"trigger_time": {1: time(10), 2: time(10, 5), 3: time(8)}[tid]}):
            groups = assignment.collect_groups(Mock(), location_id=2, start=time(10, 3), end=time(10, 5), batch_id=1,
                                               current_group={"entry": [1], "exit": [2]})
        self.assertEqual([row["entry_trigger_id"] for row in groups], [1])

    def test_standalone_trigger_cannot_be_invented_as_a_current_group(self):
        with patch.object(store, "list_overlapping_groups", return_value=([], [])), \
             patch.object(workflow.repositories, "get_grouping_batch", return_value={"status": "success", "location_id": 2, "result_payload": {"groups": [], "open_entries": [1]}}):
            with self.assertRaisesRegex(ValueError, "existing group"):
                assignment.collect_groups(Mock(), location_id=2, start=time(10), end=time(10, 5),
                                          batch_id=1, current_group={"entry": [1], "exit": [2]})

    def test_entry_only_group_is_never_sent_for_assignment(self):
        with self.assertRaisesRegex(ValueError, "at least one exit"):
            assignment.assign(Mock(), batch_id=1, location_id=2, group={"entry": [1], "exit": []},
                              start_time=time(10), end_time=time(10, 5))

    def test_recent_ungrouped_identity_does_not_force_a_paid_request(self):
        assigned, report = self.assign([receipt(1, time(10, 3))],
            [group(1, time(10), time(10, 5)), group(3, time(10, 1), None)])
        self.assertEqual(len(assigned), 1)
        self.assertEqual(report["decisions"][0]["method"], "time")
        self.calls["decide_with_jev"].assert_not_called()

    def test_group_membership_in_another_batch_does_not_validate_current_group(self):
        existing = {"batch_id": 9, "group_key": "g", "entry_ids": "1", "exit_ids": "2",
                    "entry_time": time(10), "exit_time": time(10, 5)}
        with patch.object(store, "list_overlapping_groups", return_value=([existing], [])), \
             patch.object(workflow.repositories, "get_grouping_batch", return_value={"status": "success", "location_id": 2, "result_payload": {"groups": []}}):
            with self.assertRaisesRegex(ValueError, "existing group"):
                assignment.collect_groups(Mock(), location_id=2, start=time(10), end=time(10, 5),
                                          batch_id=1, current_group={"entry": [1], "exit": [2]})

    def test_active_candidate_requires_exit_membership_not_just_exit_timestamp(self):
        value = group(1, time(10), time(10, 5))
        value["exit_trigger_ids"] = []
        self.assertEqual(assignment.active_groups([value], time(10, 3)), [])

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
            checkpoint = stack.enter_context(patch.object(store, "checkpoint_run"))
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
        finish.assert_not_called()
        self.assertEqual(checkpoint.call_args.args[2]["persistence_status"], "awaiting_receipt_commit")


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
