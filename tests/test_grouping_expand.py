from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from contextlib import ExitStack
import unittest

from pydantic import ValidationError

from app.services import grouping_expand as expand
from app.services import workflow_service as workflow
from app.services import jev_pipeline
from app.routers.settings import GroupingExpandDurationRequest, update_grouping_expand_duration


def trigger(number, hour=21, minute=0, identity=False):
    return {"trigger_id": number, "trigger_time": datetime(2026, 10, 21, hour, minute),
            "phone_entry_id": number if identity else None, "credit_card_entry_id": None,
            "frames": [{"index": 1, "image_url": f"https://example.test/{number}.jpg", "status": "ok"}]}


END = datetime(2026, 10, 21, 22, 30)


class ExpandPlanningTests(unittest.TestCase):
    def test_non_customer_candidates_and_entries_are_excluded_but_unchecked_remain(self):
        rows = [trigger(1, 21, 0, True), trigger(2, 21, 2), trigger(3, 21, 3), trigger(4, 21, 4)]
        rows[1]["motion_person_present"] = 0
        rows[2]["motion_person_present"] = None
        rows[3]["motion_person_present"] = True
        plans = expand.plan(rows, [], window_end=END, minutes=20)
        self.assertEqual([row["trigger_id"] for row in plans[0]["candidates"]], [3, 4])
        rows[0]["motion_person_present"] = False
        self.assertEqual(expand.plan(rows, [], window_end=END, minutes=20), [])

    def test_twenty_minutes_is_capped_by_running_period_end(self):
        rows = [trigger(1, 22, 23, True), trigger(2, 22, 29), trigger(3, 22, 30), trigger(4, 22, 35)]
        plans = expand.plan(rows, [], window_end=END, minutes=20)
        self.assertEqual([row["trigger_id"] for row in plans[0]["candidates"]], [2])
        self.assertEqual(plans[0]["cutoff"], END)

    def test_duration_limit_includes_exact_twenty_minutes_but_not_later(self):
        plans = expand.plan([trigger(1, 21, 0, True), trigger(2, 21, 20), trigger(3, 21, 21)], [], window_end=END, minutes=20)
        self.assertEqual([row["trigger_id"] for row in plans[0]["candidates"]], [2])

    def test_includes_identity_nonidentity_grouped_and_ungrouped_candidates(self):
        rows = [trigger(1, 21, 0, True), trigger(2, 21, 2, True), trigger(3, 21, 4), trigger(4, 21, 6)]
        plans = expand.plan(rows, [{"entry": [2], "exit": [3]}], window_end=END, minutes=20)
        self.assertEqual(len(plans), 1)
        self.assertEqual([row["trigger_id"] for row in plans[0]["candidates"]], [2, 3, 4])

    def test_only_unmatched_identity_entries_are_seeds(self):
        rows = [trigger(1, 21, 0, True), trigger(2, 21, 2), trigger(3, 21, 4, True), trigger(4, 21, 6)]
        plans = expand.plan(rows, [{"entry": [1], "exit": [2]}], window_end=END, minutes=20)
        self.assertEqual([plan["entry"]["trigger_id"] for plan in plans], [3])
        rows[2]["phone_entry_id"] = None
        rows[2]["credit_card_entry_id"] = 900
        self.assertEqual(expand.plan(rows, [], window_end=END, minutes=20)[1]["entry"]["trigger_id"], 3)

    def test_candidates_cannot_precede_entry_or_equal_its_time(self):
        rows = [trigger(1, 21, 10, True), trigger(2, 21, 9), trigger(3, 21, 10), trigger(4, 21, 11)]
        plans = expand.plan(rows, [], window_end=END, minutes=20)
        self.assertEqual([row["trigger_id"] for row in plans[0]["candidates"]], [4])

    def test_missing_frames_issues_and_out_of_window_candidates_do_not_make_calls(self):
        rows = [trigger(1, 22, 23, True), trigger(2, 22, 24), trigger(3, 22, 25), trigger(4, 22, 31)]
        rows[1]["frames"] = []
        self.assertEqual(expand.plan(rows, [], window_end=END, minutes=20, excluded_ids={3}), [])

    def test_custom_duration_changes_candidate_window(self):
        rows = [trigger(1, 21, 0, True), trigger(2, 21, 9), trigger(3, 21, 15)]
        self.assertEqual(len(expand.plan(rows, [], window_end=END, minutes=10)[0]["candidates"]), 1)
        self.assertEqual(len(expand.plan(rows, [], window_end=END, minutes=20)[0]["candidates"]), 2)

    def test_settings_default_and_validation(self):
        with patch.object(expand.repositories, "get_app_setting", return_value=None):
            self.assertEqual(expand.duration_minutes(Mock()), 20)
        for value in (0, 121, 20.5, True, "20"):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                GroupingExpandDurationRequest(duration_minutes=value)
        with patch.object(expand.repositories, "set_app_setting") as save:
            update_grouping_expand_duration(GroupingExpandDurationRequest(duration_minutes=35), Mock())
        self.assertEqual(save.call_args.args[1:], (expand.SETTING_KEY, "35"))


class ExpandExecutionTests(unittest.TestCase):
    def run_expand(self, matches=None, error=False):
        self.calls = {}
        rows = [trigger(1, 21, 0, True), trigger(2, 21, 2, True), trigger(3, 21, 3), trigger(4, 21, 4), trigger(5, 21, 5)]
        original = [{"group_id": 1, "entry": [2], "exit": [3]}]
        selected = iter(matches if matches is not None else [[1]])
        with ExitStack() as stack:
            for target, name, value in [
                (expand.repositories, "get_app_setting", "20"), (expand.repositories, "_column_exists", True),
                (expand.repositories, "list_confirmed_non_customer_trigger_ids", set()),
                (workflow, "_create_gemini_script_run", 77), (expand.repositories, "finish_script_run", None),
                (expand.repositories, "merge_script_run_stdout_fields", "{}"),
                (expand.repositories, "annotate_latest_script_run_call", None),
                (jev_pipeline, "adjacent_request", ({"questions": {}}, [])),
                (jev_pipeline, "logged_call", ({}, {"usage": {"actual_cost_usd": 0.001}})),
            ]:
                self.calls[name] = stack.enter_context(patch.object(target, name, return_value=value))
            stack.enter_context(patch.object(jev_pipeline, "adjacent_result", side_effect=lambda response, count: {
                "matched_candidates": next(selected), "confidence": 0.2, "reason": "Selected match", "jev_decision": {"choice": "candidate_1"}}))
            if error:
                self.calls["logged_call"].side_effect = RuntimeError("Provider timeout")
            result = expand.run(Mock(), batch_id=9, location_id=2, triggers=rows, groups=original, window_end=END)
        self.assertEqual(original, [{"group_id": 1, "entry": [2], "exit": [3]}])
        return result

    def test_uses_adjacent_prompt_and_preserves_existing_group_when_exit_is_shared(self):
        groups, audit, notes = self.run_expand()
        self.assertEqual(groups[0]["entry"], [1])
        self.assertEqual(groups[0]["exit"], [2])
        self.assertEqual(groups[0]["shared_trigger_ids"], [2])
        self.assertEqual(groups[0]["group_id"], 2)
        self.assertEqual(groups[0]["source"], "jev_grouping_expand")
        self.assertEqual(audit["planned_attempts"], 2)
        self.assertEqual(len(audit["attempts"]), 1)
        self.calls["adjacent_request"].assert_called_once()
        self.assertEqual(self.calls["finish_script_run"].call_args.kwargs["status"], "success")

    def test_continues_after_none_and_accepts_both_like_adjacent(self):
        groups, audit, _notes = self.run_expand(matches=[[], [1, 2]])
        self.assertEqual(groups[0]["exit"], [4, 5])
        self.assertEqual(len(audit["attempts"]), 2)

    def test_no_matches_leaves_entry_unresolved(self):
        groups, audit, _notes = self.run_expand(matches=[[], []])
        self.assertEqual(groups, [])
        self.assertEqual(len(audit["attempts"]), 2)

    def test_api_failure_is_logged_without_accepting_a_later_exit(self):
        groups, audit, _notes = self.run_expand(error=True)
        self.assertEqual(groups, [])
        self.assertEqual(audit["status"], "partial_error")
        self.assertEqual(len(audit["attempts"]), 1)
        self.assertIn("Provider timeout", audit["attempts"][0]["error"])
        self.assertEqual(self.calls["finish_script_run"].call_args.kwargs["status"], "failed")

    def test_no_candidates_skips_even_script_run_creation(self):
        with patch.object(expand.repositories, "get_app_setting", return_value="20"), \
             patch.object(expand.repositories, "list_confirmed_non_customer_trigger_ids", return_value=set()), \
             patch.object(workflow, "_create_gemini_script_run") as create:
            groups, audit, _notes = expand.run(Mock(), batch_id=9, location_id=2, triggers=[trigger(1, 22, 29, True)], groups=[], window_end=END)
        self.assertEqual(groups, [])
        self.assertEqual(audit["status"], "skipped")
        create.assert_not_called()

    def test_missing_migration_stops_before_paid_api_or_membership_overwrite(self):
        with patch.object(expand.repositories, "get_app_setting", return_value="20"), \
             patch.object(expand.repositories, "list_confirmed_non_customer_trigger_ids", return_value=set()), \
             patch.object(expand.repositories, "_column_exists", return_value=False), \
             patch.object(jev_pipeline, "logged_call") as call:
            with self.assertRaisesRegex(RuntimeError, "shared_memberships.sql"):
                expand.run(Mock(), batch_id=9, location_id=2, triggers=[trigger(1, 21, 0, True), trigger(2, 21, 2)], groups=[], window_end=END)
        call.assert_not_called()

    def test_live_non_customer_flag_excludes_stale_snapshot_before_api_call(self):
        rows = [trigger(1, 21, 0, True), trigger(2, 21, 2)]
        for excluded in ({1}, {2}):
            with self.subTest(excluded=excluded), \
                 patch.object(expand.repositories, "get_app_setting", return_value="20"), \
                 patch.object(expand.repositories, "list_confirmed_non_customer_trigger_ids", return_value=excluded.copy()), \
                 patch.object(jev_pipeline, "logged_call") as call:
                groups, audit, _notes = expand.run(Mock(), batch_id=9, location_id=2, triggers=rows, groups=[], window_end=END)
            self.assertEqual(groups, [])
            self.assertEqual(audit["excluded_non_customer_trigger_ids"], sorted(excluded))
            self.assertEqual(audit["planned_attempts"], 0)
            call.assert_not_called()

    def test_non_customer_lookup_is_scoped_and_uses_only_confirmed_negatives(self):
        db = Mock()
        with patch.object(workflow.repositories, "_fetch_all_dicts", return_value=[{"id": 2}]):
            ids = workflow.repositories.list_confirmed_non_customer_trigger_ids(db, [1, 2, 2])
        self.assertEqual(ids, {2})
        sql, params = db.execute.call_args.args
        self.assertIn("motion_person_present = 0", str(sql))
        self.assertEqual(params["trigger_ids"], [1, 2])
        db.reset_mock()
        self.assertEqual(workflow.repositories.list_confirmed_non_customer_trigger_ids(db, []), set())
        db.execute.assert_not_called()

    def test_shared_memberships_keep_each_group_role_and_frames(self):
        db = Mock()
        summary = {"groups": [{"group_id": 1, "entry": [2], "exit": [3]},
                              {"group_id": 2, "entry": [1], "exit": [2]}]}
        frames = {"frames": [{"image_url": "https://example.test/shared.jpg"}]}
        with patch.object(workflow.repositories, "list_grouping_items", return_value=[{"trigger_id": 2, "frame_payload": frames}]), \
             patch.object(workflow.repositories, "upsert_grouping_item") as save:
            workflow._persist_grouping_items_from_summary(db, batch_id=9, grouping_summary=summary)
        shared = [call.kwargs for call in save.call_args_list if call.kwargs["trigger_id"] == 2]
        self.assertEqual([(row["group_key"], row["role"]) for row in shared], [("1", "entry"), ("2", "exit")])
        self.assertTrue(all(row["frame_payload"] == frames and row["commit"] is False for row in shared))
        db.commit.assert_called_once()

    def test_failed_membership_write_rolls_back_replacement(self):
        db = Mock()
        with patch.object(workflow.repositories, "list_grouping_items", return_value=[]), \
             patch.object(workflow.repositories, "upsert_grouping_item", side_effect=RuntimeError("SQL error")), \
             self.assertLogs(workflow.logger, level="ERROR"):
            with self.assertRaisesRegex(RuntimeError, "SQL error"):
                workflow._persist_grouping_items_from_summary(db, batch_id=9, grouping_summary={"groups": [{"entry": [1], "exit": [2]}]})
        db.rollback.assert_called_once()
        db.commit.assert_not_called()

    def test_deleting_expand_group_does_not_ungroup_shared_entry_of_existing_group(self):
        summary = {"groups": [{"group_id": 1, "entry": [2], "exit": [3]},
                              {"group_id": 2, "entry": [1], "exit": [2]}], "unknown": []}
        with patch.object(workflow.repositories, "get_filter_confidence_result", return_value={"batch_id": 9, "group_key": "2"}), \
             patch.object(workflow.repositories, "get_grouping_batch", return_value={"result_payload": summary}), \
             patch.object(workflow.repositories, "update_grouping_batch") as update, \
             patch.object(workflow.repositories, "delete_grouping_items_for_group") as delete, \
             patch.object(workflow.repositories, "delete_filter_confidence_result"):
            result = workflow.delete_confidence_result_and_ungroup(Mock(), 20)
        self.assertEqual(result["ungrouped_trigger_ids"], [1])
        updated = update.call_args.args[2]["result_payload"]
        self.assertEqual(updated["unknown"], [1])
        self.assertEqual(updated["groups"], [{"group_id": 1, "entry": [2], "exit": [3]}])
        self.assertEqual(delete.call_args.kwargs["group_key"], "2")

    def test_expand_filter_uses_existing_allowed_grouping_script_name(self):
        filters, params = workflow.repositories._script_run_list_filters(script_name="grouping_expand", script_type="jev", model_name=None)
        self.assertEqual(params["script_name"], "grouping")
        self.assertEqual(params["presence_suffix"], "_grouping_expand")


if __name__ == "__main__":
    unittest.main()
