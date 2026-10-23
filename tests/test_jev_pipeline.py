import json
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from app.services import jev_pipeline as pipeline
from app.services import workflow_service as workflow
from app.services import session_analysis


def trigger(number, minutes=0, identity=False):
    return {"trigger_id": number, "trigger_time": datetime(2026, 10, 17, 12) + timedelta(minutes=minutes),
            "phone_entry_id": number if identity else None, "credit_card_entry_id": None,
            "frames": [{"index": 1, "image_url": f"https://example.test/{number}.jpg"}]}


def choice(selected, options, confidence=0.9):
    rest = (1 - 0.94) / max(1, len(options) - 1)
    return {"type": "choice", "choice": selected, "confidence": confidence,
            "probabilities": {value: 0.94 if value == selected else rest for value in options}}


def adjacent_answer(selected="both", confidence=0.9):
    return {"answers": {"matched_exits": choice(selected, ["candidate_1", "candidate_2", "both", "none"], confidence)}}


def carry_answer(status="suspicious", reason="yellow_bag", confidence=0.9):
    return {"answers": {
        "carry_status": choice(status, ["suspicious", "reasonable", "insufficient_evidence"], confidence),
        "carry_reason": choice(reason, pipeline.CARRY_REASONS),
    }}


class JevPipelineTests(unittest.TestCase):
    def test_carry_api_attempt_records_answer_usage_and_cost_on_same_run(self):
        response = carry_answer("reasonable", "consistent_with_entry")
        meta = {"provider": "tds_api_openrouter", "model": pipeline.jev.MODEL,
                "raw_response": response, "raw_usage": {"cost": 0.001},
                "usage": {"actual_cost_usd": 0.001}}
        db = Mock()
        with patch.object(workflow.settings, "openrouter_api_key", "test-key"), \
                patch.object(pipeline.jev, "call_openrouter", return_value=(response, meta)), \
                patch.object(workflow.repositories, "append_script_run_call", return_value=0) as append, \
                patch.object(workflow.repositories, "replace_script_run_call") as replace, \
                patch.object(workflow.repositories, "add_script_run_cost") as cost:
            pipeline.logged_call(db, 42, {"questions": {}}, ["https://example.test/frame.jpg"], description="carry")
        self.assertEqual(append.call_args.args[1], 42)
        saved = replace.call_args.args[3]
        self.assertEqual(saved["response"]["raw_response"], response)
        self.assertEqual(saved["response"]["raw_usage"]["cost"], 0.001)
        cost.assert_called_once_with(db, 42, cost_amount=0.001, cost_currency="USD", cost_source="openrouter_actual")

    def test_reported_zero_cost_is_distinct_from_missing_or_invalid_cost(self):
        for reported, expected in [(0, 0.0), (None, None), (-1, None), (float("nan"), None), (True, None)]:
            with self.subTest(reported=reported):
                amount, _ = workflow._openrouter_usage_cost({"usage": {"actual_cost_usd": reported}})
                self.assertEqual(amount, expected)

    def test_zero_cost_is_persisted_without_overwriting_previous_charges(self):
        db = Mock()
        with patch.object(workflow.repositories, "_script_run_has_cost_columns", return_value=True):
            workflow.repositories.add_script_run_cost(db, 42, cost_amount=0, cost_source="openrouter_actual")
        statement, params = db.execute.call_args.args
        self.assertIn("coalesce(cost_amount, 0) + :cost_amount", str(statement))
        self.assertEqual(params["cost_amount"], 0)
        db.commit.assert_called_once()

    def test_carry_and_jev_filters_select_matching_runs(self):
        filters, params = workflow.repositories._script_run_list_filters(
            script_name="carry_confidence", script_type="jev", model_name=None)
        self.assertIn("model_name like 'openrouter-jev-%'", filters)
        self.assertEqual(params["script_name"], "carry_confidence")

    def test_adjacent_both_option_and_real_image_parts(self):
        with patch.object(pipeline.jev, "download_image", return_value="data:image/jpeg;base64,AA=="):
            body, urls = pipeline.adjacent_request(trigger(1, identity=True), [trigger(2, 1), trigger(3, 2)])
        self.assertIn("both", body["questions"]["matched_exits"]["criteria"])
        self.assertEqual(len(body["questions"]), 1)
        self.assertEqual(len(urls), 3)
        self.assertEqual(body["state"]["candidates"][1]["trigger_id"], 3)
        result = pipeline.adjacent_result(adjacent_answer(), 2)
        self.assertEqual(result["matched_candidates"], [1, 2])
        self.assertIsNone(result["matched_candidate"])

    def test_both_is_not_valid_for_one_candidate(self):
        with self.assertRaises(ValueError):
            pipeline.adjacent_result(adjacent_answer(), 1)

    def test_adjacent_uses_selected_option_regardless_of_confidence(self):
        self.assertEqual(pipeline.adjacent_result(adjacent_answer("none"), 2)["matched_candidates"], [])
        for option, expected in [("candidate_1", [1]), ("candidate_2", [2]), ("both", [1, 2]), ("none", [])]:
            with self.subTest(option=option):
                result = pipeline.adjacent_result(adjacent_answer(option, confidence=0.0), 2)
                self.assertEqual(result["matched_candidates"], expected)
                self.assertIsNone(result["minimum_confidence"])
                self.assertEqual(result["confidence"], 0.0)

    def test_single_candidate_and_unknown_choice_validation(self):
        answer = {"answers": {"matched_exits": choice("candidate_1", ["candidate_1", "none"])}}
        self.assertEqual(pipeline.adjacent_result(answer, 1)["matched_candidates"], [1])
        answer["answers"]["matched_exits"]["choice"] = "candidate_9"
        with self.assertRaises(ValueError):
            pipeline.adjacent_result(answer, 1)

    def test_both_exits_are_persisted_and_cannot_be_reused_by_next_entry(self):
        inputs = [trigger(1, identity=True), trigger(2, 0.5, identity=True), trigger(3, 1), trigger(4, 2)]
        result = pipeline.adjacent_result(adjacent_answer(confidence=0.0), 2)
        with patch.object(workflow, "_find_resumable_adjacent_result", return_value=None), patch.object(workflow, "_current_jev_stage_provider", return_value="jev"), patch.object(workflow, "_create_gemini_script_run", return_value=7), patch.object(workflow, "_verify_entry_groups_against_candidates_batch", return_value=({1: result}, {}, {}, None)), patch.object(workflow.repositories, "get_app_setting", return_value=None), patch.object(workflow.repositories, "annotate_latest_script_run_call") as annotate, patch.object(workflow.repositories, "merge_script_run_stdout_fields", return_value="{}"), patch.object(workflow.repositories, "finish_script_run"), patch.object(workflow.time, "sleep"):
            remaining, groups, *_ = workflow._run_grouping_adjacent_pass(Mock(), batch_id=10, location_id=2, trigger_inputs=inputs, model_name="mimo", resize_scale=None)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["entry"], [1])
        self.assertEqual(groups[0]["exit"], [3, 4])
        self.assertEqual([row["trigger_id"] for row in remaining], [2])
        self.assertEqual(annotate.call_args_list[0].args[2]["matched_trigger_ids"], [3, 4])

    def test_carry_sends_entry_all_exits_and_full_receipt_details_in_one_request(self):
        transactions = [{"receipt_number": "r1", "total_items": 2, "details": [{"item_name": "Drink 1.5L", "quantity": 2}]}]
        with patch.object(pipeline.jev, "download_image", return_value="data:image/jpeg;base64,AA=="):
            body, urls = pipeline.carry_request([trigger(1)], [trigger(2, 1), trigger(3, 2)], transactions)
        self.assertEqual(len(body["state"]["exit_sequences"]), 2)
        self.assertEqual(body["state"]["paid_transactions_in_visit_window"], transactions)
        self.assertIn("not_confirmed", body["state"]["receipt_attribution"])
        self.assertEqual(len(body["questions"]), 2)
        self.assertEqual(len(urls), 3)
        self.assertNotIn("entry_carry", body["state"])

    def test_suspicious_only_is_a_hit(self):
        for status, reason in [("suspicious", "bag_enlarged"), ("reasonable", "consistent_with_entry"), ("insufficient_evidence", "unclear_images")]:
            result = pipeline.carry_result(carry_answer(status, reason))
            self.assertEqual(result["hit"], status == "suspicious")
            self.assertEqual(workflow._carry_factor_hit(result, low_purchase=True), status == "suspicious")
            self.assertEqual(result["insufficient_evidence"], status == "insufficient_evidence")

    def test_carry_preserves_selected_status_and_reason_without_overrides(self):
        result = pipeline.carry_result(carry_answer("reasonable", "yellow_bag"))
        self.assertEqual(result["status"], "reasonable")
        self.assertEqual(result["reason"], "yellow_bag")
        result = pipeline.carry_result(carry_answer("suspicious", confidence=0.2))
        self.assertEqual(result["status"], "suspicious")
        self.assertTrue(result["hit"])

    def test_all_carry_choices_are_honored_even_at_zero_confidence(self):
        for status, reason in [("suspicious", "bag_enlarged"), ("reasonable", "consistent_with_entry"), ("insufficient_evidence", "unclear_images")]:
            with self.subTest(status=status):
                response = carry_answer(status, reason, confidence=0.0)
                response["answers"]["carry_reason"]["confidence"] = 0.0
                result = pipeline.carry_result(response)
                self.assertEqual(result["status"], status)
                self.assertEqual(result["reason"], reason)
                self.assertEqual(result["hit"], status == "suspicious")
                self.assertEqual(result["confidence"], 0.0)
                self.assertIsNone(result["minimum_confidence"])

    def test_reported_no_purchase_reasonable_at_049_stays_reasonable(self):
        response = carry_answer("reasonable", "consistent_with_entry", confidence=0.49)
        response["answers"]["carry_reason"] = choice("consistent_with_entry", pipeline.NO_PURCHASE_REASONS, confidence=0.7)
        result = pipeline.carry_result(response, reason_options=pipeline.NO_PURCHASE_REASONS)
        self.assertEqual(result["status"], "reasonable")
        self.assertFalse(result["insufficient_evidence"])
        self.assertFalse(result["hit"])

    def test_rollback_keeps_legacy_carry_low_purchase_policy(self):
        self.assertTrue(workflow._carry_factor_hit({"insufficient_evidence": True, "hit": False, "source": "mimo_carry"}, True))
        self.assertFalse(workflow._carry_factor_hit({"insufficient_evidence": True, "hit": False, "source": "mimo_carry"}, False))
        with patch.object(workflow, "_current_jev_stage_provider", return_value="inherit"), patch.object(workflow, "_current_grouping_provider", return_value="openrouter-mimo-v2.6-flash"):
            self.assertEqual(workflow._grouping_stage_label(Mock(), "adjacent"), "openrouter-mimo-v2.6-flash_grouping_adjacent")

    def test_resume_does_not_reuse_mimo_result_after_switch_to_jev(self):
        prior = {"id": 7, "model_name": "openrouter-mimo-v2.6-flash_grouping_adjacent", "stdout_log": json.dumps({"input_trigger_ids": [1], "matched_groups": [], "remaining_trigger_ids": [1]})}
        with patch.object(workflow.repositories, "get_latest_successful_grouping_stage_script_run", return_value=prior), patch.object(workflow, "_grouping_stage_label", return_value="openrouter-jev-1.14_grouping_adjacent"):
            self.assertIsNone(workflow._find_resumable_adjacent_result(Mock(), batch_id=1, trigger_inputs=[trigger(1)]))

    def test_adjacent_override_does_not_change_direct_provider(self):
        with patch.object(workflow.repositories, "get_app_setting", side_effect=lambda db, key: "jev" if key == "adjacent_provider" else "openrouter-mimo-v2.6-flash"):
            self.assertEqual(workflow._grouping_stage_label(Mock(), "adjacent"), "openrouter-jev-1.14_grouping_adjacent")
            self.assertEqual(workflow._grouping_stage_label(Mock(), "direct"), "openrouter-mimo-v2.6-flash_grouping_direct")

    def test_jev_carry_failure_is_unclear_and_script_run_is_failed(self):
        with patch.object(session_analysis, "PAUSED", False), patch.object(workflow.repositories, "create_script_run_started", return_value=9), patch.object(workflow.repositories, "get_trigger", return_value={"location_id": 2}), patch.object(workflow.repositories, "list_trigger_frame_assets", return_value=[]), patch.object(workflow.repositories, "get_app_setting", return_value="6"), patch.object(workflow.repositories, "merge_script_run_stdout_fields", return_value="{}"), patch.object(workflow.repositories, "finish_script_run") as finish:
            result = pipeline.evaluate_carry(Mock(), batch_id=1, group_key="1", location_id=2, group={"entry": [1], "exit": [2]}, transactions=[])
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertFalse(workflow._carry_factor_hit(result, True))
        self.assertEqual(finish.call_args.kwargs["status"], "failed")

    def test_carry_router_bypasses_text_only_provider(self):
        with patch.object(workflow, "_current_jev_stage_provider", return_value="jev"), patch.object(pipeline, "evaluate_carry", return_value={"status": "reasonable"}) as evaluate, patch.object(workflow, "_call_grouping_vision") as generic:
            workflow._evaluate_carry_item_signal_with_ai(Mock(), batch_id=1, group_key="1", location_id=2, trigger_ids=[1, 2], group={"entry": [1], "exit": [2]}, transactions=[{"total_items": 1}], total_quantity=1, total_value=100)
        evaluate.assert_called_once()
        generic.assert_not_called()

    def test_cost_is_recorded_before_typed_output_validation(self):
        with patch.object(workflow.settings, "openrouter_api_key", "test-key"), patch.object(workflow, "_call_vision_with_retry", return_value=({"answers": {}}, {"provider": "tds_api_openrouter", "usage": {"actual_cost_usd": 0.001}})), patch.object(workflow, "_record_openrouter_cost") as cost:
            response, _ = pipeline.logged_call(Mock(), 7, {"questions": {}}, [], description="test")
        cost.assert_called_once()
        with self.assertRaises(ValueError):
            pipeline.carry_result(response)

    def test_carry_success_loads_all_exits_and_writes_the_result_not_groups(self):
        assets = [{"frames": [{"frame_index": 1, "status": "ok", "image_url": "https://example.test/frame.jpg"}]}]
        db = Mock()
        with patch.object(workflow.repositories, "create_script_run_started", return_value=9), patch.object(workflow.repositories, "get_trigger", return_value={"location_id": 2}), patch.object(workflow.repositories, "list_trigger_frame_assets", return_value=assets) as load, patch.object(workflow.repositories, "get_app_setting", return_value="6"), patch.object(workflow.repositories, "merge_script_run_stdout_fields", side_effect=lambda db, run, payload: json.dumps(payload)), patch.object(workflow.repositories, "finish_script_run") as finish, patch.object(pipeline, "carry_request", return_value=({"questions": {}}, ["https://example.test/frame.jpg"])) as request, patch.object(pipeline, "logged_call", return_value=(carry_answer("reasonable", "consistent_with_entry"), {"usage": {"actual_cost_usd": 0.001}})):
            result = pipeline.evaluate_carry(db, batch_id=1, group_key="1", location_id=2, group={"entry": [1], "exit": [2, 3]}, transactions=[{"total_items": 1}])
        self.assertEqual([call.kwargs["trigger_id"] for call in load.call_args_list], [1, 2, 3])
        self.assertEqual(len(request.call_args.args[1]), 2)
        self.assertEqual(result["status"], "reasonable")
        self.assertEqual(finish.call_args.kwargs["status"], "success")
        db.execute.assert_not_called()

    def test_adjacent_failure_does_not_leave_a_running_script_or_fake_success(self):
        with patch.object(workflow, "_find_resumable_adjacent_result", return_value=None), patch.object(workflow, "_current_jev_stage_provider", return_value="jev"), patch.object(workflow, "_create_gemini_script_run", return_value=7), patch.object(workflow, "_verify_entry_groups_against_candidates_batch", side_effect=RuntimeError("HTTP 401")), patch.object(workflow.repositories, "merge_script_run_stdout_fields", return_value="{}"), patch.object(workflow.repositories, "finish_script_run") as finish:
            with self.assertRaisesRegex(RuntimeError, "401"):
                workflow._run_grouping_adjacent_pass(Mock(), batch_id=10, location_id=2, trigger_inputs=[trigger(1, identity=True), trigger(2, 1)], model_name="mimo", resize_scale=None)
        self.assertEqual(finish.call_args.kwargs["status"], "failed")

    def test_stage_settings_do_not_overwrite_global_grouping_provider(self):
        from app.routers.settings import JevStageProviderRequest, update_jev_stage_provider
        with patch.object(workflow.repositories, "set_app_setting") as save:
            update_jev_stage_provider("carry", JevStageProviderRequest(provider="inherit"), Mock())
        self.assertEqual(save.call_args.args[1:], ("carry_provider", "inherit"))


if __name__ == "__main__":
    unittest.main()
