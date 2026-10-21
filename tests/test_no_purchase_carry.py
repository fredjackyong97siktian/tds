from contextlib import ExitStack
from datetime import datetime, timedelta
import json
import unittest
from unittest.mock import Mock, patch

from app.services import identifier_capture
from app.services import jev_pipeline as pipeline
from app.services import workflow_service as workflow


def trigger(number):
    return {"trigger_id": number, "trigger_time": datetime(2026, 10, 21, 11) + timedelta(minutes=number),
            "frames": [{"frame_index": 1, "image_url": f"https://example.test/entrance/{number}.jpg"}]}


def identifier(number):
    return {**trigger(number), "section": "identifier", "recorder_channel": "3", "exit_direction": "left",
            "script_run_id": 100 + number, "frames": [
                {"frame_index": index + 1, "sample_time": time.isoformat(), "status": "ok",
                 "image_url": f"https://example.test/identifier/{number}/{index}.jpg"}
                for index, time in enumerate(identifier_capture.sample_times(trigger(number)["trigger_time"]))]}


def answer(status, reason):
    def choice(selected, options):
        return {"type": "choice", "choice": selected, "confidence": 0.9,
                "probabilities": {value: 0.94 if value == selected else 0.06 / (len(options) - 1) for value in options}}
    return {"answers": {"carry_status": choice(status, {"suspicious", "reasonable", "insufficient_evidence"}),
                        "carry_reason": choice(reason, pipeline.NO_PURCHASE_REASONS)}}


class NoPurchaseCarryTests(unittest.TestCase):
    def test_one_combined_request_includes_all_reference_and_identifier_frames(self):
        with patch.object(pipeline.jev, "download_image", return_value="data:image/jpeg;base64,AA=="):
            body, urls = pipeline.no_purchase_request([trigger(1)], [trigger(2), trigger(3)], [identifier(2), identifier(3)])
        self.assertEqual(len(urls), 27)
        state = body["state"]
        self.assertEqual(len(state["entry_sequences"]), 1)
        self.assertEqual(len(state["exit_sequences"]), 2)
        self.assertEqual([row["exit_trigger_id"] for row in state["identifier_sequences"]], [2, 3])
        self.assertEqual(state["identifier_sequences"][0]["recorder_channel"], "3")
        self.assertEqual(state["identifier_sequences"][0]["exit_direction"], "left")
        self.assertEqual(len(state["identifier_sequences"][1]["frames"]), 12)
        instructions = body["questions"]["carry_status"]["instructions"]
        self.assertIn("one orange plus one pink bottle", instructions)
        self.assertIn("Only that same orange bottle", instructions)
        self.assertIn("do not double-count", instructions)
        self.assertIn("do not apply an identifier camera direction to the entrance camera", instructions)
        self.assertIn("yellow container alone is not suspicious", instructions)
        self.assertNotIn("yellow_bag", body["questions"]["carry_reason"]["criteria"])

    def test_every_exit_needs_complete_identifier_evidence(self):
        incomplete = identifier(3)
        incomplete["frames"].pop()
        for sequences in [[identifier(2)], [identifier(2), incomplete]]:
            with self.subTest(sequences=len(sequences)), patch.object(pipeline.jev, "download_image") as download, self.assertRaises(ValueError):
                pipeline.no_purchase_request([trigger(1)], [trigger(2), trigger(3)], sequences)
            download.assert_not_called()

    def test_zero_paid_visits_capture_each_exit_then_use_one_jev_call(self):
        for status, reason, expected_hit in [("suspicious", "unexplained_carry_change", True),
                                            ("reasonable", "consistent_with_entry", False),
                                            ("insufficient_evidence", "unclear_images", False)]:
            with self.subTest(status=status), ExitStack() as stack:
                db = Mock()
                stack.enter_context(patch.object(workflow.repositories, "create_script_run_started", return_value=9))
                stack.enter_context(patch.object(workflow.repositories, "get_trigger", side_effect=lambda db, number:
                                             {"location_id": 2, "trigger_time": trigger(number)["trigger_time"]}))
                assets = [{"frames": [{"status": "ok", "frame_index": 1, "image_url": "https://example.test/reference.jpg"}]}]
                stack.enter_context(patch.object(workflow.repositories, "list_trigger_frame_assets", return_value=assets))
                stack.enter_context(patch.object(workflow.repositories, "get_app_setting", return_value="6"))
                stack.enter_context(patch.object(workflow.repositories, "merge_script_run_stdout_fields", side_effect=lambda db, run, value: json.dumps(value, default=str)))
                finish = stack.enter_context(patch.object(workflow.repositories, "finish_script_run"))
                capture = stack.enter_context(patch.object(identifier_capture, "capture_exit", side_effect=lambda db, **kw: identifier(kw["exit_trigger"]["trigger_id"])))
                stack.enter_context(patch.object(pipeline.jev, "download_image", return_value="data:image/jpeg;base64,AA=="))
                call = stack.enter_context(patch.object(pipeline, "logged_call", return_value=(answer(status, reason), {"usage": {"actual_cost_usd": 0.001}})))
                result = pipeline.evaluate_carry(db, batch_id=10, group_key="1", location_id=2,
                                                group={"entry": [1], "exit": [2, 3]}, transactions=[])
                self.assertEqual([item.kwargs["exit_trigger"]["trigger_id"] for item in capture.call_args_list], [2, 3])
                self.assertTrue(all(item.kwargs["parent_script_run_id"] == 9 for item in capture.call_args_list))
                call.assert_called_once()
                self.assertEqual(result["hit"], expected_hit)
                self.assertEqual(workflow._carry_factor_hit(result, low_purchase=True), expected_hit)
                self.assertTrue(result["no_purchase_identifier_check"])
                self.assertEqual(len(result["identifier_sequences"]), 2)
                self.assertEqual(result["cost_detail"]["actual_cost_usd"], 0.001)
                self.assertEqual(finish.call_args.kwargs["status"], "success")

    def test_no_paid_path_uses_jev_even_if_regular_carry_inherits_grouping_model(self):
        with patch.object(workflow, "_current_jev_stage_provider", return_value="inherit"), \
                patch.object(pipeline, "evaluate_carry", return_value={"status": "reasonable"}) as evaluate, \
                patch.object(workflow, "_call_grouping_vision") as inherited:
            workflow._evaluate_carry_item_signal_with_ai(Mock(), batch_id=10, group_key="1", location_id=2,
                trigger_ids=[1, 2], group={"entry": [1], "exit": [2]}, transactions=[], total_quantity=0, total_value=0)
        evaluate.assert_called_once()
        inherited.assert_not_called()

    def test_identifier_failure_never_sends_partial_evidence_to_jev(self):
        with patch.object(workflow.repositories, "create_script_run_started", return_value=9), \
                patch.object(workflow.repositories, "get_trigger", return_value={"location_id": 2}), \
                patch.object(workflow.repositories, "list_trigger_frame_assets", return_value=[{"frames": [{"status": "ok", "image_url": "https://example.test/frame.jpg"}]}]), \
                patch.object(workflow.repositories, "get_app_setting", return_value="6"), \
                patch.object(workflow.repositories, "merge_script_run_stdout_fields", return_value="{}"), \
                patch.object(workflow.repositories, "finish_script_run") as finish, \
                patch.object(identifier_capture, "capture_exit", side_effect=RuntimeError("Missing identifier camera")), \
                patch.object(pipeline, "logged_call") as call:
            result = pipeline.evaluate_carry(Mock(), batch_id=10, group_key="1", location_id=2,
                                            group={"entry": [1], "exit": [2]}, transactions=[])
        call.assert_not_called()
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertFalse(result["hit"])
        self.assertIn("Missing identifier camera", result["error"])
        self.assertEqual(finish.call_args.kwargs["status"], "failed")


if __name__ == "__main__":
    unittest.main()
