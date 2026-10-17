import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from app.services import jev_pipeline as pipeline
from app.services import workflow_service as workflow


def trigger(number):
    return {"trigger_id": number, "trigger_time": datetime(2026, 10, 17, 12) + timedelta(minutes=number),
            "frames": [{"frame_index": 1, "image_url": f"https://example.test/{number}.jpg"}]}


class JevDirectVerificationTests(unittest.TestCase):
    def candidates(self):
        return [{"match_number": 0, "entry": trigger(1), "exit_triggers": [trigger(2), trigger(3)]},
                {"match_number": 1, "entry": trigger(4), "exit_triggers": [trigger(5)]}]

    def test_independent_pair_questions_and_weakest_exit_probability(self):
        response = {"answers": {
            "match_0_exit_0": {"type": "noul", "noul": 0.95},
            "match_0_exit_1": {"type": "noul", "noul": 0.2},
            "match_1_exit_0": {"type": "noul", "noul": 0.91},
        }}
        with patch.object(pipeline.jev, "download_image", return_value="data:image/jpeg;base64,AA=="), \
                patch.object(pipeline, "logged_call", return_value=(response, {"usage": {"actual_cost_usd": 0.001}})) as call:
            results, meta = pipeline.verify_direct_matches(Mock(), 42, self.candidates())
        self.assertEqual(results[0]["confidence"], 0.2)
        self.assertEqual(results[1]["confidence"], 0.91)
        self.assertEqual(len(results[0]["exit_checks"]), 2)
        body = call.call_args.args[2]
        self.assertEqual(body["model"], "typesafe/jev-1.14")
        self.assertEqual(len(body["questions"]), 3)
        question = body["questions"]["match_0_exit_0"]["instructions"]
        self.assertIn("`triggers.t1`", question)
        self.assertIn("`triggers.t2`", question)
        self.assertNotIn("`triggers.t4`", question)
        self.assertEqual(body["state"]["triggers"]["t1"]["frames"][0]["image"]["type"], "image_url")
        self.assertEqual(meta["usage"]["actual_cost_usd"], 0.001)

    def test_missing_typed_result_does_not_confirm_match(self):
        with patch.object(pipeline.jev, "download_image", return_value="data:image/jpeg;base64,AA=="), \
                patch.object(pipeline, "logged_call", return_value=({"answers": {}}, {})), \
                self.assertRaises(ValueError):
            pipeline.verify_direct_matches(Mock(), 42, self.candidates())

    def test_exit_must_be_later_and_cannot_be_entry_itself(self):
        for exit_id in [1, 0]:
            with self.subTest(exit_id=exit_id), patch.object(pipeline, "logged_call") as call, self.assertRaises(ValueError):
                pipeline.verify_direct_matches(Mock(), 42, [{"match_number": 0, "entry": trigger(1),
                                                            "exit_triggers": [trigger(exit_id)]}])
            call.assert_not_called()

    def test_selected_verification_frames_are_preserved(self):
        candidate = self.candidates()[0]
        candidate["entry"]["frames"].append({"frame_index": 2, "image_url": "https://example.test/best.jpg", "verification_selected": True})
        with patch.object(pipeline.jev, "download_image", return_value="data:image/jpeg;base64,AA==") as download, \
                patch.object(pipeline, "logged_call", return_value=({"answers": {
                    "match_0_exit_0": {"type": "noul", "noul": 0.9},
                    "match_0_exit_1": {"type": "noul", "noul": 0.9}}}, {})):
            pipeline.verify_direct_matches(Mock(), 42, [candidate])
        urls = [call.args[0] for call in download.call_args_list]
        self.assertIn("https://example.test/best.jpg", urls)
        self.assertNotIn("https://example.test/1.jpg", urls)

    def test_verification_has_own_script_run_and_records_failure(self):
        for fail in [False, True]:
            with self.subTest(fail=fail), \
                    patch.object(workflow.repositories, "create_script_run_started", return_value=70) as create, \
                    patch.object(workflow.repositories, "merge_script_run_stdout_fields", return_value="{}"), \
                    patch.object(workflow.repositories, "finish_script_run") as finish, \
                    patch.object(pipeline, "verify_direct_matches", side_effect=RuntimeError("API unavailable") if fail else None,
                                 return_value=({0: {"confidence": 0.95, "error": None}}, {})) as verify:
                db = Mock()
                result, _ = workflow._verify_jev_direct_grouping_matches_batch(db, 42, match_candidates=self.candidates())
                verify.assert_called_once_with(db, 70, self.candidates())
                self.assertEqual(create.call_args.kwargs["runner_payload"]["parent_script_run_id"], 42)
                self.assertEqual(create.call_args.kwargs["model_name"], "openrouter-jev-1.14_grouping_verification")
                self.assertEqual(finish.call_args.kwargs["status"], "failed" if fail else "success")
                self.assertEqual(result[0]["script_run_id"], 70)
                if fail:
                    self.assertIsNone(result[0]["confidence"])


if __name__ == "__main__":
    unittest.main()
