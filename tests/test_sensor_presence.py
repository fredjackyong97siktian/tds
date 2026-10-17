import json
import unittest
from unittest.mock import Mock, patch

from app.services import jev_pipeline as pipeline
from app.services import workflow_service as workflow


class SensorPresenceTests(unittest.TestCase):
    def test_presence_filter_uses_existing_rows_and_combines_with_provider(self):
        filters, params = workflow.repositories._script_run_list_filters(
            script_name="person_presence", script_type="jev", model_name=None)
        self.assertEqual(params["script_name"], "grouping")
        self.assertEqual(params["presence_suffix"], "_person_presence")
        self.assertIn("right(model_name, length(:presence_suffix)) = :presence_suffix", filters)
        self.assertIn("model_name like 'openrouter-jev-%'", filters)

    def test_other_script_filters_are_unchanged(self):
        filters, params = workflow.repositories._script_run_list_filters(
            script_name="carry_confidence", script_type=None, model_name=None)
        self.assertEqual(params, {"script_name": "carry_confidence"})
        self.assertEqual(filters, ["script_name = :script_name"])

    def setUp(self):
        self.db = Mock()
        self.row = {"trigger_id": 17, "location_id": 2}
        self.mocks = {}
        for name, value in {
            "create_script_run_started": 42,
            "finish_script_run": None,
            "set_trigger_motion_person_present": None,
            "merge_script_run_stdout_fields": "{}",
        }.items():
            patcher = patch.object(workflow.repositories, name, return_value=value)
            self.mocks[name] = patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(workflow, "_frame_urls_from_trigger_frame_asset", return_value=[{"image_url": "https://example.test/frame.jpg"}])
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(workflow, "_first_trigger_frame_payload", side_effect=lambda frames: frames)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_inherited_presence_call_logs_trigger_and_cost(self):
        meta = {"usage": {"actual_cost_usd": 0.001}}
        with patch.object(workflow, "_current_jev_stage_provider", return_value="inherit"), \
                patch.object(workflow, "_current_grouping_model_name", return_value="mimo"), \
                patch.object(workflow, "_call_grouping_vision", return_value=({"person_present": False, "confidence": 0.95}, meta)) as call, \
                patch.object(workflow, "_record_grouping_cost") as cost:
            self.assertFalse(workflow._check_motion_sensor_person_presence(self.db, self.row))
        self.assertEqual(call.call_args.kwargs["script_run_id"], 42)
        self.assertIn("in front of the store door", call.call_args.kwargs["prompt"])
        self.assertIn("need not open the door", call.call_args.kwargs["prompt"])
        cost.assert_called_once_with(self.db, 42, meta)
        created = self.mocks["create_script_run_started"].call_args.kwargs
        self.assertEqual(created["trigger_id"], 17)
        self.assertEqual(created["model_name"], "mimo_person_presence")
        self.assertEqual(self.mocks["finish_script_run"].call_args.kwargs["status"], "success")

    def test_jev_failure_is_logged_and_does_not_cache_absence(self):
        with patch.object(workflow, "_current_jev_stage_provider", return_value="jev"), \
                patch.object(pipeline, "check_person_presence", side_effect=RuntimeError("API failed")), \
                patch.object(workflow.logger, "warning"):
            self.assertTrue(workflow._check_motion_sensor_person_presence(self.db, self.row))
        self.assertEqual(self.mocks["finish_script_run"].call_args.kwargs["status"], "failed")
        self.mocks["set_trigger_motion_person_present"].assert_not_called()

    def test_uncertain_jev_check_keeps_trigger_without_caching(self):
        with patch.object(workflow, "_current_jev_stage_provider", return_value="jev"), \
                patch.object(pipeline, "check_person_presence", return_value={"person_present": True, "uncertain": True}):
            self.assertTrue(workflow._check_motion_sensor_person_presence(self.db, self.row))
        self.mocks["set_trigger_motion_person_present"].assert_not_called()

    def test_presence_defaults_to_existing_model(self):
        with patch.object(workflow.repositories, "get_app_setting", return_value=None):
            self.assertEqual(workflow._current_jev_stage_provider(self.db, "presence"), "inherit")

    def test_jev_receives_images_and_only_confident_absence_excludes(self):
        for selected, confidence, expected in [("absent", 0.9, False), ("absent", 0.3, True), ("present", 0.9, True), ("unclear", 0.9, True)]:
            with self.subTest(selected=selected, confidence=confidence):
                options = ["present", "absent", "unclear"]
                response = {"answers": {"person_presence": {"type": "choice", "choice": selected,
                    "confidence": confidence, "probabilities": {key: 0.9 if key == selected else 0.05 for key in options}}}}
                with patch.object(pipeline.jev, "download_image", return_value="data:image/jpeg;base64,AA=="), \
                        patch.object(pipeline, "logged_call", return_value=(response, {})) as call:
                    result = pipeline.check_person_presence(self.db, 42, ["https://example.test/frame.jpg"])
                self.assertEqual(result["person_present"], expected)
                body = call.call_args.args[2]
                question = body["questions"]["person_presence"]
                self.assertIn("the door of the store", question["instructions"])
                self.assertIn("in front of the store door", question["criteria"]["present"])
                self.assertIn("background", question["criteria"]["absent"])
                self.assertIn("cannot be identified", question["criteria"]["unclear"])
                self.assertIn("data:image/jpeg;base64,AA==", json.dumps(body))
                self.assertEqual(call.call_args.args[1], 42)


if __name__ == "__main__":
    unittest.main()
