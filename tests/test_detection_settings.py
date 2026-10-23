import unittest
from unittest.mock import Mock, patch

from sqlalchemy.exc import OperationalError

from app import repositories
from app.routers import settings as routes
from app.services import detection_analysis, detection_settings


class DetectionSettingsTests(unittest.TestCase):
    def test_missing_setting_defaults_to_mimo(self):
        with patch.object(repositories, "get_app_setting", return_value=None):
            self.assertEqual(detection_settings.current_model(Mock()), "xiaomi/mimo-v2.6-flash")

    def test_saved_model_is_not_overridden_by_environment_or_default(self):
        with patch.object(repositories, "get_app_setting", return_value="vendor/custom-vision"):
            self.assertEqual(detection_settings.current_model(Mock()), "vendor/custom-vision")

    def test_model_validation(self):
        self.assertEqual(detection_settings.validate_model(" vendor/model:free "), "vendor/model:free")
        for value in ("", "mimo", "https://example.com/model", "vendor/model\nignore", "vendor/../model", "a/" + "b" * 200):
            with self.subTest(value=value), self.assertRaises(ValueError):
                detection_settings.validate_model(value)

    def test_update_persists_only_detection_setting(self):
        db = Mock()
        with patch.object(repositories, "set_app_setting") as save:
            result = routes.update_detection_analysis_model(routes.DetectionAnalysisModelRequest(model="vendor/vision"), db)
        save.assert_called_once_with(db, "detection_analysis_model", "vendor/vision")
        self.assertEqual(result["model"], "vendor/vision")

    def test_old_name_constraint_falls_back_before_any_paid_request(self):
        db = Mock()
        error = OperationalError("insert", {}, Exception(3819, "Check constraint 'chk_script_run_name' is violated."))
        with patch.object(repositories, "create_script_run_started", side_effect=[error, 99]) as create:
            result = detection_analysis.create_run(db, model_name="vendor/vision", session_id=7, trigger_id=1, command="redacted")
        self.assertEqual(result, 99)
        self.assertEqual(create.call_args.kwargs["script_name"], "carry_confidence")
        self.assertEqual(create.call_args.kwargs["model_name"], "vendor/vision_detection_analysis")
        db.rollback.assert_called_once()

    def test_other_db_errors_are_not_retried_or_disguised(self):
        for original in (Exception(3819, "Check constraint 'another_constraint' violated"), Exception(2013, "Disconnected")):
            error = OperationalError("insert", {}, original)
            with patch.object(repositories, "create_script_run_started", side_effect=error) as create:
                with self.assertRaises(OperationalError):
                    detection_analysis.create_run(Mock(), model_name="vendor/vision", session_id=7, command="redacted")
                create.assert_called_once()

    def test_filters_include_legacy_detection_without_mixing_regular_carry(self):
        kwargs = dict(model_name=None, script_type=None)
        detection_filter, _ = repositories._script_run_list_filters(script_name="detection_analysis", **kwargs)
        self.assertIn("_detection_analysis", " ".join(detection_filter))
        carry_filter, _ = repositories._script_run_list_filters(script_name="carry_confidence", **kwargs)
        self.assertIn("<>", " ".join(carry_filter))


if __name__ == "__main__":
    unittest.main()
