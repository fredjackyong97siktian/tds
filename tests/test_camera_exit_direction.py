import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError

from app import repositories
from app.schemas import CctvCreate, CctvUpdate


class CameraExitDirectionTests(unittest.TestCase):
    def test_direction_supported_only_for_identifier_and_kiosk(self):
        for section in ["identifier", "kiosk"]:
            payload = CctvCreate(location_id=2, section=section, recorder_channel="3", exit_direction="right")
            self.assertEqual(payload.exit_direction, "right")
        with self.assertRaises(ValidationError):
            CctvCreate(location_id=2, section="entrance", exit_direction="left")
        with self.assertRaises(ValidationError):
            CctvCreate(location_id=2, section="identifier", exit_direction="invented")

    def test_existing_clients_do_not_clear_saved_direction(self):
        payload = CctvUpdate(location_id=2, section="identifier").model_dump(exclude_unset=True)
        db = Mock()
        repositories._save_cctv_exit_direction(db, 4, payload)
        db.execute.assert_not_called()

    def test_explicit_clear_removes_direction(self):
        db = Mock()
        repositories._save_cctv_exit_direction(db, 4, {"section": "identifier", "exit_direction": None})
        sql, params = db.execute.call_args.args
        self.assertIn("delete from", str(sql))
        self.assertEqual(params["key"], "cctv_exit_direction:4")
        db.commit.assert_not_called()

    def test_save_uses_camera_specific_key_and_callers_transaction(self):
        db = Mock()
        repositories._save_cctv_exit_direction(db, 4, {"section": "identifier", "exit_direction": "towards_camera"})
        sql, params = db.execute.call_args.args
        self.assertIn("on duplicate key update", str(sql))
        self.assertEqual(params, {"key": "cctv_exit_direction:4", "value": "towards_camera"})
        db.commit.assert_not_called()

    def test_read_directions_in_one_query_and_never_attach_to_entrance(self):
        rows = [{"id": 1, "section": "entrance"}, {"id": 2, "section": "identifier"},
                {"id": 3, "section": "kiosk"}]
        db = Mock()
        with patch.object(repositories, "_fetch_all_dicts", return_value=[
            {"setting_key": "cctv_exit_direction:2", "setting_value": "left"},
            {"setting_key": "cctv_exit_direction:3", "setting_value": "right"},
        ]):
            result = repositories._attach_cctv_exit_directions(db, rows)
        self.assertIsNone(result[0]["exit_direction"])
        self.assertEqual(result[1]["exit_direction"], "left")
        self.assertEqual(result[2]["exit_direction"], "right")
        db.execute.assert_called_once()
        self.assertEqual(db.execute.call_args.args[1]["keys"], ["cctv_exit_direction:2", "cctv_exit_direction:3"])

    def test_entrance_does_not_require_settings_query(self):
        db = Mock()
        self.assertEqual(repositories._attach_cctv_exit_directions(db, [{"id": 1, "section": "entrance"}]),
                         [{"id": 1, "section": "entrance", "exit_direction": None}])
        db.execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
