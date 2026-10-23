import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from app.services import detection_frames, media_retention


class DetectionFrameTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        patcher = patch.object(detection_frames.settings, "video_storage_dir", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        output = BytesIO()
        Image.new("RGB", (32, 24), "blue").save(output, "JPEG", quality=85)
        self.raw = output.getvalue()
        self.frame = {"role": "kiosk", "image_url": "data:image/jpeg;base64," + base64.b64encode(self.raw).decode(),
                      "video_asset_id": 99, "playback_offset_seconds": 12.5, "camera_clock": "uncalibrated"}

    def preserve(self, run_id=11):
        return detection_frames.preserve([self.frame], location_id=2, session_id=7, run_id=run_id)

    def bind_run(self, manifest, version=1):
        db = Mock()
        db.execute.return_value.mappings.return_value.first.return_value = {
            "id": 11, "script_name": "detection_analysis", "model_name": "model", "runner_payload": json.dumps({"frame_snapshot_version": version, "frames": manifest})}
        patcher = patch.object(detection_frames.repositories, "get_session", return_value={"id": 7, "location_id": 2,
            "result_summary": {"detection_analysis": {"script_run_id": 11}}})
        patcher.start()
        self.addCleanup(patcher.stop)
        return db

    def test_exact_bytes_survive_and_manifest_contains_no_base64_or_private_paths(self):
        frames = self.preserve()
        path = detection_frames._directory(2, 7, 11) / "frame_01.jpg"
        self.assertEqual(path.read_bytes(), self.raw)
        self.assertEqual(frames[0]["sha256"], hashlib.sha256(self.raw).hexdigest())
        self.assertEqual(frames[0]["playback_offset_seconds"], 12.5)
        self.assertNotIn("base64", json.dumps(frames))
        self.assertNotIn(str(self.root), json.dumps(frames))
        self.assertEqual(frames[0]["image_url"], "/api/v1/sessions/7/detection-analysis/frames/11/1")

    def test_attempts_never_overwrite_previous_images(self):
        self.preserve()
        self.preserve(run_id=12)
        with self.assertRaises(FileExistsError):
            self.preserve()
        self.assertTrue((detection_frames._directory(2, 7, 11) / "frame_01.jpg").exists())
        self.assertTrue((detection_frames._directory(2, 7, 12) / "frame_01.jpg").exists())

    def test_viewing_frames_reads_saved_images_without_resampling_or_model_call(self):
        manifest = self.preserve()
        db = self.bind_run(manifest)
        result = detection_frames.list_frames(db, 7, 11)
        self.assertEqual(len(result["frames"]), 1)
        self.assertTrue(result["frames"][0]["available"])
        self.assertEqual(detection_frames.image_path(db, 7, 11, 1).read_bytes(), self.raw)
        self.assertEqual(db.execute.call_args.args[1], {"id": 11, "session_id": 7})

    def test_historical_frames_are_not_invented(self):
        db = self.bind_run([{"role": "kiosk", "playback_offset_seconds": 2}], version=None)
        result = detection_frames.list_frames(db, 7, 11)
        self.assertEqual(result["frames"], [])
        self.assertIn("not retained", result["message"])
        with self.assertRaises(FileNotFoundError):
            detection_frames.image_path(db, 7, 11, 1)

    def test_cross_session_run_is_rejected(self):
        db = self.bind_run([])
        db.execute.return_value.mappings.return_value.first.return_value = None
        with self.assertRaises(ValueError):
            detection_frames.list_frames(db, 8, 11)
        self.assertEqual(db.execute.call_args.args[1]["session_id"], 8)

    def test_expired_or_modified_frames_are_never_replaced_with_guesses(self):
        manifest = self.preserve()
        db = self.bind_run(manifest)
        path = detection_frames._directory(2, 7, 11) / "frame_01.jpg"
        path.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "original request"):
            detection_frames.image_path(db, 7, 11, 1)
        path.unlink()
        self.assertFalse(detection_frames.list_frames(db, 7, 11)["frames"][0]["available"])
        with self.assertRaises(FileNotFoundError):
            detection_frames.image_path(db, 7, 11, 1)

    def test_invalid_or_path_escape_input_is_rejected(self):
        for frame in ({**self.frame, "image_url": "https://example.test/image.jpg"},
                      {**self.frame, "image_url": "data:image/jpeg;base64,invalid"},
                      {**self.frame, "role": "../unknown"}):
            with self.assertRaises(ValueError):
                detection_frames.preserve([frame], location_id=2, session_id=7, run_id=11)

    def test_known_theft_session_path_is_protected_by_existing_cleaner_policy(self):
        path = detection_frames._directory(2, 7, 11) / "frame_01.jpg"
        self.assertTrue(media_retention._owner_protected(path, {7}, set()))
        self.assertFalse(media_retention._owner_protected(path, {8}, set()))


if __name__ == "__main__":
    unittest.main()
