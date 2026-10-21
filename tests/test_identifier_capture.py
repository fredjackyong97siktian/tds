from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
import subprocess
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from app.services import identifier_capture as capture
from app.services import workflow_service as workflow


class IdentifierCaptureTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
    def test_real_ffmpeg_produces_twelve_frames_from_a_local_fixture(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            subprocess.run([shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                            "testsrc=duration=8:size=96x64:rate=10", "-c:v", "mpeg4", str(source)],
                           capture_output=True, check=True, timeout=15)
            with patch.object(workflow.settings, "ffmpeg_bin", shutil.which("ffmpeg")):
                command = workflow._build_frame_batch_capture_command(
                    str(source), start_offset_seconds=0.5, gap_seconds=0.5, frame_count=12,
                    output_pattern=Path(directory) / "frame_%02d.jpg", apply_entrance_crop=False)
            transport = command.index("-rtsp_transport")
            del command[transport:transport + 2]  # Only transport differs for a local fixture.
            subprocess.run(command, capture_output=True, check=True, timeout=15)
            frames = sorted(Path(directory).glob("frame_*.jpg"))
            self.assertEqual(len(frames), 12)
            self.assertTrue(all(path.stat().st_size > 0 for path in frames))

    def test_twelve_half_second_samples_end_three_seconds_before_exit(self):
        times = capture.sample_times(datetime(2026, 10, 21, 11))
        self.assertEqual(len(times), 12)
        self.assertEqual(times[0], datetime(2026, 10, 21, 10, 59, 51, 500000))
        self.assertEqual(times[-1], datetime(2026, 10, 21, 10, 59, 57))
        self.assertTrue(all((second - first).total_seconds() == 0.5 for first, second in zip(times, times[1:])))

    def test_identifier_batch_disables_entrance_crop(self):
        with patch.object(workflow, "_trigger_frame_crop_filter", return_value="crop=iw/2:ih:0:0") as crop:
            command = workflow._build_frame_batch_capture_command(
                "rtsp://example.test", start_offset_seconds=0.5, gap_seconds=0.5,
                frame_count=12, output_pattern=Path("frame_%02d.jpg"), apply_entrance_crop=False)
        crop.assert_not_called()
        self.assertEqual(command[command.index("-vf") + 1], "fps=2.000000")
        self.assertEqual(command[command.index("-ss") + 1], "0.500")
        self.assertEqual(command[command.index("-frames:v") + 1], "12")

    def run_capture(self, *, frames=12, timeout=False, direction="left"):
        self.calls = {}
        with TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.object(capture, "tmp_trigger_root", return_value=Path(directory)))
            mocks = {
                "create_script_run_started": 100,
                "get_cctv_by_location_section": {"id": 7, "recorder_channel": "3", "exit_direction": direction},
                "get_location_endpoint": {"dahua_host": "example.test", "dahua_username": "camera-user",
                                          "dahua_password_encrypted": "encrypted"},
                "merge_script_run_stdout_fields": "{}", "finish_script_run": None,
            }
            for name, value in mocks.items():
                self.calls[name] = stack.enter_context(patch.object(workflow.repositories, name, return_value=value))
            stack.enter_context(patch.object(workflow, "is_spaces_configured", return_value=True))
            stack.enter_context(patch.object(workflow, "decrypt_secret", return_value="camera-secret"))
            self.calls["delay"] = stack.enter_context(patch.object(workflow, "_current_delayed_seconds", return_value=425))
            self.calls["url"] = stack.enter_context(patch.object(workflow, "_build_dahua_rtsp_playback_url", return_value="rtsp://camera-user:camera-secret@example.test"))
            self.calls["lock"] = stack.enter_context(patch.object(workflow, "_acquire_retrieval_ffmpeg_lock", return_value=Mock()))
            self.calls["release"] = stack.enter_context(patch.object(workflow, "_release_retrieval_ffmpeg_lock"))
            self.calls["upload"] = stack.enter_context(patch.object(workflow, "upload_private_file", side_effect=lambda path, key, **kw: {"public_url": f"https://example.test/{key}"}))

            def fake_run(command, **kwargs):
                if timeout:
                    raise subprocess.TimeoutExpired(command, 1)
                for index in range(1, frames + 1):
                    Path(command[-1] % index).write_bytes(b"test-jpeg")
                return Mock(returncode=0, stderr="")

            self.calls["run"] = stack.enter_context(patch.object(capture.subprocess, "run", side_effect=fake_run))
            return capture.capture_exit(Mock(), location_id=2, exit_trigger={"trigger_id": 58,
                                        "trigger_time": datetime(2026, 10, 21, 11)}, parent_script_run_id=99)

    def test_capture_uses_identifier_channel_and_offset_and_logs_all_frames(self):
        result = self.run_capture()
        self.assertEqual(len(result["frames"]), 12)
        self.calls["get_cctv_by_location_section"].assert_called_once()
        self.assertEqual(self.calls["get_cctv_by_location_section"].call_args.kwargs["section"], "identifier")
        url = self.calls["url"].call_args.kwargs
        self.assertEqual(url["channel"], "3")
        self.assertEqual(url["start_time"], datetime(2026, 10, 21, 10, 52, 46))
        self.assertEqual(url["end_time"], datetime(2026, 10, 21, 10, 52, 53))
        self.assertEqual(result["frames"][0]["camera_sample_time"], "2026-10-21T10:52:46.500000")
        self.assertEqual(self.calls["run"].call_count, 1)
        self.assertEqual(self.calls["upload"].call_count, 12)
        self.calls["release"].assert_called_once()
        self.assertEqual(self.calls["finish_script_run"].call_args.kwargs["status"], "success")
        self.assertEqual(self.calls["create_script_run_started"].call_args.kwargs["runner_payload"]["parent_script_run_id"], 99)

    def test_partial_capture_is_not_padded_or_treated_as_complete(self):
        with self.assertRaisesRegex(RuntimeError, "5/12 frames"):
            self.run_capture(frames=5)
        self.calls["upload"].assert_not_called()
        self.calls["release"].assert_called_once()
        self.assertEqual(self.calls["finish_script_run"].call_args.kwargs["status"], "failed")

    def test_timeout_releases_lock_and_redacts_camera_credentials(self):
        with self.assertRaises(RuntimeError) as error:
            self.run_capture(timeout=True)
        self.assertNotIn("camera-secret", str(error.exception))
        self.assertNotIn("camera-secret", self.calls["finish_script_run"].call_args.kwargs["stderr_log"])
        self.calls["release"].assert_called_once()

    def test_missing_direction_is_a_logged_configuration_error(self):
        with self.assertRaisesRegex(RuntimeError, "Exit Direction"):
            self.run_capture(direction=None)
        self.calls["run"].assert_not_called()
        self.assertEqual(self.calls["finish_script_run"].call_args.kwargs["status"], "failed")


if __name__ == "__main__":
    unittest.main()
