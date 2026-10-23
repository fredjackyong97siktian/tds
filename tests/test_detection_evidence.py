from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

from PIL import Image

from app.services import detection_evidence as evidence


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        storage = patch.object(evidence.settings, "video_storage_dir", self.root)
        storage.start()
        self.addCleanup(storage.stop)
        self.image = self.root / "reference.jpg"
        Image.new("RGB", (32, 32), "green").save(self.image)
        self.video = self.root / "original.mp4"
        self.video.write_bytes(b"synthetic fixture")
        self.payload = {"entry_trigger_ids": [10], "exit_trigger_ids": [20, 21],
                        "media_retrieval": {"links": [{"video_asset_id": 1, "role": "kiosk",
                                                       "start": "2026-01-01T10:00:00", "end": "2026-01-01T10:30:00"}]}}
        self.asset = {"id": 1, "section": "kiosk", "file_path": str(self.video),
                      "video_url": "https://not-used.invalid/processed/annotated.mp4",
                      "captured_start_time": "2026-01-01T10:00:00",
                      "captured_end_time": "2026-01-01T10:30:00"}
        self.frames_patch = patch.object(evidence.repositories, "list_trigger_frame_assets", side_effect=self.references)
        self.frames_mock = self.frames_patch.start()
        self.addCleanup(self.frames_patch.stop)
        self.asset_patch = patch.object(evidence.repositories, "get_video_asset", return_value=self.asset)
        self.asset_mock = self.asset_patch.start()
        self.addCleanup(self.asset_patch.stop)
        self.duration_patch = patch.object(evidence, "_duration", return_value=100.0)
        self.duration_mock = self.duration_patch.start()
        self.addCleanup(self.duration_patch.stop)
        self.decode_patch = patch.object(evidence, "_decode", return_value="data:image/jpeg;base64,fixture")
        self.decode_mock = self.decode_patch.start()
        self.addCleanup(self.decode_patch.stop)

    def references(self, db, *, trigger_id, limit):
        return [{"phone_number": "DO_NOT_SEND", "frames": [
            {"frame_index": i, "sample_time": f"2026-01-01T10:00:{i:02}",
             "image_url": str(self.image), "status": "ok"} for i in range(12)]}]

    def build(self):
        return evidence.build_evidence(None, {"id": 9, "summary": "OLD_AI_OUTPUT"},
                                       {"factor_payload": self.payload}, self.root / "work")

    def test_budget_balanced_references_and_uniform_chronological_kiosk(self):
        context, frames = self.build()
        self.assertEqual(len(frames), 40)
        self.assertEqual(sum(f["role"] == "entry" for f in frames), 6)
        self.assertEqual(sum(f["role"] == "exit" for f in frames), 6)
        kiosk = [f for f in frames if f["role"] == "kiosk"]
        self.assertEqual(len(kiosk), 28)
        offsets = [f["playback_offset_seconds"] for f in kiosk]
        self.assertEqual(offsets, sorted(offsets))
        self.assertAlmostEqual(offsets[0], 100 / 56)
        self.assertAlmostEqual(offsets[-1], 100 - 100 / 56)
        self.assertEqual(context["reference_coverage"]["exit"], {"20": 3, "21": 3})
        self.assertEqual(context["kiosk_videos"][0]["playback_duration_seconds"], 100)
        self.assertIn("uncalibrated", context["timing"])
        self.assertNotIn("video_url", json.dumps(context))
        self.assertEqual([f["role"] for f in frames], ["entry"] * 6 + ["kiosk"] * 28 + ["exit"] * 6)

    def test_sparse_references_leave_remaining_slots_to_kiosk(self):
        self.frames_mock.side_effect = lambda *args, **kwargs: [{"frames": [{"image_url": str(self.image)}]}]
        _, frames = self.build()
        self.assertEqual(sum(f["role"] == "kiosk" for f in frames), 37)

    def test_all_exits_even_above_six(self):
        self.payload["exit_trigger_ids"] = list(range(20, 30))
        context, frames = self.build()
        self.assertEqual(len(context["reference_coverage"]["exit"]), 10)
        self.assertTrue(all(context["reference_coverage"]["exit"].values()))
        self.assertLessEqual(len(frames), 40)

    def test_budget_impossible_fails_before_repository_reads(self):
        self.payload["exit_trigger_ids"] = list(range(20, 60))
        with self.assertRaisesRegex(ValueError, "budget"):
            self.build()
        self.frames_mock.assert_not_called()

    def test_missing_exit_fails_instead_of_silently_omitting(self):
        self.frames_mock.side_effect = lambda *args, **kwargs: [] if kwargs["trigger_id"] == 21 else self.references(*args, **kwargs)
        with self.assertRaisesRegex(ValueError, "Missing exit reference for trigger 21"):
            self.build()

    def test_receipts_paid_only_normalization_privacy_and_no_mutation(self):
        self.payload.update(transactions=[{"status": "paid", "total_amount": "207", "total_items": "2",
                                          "phone_number": "SECRET_PHONE", "raw_payload": {"card": "SECRET_CARD"},
                                          "details": [{"item_name": "Bottle", "quantity": "2", "price": "100",
                                                       "subtotal": "200", "card": "SECRET_CARD"}]},
                                         {"status": "failed", "total_amount": 500}, {"total_amount": 900}],
                            issue_transactions=[{"status": "pending", "total_amount": 200}],
                            factor_hits=["EXPECTED_LABEL"], reason="EXPECTED_LABEL", previous_ai_status="EXPECTED_LABEL")
        original = deepcopy(self.payload)
        context, frames = self.build()
        self.assertEqual(len(context["transactions"]), 1)
        self.assertEqual(context["transactions"][0]["total_amount"], 2.07)
        self.assertEqual(context["transactions"][0]["total_items"], 2)
        self.assertEqual(context["transactions"][0]["details"], [{"item_name": "Bottle", "quantity": 2, "price": 1.0, "subtotal": 2.0}])
        self.assertEqual([r.get("status") for r in context["issue_transactions"]], ["pending", "failed", None])
        serialized = json.dumps((context, frames))
        for private in ("SECRET_PHONE", "SECRET_CARD", "DO_NOT_SEND", "OLD_AI_OUTPUT", "EXPECTED_LABEL"):
            self.assertNotIn(private, serialized)
        self.assertEqual(original, self.payload)

    def test_assignment_uncertainty_is_not_nonpayment(self):
        for status, count in (("needs_review", 0), ("unresolved", 2), ("complete", 1)):
            with self.subTest(status=status):
                self.payload["transaction_assignment"] = {"status": status, "unresolved_count": count,
                                                          "reason": "DO_NOT_SEND", "phone_number": "DO_NOT_SEND"}
                context, _ = self.build()
                self.assertEqual(context["transaction_assignment"], {"status": status, "unresolved_count": count})
                self.assertTrue(any("do not establish absence of payment" in item for item in context["limitations"]))

    def test_source_metadata_rescues_processed_file_path(self):
        self.asset.update(file_path="spaces://bucket/processed/result.mp4",
                          metadata=json.dumps({"source_file_path": str(self.video)}))
        context, frames = self.build()
        self.assertTrue(context["kiosk_videos"])
        self.assertTrue(any(f["role"] == "kiosk" for f in frames))

    def test_processed_without_source_never_falls_back_to_video_url(self):
        self.asset["file_path"] = "spaces://bucket/processed/result.mp4"
        with self.assertRaisesRegex(ValueError, "No kiosk visual evidence"):
            self.build()
        self.duration_mock.assert_not_called()

    def test_source_object_key_metadata(self):
        self.asset.update(file_path="spaces://bucket/processed/result.mp4", metadata={"source_object_key": "bucket/source/raw.mp4"})
        self.assertEqual(evidence._source_path(self.asset), "spaces://bucket/source/raw.mp4")

    def test_multiple_clips_chronological_measured_offsets(self):
        self.payload["media_retrieval"]["links"] = [
            {"video_asset_id": 2, "section": "kiosk", "start": "2026-01-01T11:00:00"},
            {"video_asset_id": 1, "section": "kiosk", "start": "2026-01-01T10:00:00"}]
        self.asset_mock.side_effect = lambda db, asset_id: {**self.asset, "id": asset_id}
        _, frames = self.build()
        ids = [f["video_asset_id"] for f in frames if f["role"] == "kiosk"]
        self.assertEqual(ids, [1] * 14 + [2] * 14)

    def test_no_filename_duration_inference_and_decode_failure_is_limitation(self):
        self.duration_mock.side_effect = ValueError("no duration")
        with self.assertRaisesRegex(ValueError, "No kiosk visual evidence"):
            self.build()
        self.decode_mock.assert_not_called()

    def test_partial_decode_failure_has_explicit_counts_and_limitations(self):
        self.decode_mock.side_effect = [ValueError("missing frame")] + ["data:image/jpeg;base64,fixture"] * 27
        context, frames = self.build()
        self.assertEqual(len(frames), 39)
        self.assertEqual(context["frame_counts"]["kiosk_requested"], 28)
        self.assertEqual(context["frame_counts"]["kiosk_supplied"], 27)
        self.assertIn("Only 27 of 28 requested kiosk frames decoded", context["limitations"])

    def test_local_paths_and_symlinks_cannot_escape_storage(self):
        allowed = self.root / "storage"
        allowed.mkdir()
        (allowed / "escape.jpg").symlink_to(self.image)
        with patch.object(evidence.settings, "video_storage_dir", allowed):
            media = evidence._Materializer(self.root / "work")
            for source in (self.image, allowed / "escape.jpg"):
                with self.assertRaisesRegex(ValueError, "outside permitted"):
                    media.get(str(source), 10000)

    def test_local_materialization_size_and_deadline_bounds(self):
        media = evidence._Materializer(self.root)
        with self.assertRaisesRegex(ValueError, "size"):
            media.get(str(self.video), 2)
        media.deadline = 0
        with self.assertRaises(TimeoutError):
            media.get(str(self.video), 100)

    def test_unknown_remote_and_nvr_urls_never_requested(self):
        media = evidence._Materializer(self.root)
        with patch.object(evidence, "build_opener") as opener:
            for value in ("http://192.168.1.2/video", "rtsp://nvr/video", "https://unconfigured.invalid/video"):
                with self.assertRaises(ValueError):
                    media.get(value, 100)
            opener.assert_not_called()

    def test_bounded_download_and_no_redirect(self):
        media = evidence._Materializer(self.root)
        incoming = BytesIO(b"0123456789")
        incoming.headers = {"Content-Length": "10"}
        with patch.object(evidence, "_storage_url", return_value="https://storage.invalid/raw"), \
             patch.object(evidence, "build_opener") as opener:
            opener.return_value.open.return_value = incoming
            with self.assertRaisesRegex(ValueError, "size"):
                media.get("spaces://bucket/source/raw", 5)
        with self.assertRaises(ValueError):
            evidence._NoRedirect().redirect_request(None, None, 302, "", {}, "https://nvr.invalid")

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe required")
    def test_real_local_video_probe_decode_no_network(self):
        self.duration_patch.stop()
        self.decode_patch.stop()
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=64x48:d=2:r=5",
                        "-c:v", "mpeg4", "-y", str(self.video)], check=True, timeout=10)
        with patch.object(evidence, "build_opener") as opener:
            context, frames = self.build()
        opener.assert_not_called()
        self.assertAlmostEqual(context["kiosk_videos"][0]["playback_duration_seconds"], 2.0)
        self.assertGreater(len(frames), 12)
        self.assertLessEqual(len(frames), 40)
        self.assertEqual(context["frame_counts"]["supplied"], len(frames))
        if len(frames) < 40:
            self.assertTrue(any("requested kiosk frames decoded" in item for item in context["limitations"]))
        self.assertTrue(all(f["image_url"].startswith("data:image/jpeg;base64,") for f in frames))


def stream(*chunks, done=True):
    data = ": OPENROUTER PROCESSING\n\n" + "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
    return BytesIO((data + ("data: [DONE]\n\n" if done else "")).encode())


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.key = patch.object(evidence.settings, "openrouter_api_key", "fake-unit-test-key")
        self.key.start()
        self.addCleanup(self.key.stop)
        self.frames = [{"role": "kiosk", "image_url": "data:image/jpeg;base64,fixture"}]
        self.chunks = [
            {"id": "gen-test", "model": evidence.MODEL, "choices": [{"index": 0, "delta": {"role": "assistant", "content": '{"status":'}, "finish_reason": None}]},
            {"id": "gen-test", "choices": [{"index": 0, "delta": {"content": '"insufficient_evidence"}'}, "finish_reason": "stop"}]},
            {"id": "gen-test", "choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001}}]

    def test_one_request_raw_chunks_usage_and_finish_reason(self):
        captured = []
        with patch.object(evidence, "urlopen", return_value=stream(*self.chunks)) as send:
            raw = evidence.call_mimo({"session_id": 1}, self.frames, on_chunk=captured.append)
        self.assertEqual(captured, self.chunks)
        self.assertEqual(raw["usage"], self.chunks[-1]["usage"])
        self.assertEqual(raw["choices"][0]["finish_reason"], "stop")
        self.assertEqual(raw["choices"][0]["message"]["content"], '{"status":"insufficient_evidence"}')
        send.assert_called_once()
        body = json.loads(send.call_args.args[0].data)
        self.assertEqual(body["model"], "xiaomi/mimo-v2.6-flash")
        self.assertTrue(body["stream"])
        self.assertTrue(body["stream_options"]["include_usage"])
        self.assertIn("Missing or unclear evidence", body["messages"][0]["content"])
        self.assertNotIn("fake-unit-test-key", json.dumps(body))

    def test_length_finish_reason_preserved_without_parsing_partial_json(self):
        self.chunks[1]["choices"][0]["finish_reason"] = "length"
        with patch.object(evidence, "urlopen", return_value=stream(*self.chunks)):
            self.assertEqual(evidence.call_mimo({}, self.frames)["choices"][0]["finish_reason"], "length")

    def test_selected_model_is_sent_instead_of_default(self):
        with patch.object(evidence, "urlopen", return_value=stream(*self.chunks)) as send:
            evidence.call_mimo({}, self.frames, model_name="vendor/custom-vision")
        self.assertEqual(json.loads(send.call_args.args[0].data)["model"], "vendor/custom-vision")

    def test_callback_exception_propagates_unchanged_no_retry(self):
        for error in (RuntimeError("persist failed"), OSError("persist failed"), ValueError("persist failed")):
            with self.subTest(error=type(error)), \
                 patch.object(evidence, "urlopen", return_value=stream(*self.chunks)) as send:
                def fail(chunk):
                    raise error
                with self.assertRaises(type(error)) as raised:
                    evidence.call_mimo({}, self.frames, on_chunk=fail)
                self.assertIs(raised.exception, error)
                send.assert_called_once()

    def test_usage_and_error_are_delivered_before_stream_failure(self):
        chunks = self.chunks + [{"error": {"message": "provider failed"}}]
        captured = []
        with patch.object(evidence, "urlopen", return_value=stream(*chunks)) as send:
            with self.assertRaisesRegex(RuntimeError, "stream error"):
                evidence.call_mimo({}, self.frames, on_chunk=captured.append)
        self.assertEqual(captured, chunks)
        send.assert_called_once()

    def test_disconnect_does_not_reuse_partial_completion(self):
        captured = []
        with patch.object(evidence, "urlopen", return_value=stream(*self.chunks, done=False)) as send:
            with self.assertRaisesRegex(RuntimeError, "before DONE"):
                evidence.call_mimo({}, self.frames, on_chunk=captured.append)
        self.assertEqual(captured[-1]["usage"], self.chunks[-1]["usage"])
        send.assert_called_once()

    def test_transport_error_sanitized_and_no_retry(self):
        with patch.object(evidence, "urlopen", side_effect=URLError("fake-unit-test-key")) as send:
            with self.assertRaisesRegex(RuntimeError, "transport failed") as raised:
                evidence.call_mimo({}, self.frames)
        self.assertNotIn("fake-unit-test-key", str(raised.exception))
        send.assert_called_once()

    def test_key_missing_no_runtime_file_fallback(self):
        with patch.object(evidence.settings, "openrouter_api_key", None), \
             patch.dict(evidence.os.environ, {}, clear=True), patch.object(evidence, "urlopen") as send:
            with self.assertRaisesRegex(RuntimeError, "not configured"):
                evidence.call_mimo({}, self.frames)
        send.assert_not_called()

    def test_env_key_and_local_frames_only(self):
        with patch.object(evidence.settings, "openrouter_api_key", None), \
             patch.dict(evidence.os.environ, {"OPENROUTER_API_KEY": "env-test"}), \
             patch.object(evidence, "urlopen", return_value=stream(*self.chunks)) as send:
            evidence.call_mimo({}, self.frames)
        self.assertEqual(send.call_args.args[0].get_header("Authorization"), "Bearer env-test")
        with patch.object(evidence, "urlopen") as send:
            with self.assertRaises(ValueError):
                evidence.call_mimo({}, [{"image_url": "https://nvr.invalid/image"}])
            with self.assertRaises(ValueError):
                evidence.call_mimo({}, [{}] * 41)
            send.assert_not_called()

    def test_door_only_request_rejected_before_paid_request(self):
        with patch.object(evidence, "urlopen") as send:
            with self.assertRaisesRegex(ValueError, "No kiosk visual evidence"):
                evidence.call_mimo({}, [{"role": "exit", "image_url": "data:image/jpeg;base64,fixture"}])
        send.assert_not_called()

    def test_usage_chunk_null_finish_reason_does_not_erase_stop(self):
        self.chunks[-1]["choices"] = [{"index": 0, "delta": {}, "finish_reason": None}]
        with patch.object(evidence, "urlopen", return_value=stream(*self.chunks)):
            self.assertEqual(evidence.call_mimo({}, self.frames)["choices"][0]["finish_reason"], "stop")


if __name__ == "__main__":
    unittest.main()
