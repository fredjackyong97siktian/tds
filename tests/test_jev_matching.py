import base64
import json
import unittest
from io import BytesIO
from unittest.mock import Mock, patch

from PIL import Image

from app.services import jev_matching as jev


def trigger(number, time="2026-10-17T12:00:00"):
    return {"trigger_id": number, "trigger_time": time, "location_id": 2,
            "frames": [{"frame_index": i, "image_url": f"https://example.test/{number}/{i}.jpg"} for i in [3, 1, 2]]}


def answer(selected="2", confidence=0.98, probability=0.99):
    return {"answers": {
        "exit_choice": {"type": "choice", "choice": selected, "confidence": confidence,
                        "probabilities": {"2": 0.99 if selected == "2" else 0.01, "none": 0.01 if selected == "2" else 0.99}},
        "same_customer_2": {"type": "noul", "noul": probability},
    }}


class JevMatchingTests(unittest.TestCase):
    def test_request_sends_actual_images_in_chronological_sequences(self):
        loader = Mock(return_value="data:image/jpeg;base64,AA==")
        body, urls = jev.build_request(trigger(1), [trigger(2, "2026-10-17T12:02:00")], image_loader=loader)
        self.assertEqual(body["model"], "typesafe/jev-1.14")
        self.assertNotIn("messages", body)
        self.assertEqual([frame["frame_index"] for frame in body["state"]["entry"]["frames"]], [1, 2, 3])
        self.assertEqual(body["state"]["entry"]["frames"][0]["image"]["type"], "image_url")
        self.assertIn("none", body["questions"]["exit_choice"]["criteria"])
        self.assertEqual(len(urls), 6)
        self.assertEqual(loader.call_count, 6)

    def test_limits_frames_and_candidate_count(self):
        _, urls = jev.build_request(trigger(1), [trigger(2, "2026-10-17T12:02:00")], frames_per_trigger=1, image_loader=lambda url: "data:image/jpeg;base64,AA==")
        self.assertEqual(len(urls), 2)
        with self.assertRaises(ValueError):
            jev.build_request(trigger(1), [])
        with self.assertRaises(ValueError):
            jev.build_request(trigger(1), [trigger(i) for i in range(2, 5)])

    def test_same_trigger_or_non_later_exit_rejected_before_downloads(self):
        loader = Mock()
        for candidate in [trigger(1, "2026-10-17T12:01:00"), trigger(2), trigger(2, "2026-10-17T12:31:00")]:
            with self.assertRaises(ValueError):
                jev.build_request(trigger(1), [candidate], image_loader=loader)
        loader.assert_not_called()

    def test_cross_location_candidate_is_rejected(self):
        candidate = trigger(2, "2026-10-17T12:02:00")
        candidate["location_id"] = 3
        with self.assertRaises(ValueError):
            jev.build_request(trigger(1), [candidate])

    def test_matching_choice_and_pair_probability_form_preview_group(self):
        result = jev.interpret_response(answer(), entry_id=1, candidate_ids=[2])
        self.assertEqual(result["decision"], "match")
        self.assertEqual(result["groups"], [{"entry": [1], "exit": [2]}])
        self.assertTrue(result["test_only"])

    def test_no_matching_exit_is_normal_not_a_forced_group(self):
        result = jev.interpret_response(answer("none", probability=0.01), entry_id=1, candidate_ids=[2])
        self.assertEqual(result["decision"], "no_match")
        self.assertEqual(result["groups"], [])

    def test_conflicting_or_uncertain_answers_never_form_group(self):
        for response in [answer(probability=0.1), answer(confidence=0.4), answer("none", probability=0.95)]:
            result = jev.interpret_response(response, entry_id=1, candidate_ids=[2])
            self.assertEqual(result["decision"], "uncertain")
            self.assertEqual(result["groups"], [])

    def test_unknown_choices_and_invalid_probabilities_are_rejected(self):
        for selected, probability in [("999", 0.9), ("2", float("nan")), ("2", 1.1), ("2", True)]:
            with self.assertRaises(ValueError):
                jev.interpret_response(answer(selected, probability=probability), entry_id=1, candidate_ids=[2])

    def test_images_are_baseline_jpeg_and_fit_upstream_dimensions(self):
        for size in [(1920, 1080), (100, 80), (1500, 100)]:
            image = Image.new("RGB", size, "red")
            buffer = BytesIO()
            image.save(buffer, "PNG")
            encoded = jev.prepare_image(buffer.getvalue())
            raw = base64.b64decode(encoded.split(",", 1)[1])
            with Image.open(BytesIO(raw)) as prepared:
                self.assertEqual(prepared.format, "JPEG")
                self.assertGreaterEqual(min(prepared.size), 280)
                self.assertLessEqual(max(prepared.size), 1568)

    def test_openrouter_decisions_endpoint_and_billed_usage_are_preserved(self):
        response = {**answer(), "id": "test-response", "usage": {"input_tokens": 123, "output_tokens": 0, "cost": 0.00001}}
        opened = Mock()
        opened.__enter__ = Mock(return_value=Mock(read=Mock(return_value=json.dumps(response).encode())))
        opened.__exit__ = Mock(return_value=False)
        with patch.object(jev, "urlopen", return_value=opened) as call:
            parsed, meta = jev.call_openrouter({"model": jev.MODEL, "state": [], "questions": {}}, api_key="secret-test-only")
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, "https://openrouter.ai/api/alpha/decisions")
        self.assertEqual(parsed["answers"], response["answers"])
        self.assertEqual(meta["usage"]["actual_cost_usd"], 0.00001)
        self.assertEqual(meta["raw_response"], response)
        self.assertNotIn("secret-test-only", json.dumps(meta))

    def test_missing_openrouter_key_does_not_make_network_call(self):
        with patch.object(jev, "urlopen") as call, self.assertRaises(ValueError):
            jev.call_openrouter({}, api_key="")
        call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
