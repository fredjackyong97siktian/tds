import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from app.services import grouping_estimate as estimate
from app.services import workflow_service as workflow


def trigger(number, minutes=0, identity=False, frame_count=5):
    return {
        "trigger_id": number,
        "trigger_time": datetime(2026, 10, 17, 12) + timedelta(minutes=minutes),
        "phone_entry_id": number if identity else None,
        "credit_card_entry_id": None,
        "frames": [{"image_url": f"https://example.test/{number}/{i}.jpg"} for i in range(frame_count)],
    }


class GroupingEstimateTests(unittest.TestCase):
    def test_adjacent_uses_actual_candidate_rules_and_limit(self):
        items = [trigger(1, identity=True), trigger(2, 1, identity=True), trigger(3, 2),
                 trigger(4, 3), trigger(5, 4), trigger(6, 31)]
        plan = workflow._plan_grouping_adjacent_checks(items[::-1])
        self.assertEqual([item["entry_id"] for item in plan], [1, 2])
        self.assertEqual([item["trigger_id"] for item in plan[0]["candidates"]], [3, 4])

    def test_no_calls_for_entries_without_later_framed_candidates(self):
        items = [trigger(1, 1), trigger(2, 2, identity=True), trigger(3, 3, frame_count=0), trigger(4, 33)]
        self.assertEqual(workflow._plan_grouping_adjacent_checks(items), [])

    def test_adjacent_includes_thirty_minute_boundary_but_not_same_time(self):
        items = [trigger(1, identity=True), trigger(2), trigger(3, 30), trigger(4, 31)]
        plan = workflow._plan_grouping_adjacent_checks(items)
        self.assertEqual([item["trigger_id"] for item in plan[0]["candidates"]], [3])

    def test_chunk_packing_keeps_whole_triggers(self):
        self.assertEqual(estimate._chunk_count([trigger(i) for i in range(3)], 9), 3)
        self.assertEqual(estimate._chunk_count([trigger(i) for i in range(3)], 10), 2)
        self.assertEqual(estimate._chunk_count([trigger(1, frame_count=0)], 10), 0)
        self.assertEqual(estimate._chunk_count([trigger(1, frame_count=12)], 10), 1)

    def test_timing_uses_median_of_per_unit_durations(self):
        samples = [{"seconds": 60, "entry_count": 2}, {"seconds": 100, "entry_count": 2},
                   {"seconds": 30, "entry_count": 0}, {"seconds": "bad", "entry_count": 2}]
        with patch.object(estimate.repositories, "list_grouping_timing_samples", return_value=samples), patch.object(workflow, "_grouping_stage_label", return_value="test_adjacent"):
            timing = estimate._timing(Mock(), "adjacent")
        self.assertEqual(timing, {"sample_runs": 2, "seconds_per_unit": 40})

    def test_counts_are_separate_for_each_location_batch(self):
        with patch.object(estimate, "_timing", return_value={"sample_runs": 3, "seconds_per_unit": 20}), patch.object(workflow, "_grouping_gemini_max_images_per_request", return_value=40), patch.object(workflow.settings, "grouping_direct_enabled", True):
            result = estimate.estimate_inputs(Mock(), [[trigger(1, identity=True), trigger(2, 1)], [trigger(3, identity=True), trigger(4, 1)]])
        self.assertEqual(result["adjacent_checks"], 2)
        self.assertEqual(result["direct_chunks_estimate"], 2)
        self.assertEqual(result["image_count"], 20)
        self.assertEqual(result["timing_basis"], "recent_runs")

    def test_missing_history_is_explicit_and_direct_can_be_disabled(self):
        with patch.object(estimate, "_timing", return_value={"sample_runs": 0, "seconds_per_unit": None}), patch.object(workflow.settings, "grouping_direct_enabled", False):
            result = estimate.estimate_inputs(Mock(), [[trigger(1, identity=True), trigger(2, 1)]])
        self.assertEqual(result["direct_chunks_estimate"], 0)
        self.assertEqual(result["timing_basis"], "rough_fallback")
        self.assertGreater(result["estimated_seconds_max"], result["estimated_seconds_min"])

    def test_retry_preview_reads_without_modifying_batch_or_calling_ai(self):
        item = trigger(1, identity=True)
        item["frame_payload"] = {"frames": item.pop("frames")}
        db = Mock()
        with patch.object(estimate.repositories, "get_grouping_batch", return_value={}), patch.object(estimate.repositories, "list_grouping_estimate_triggers", return_value=[item]), patch.object(estimate, "estimate_inputs", return_value={"trigger_count": 1, "note": ""}) as calculate:
            result = estimate.preview_grouping(db, mode="retry", batch_id=7)
        self.assertEqual(result["trigger_count"], 1)
        self.assertTrue(calculate.call_args.args[1][0][0]["frames"])
        db.commit.assert_not_called()
        db.execute.assert_not_called()

    def test_manual_preview_uses_hour_before_newest_trigger_not_now(self):
        rows = []
        for number, minute in [(1, 0), (2, 60), (3, 61)]:
            row = trigger(number, minute)
            row["trigger_frames"] = row.pop("frames")
            rows.append(row)
        with patch.object(estimate.repositories, "list_locations", return_value=[{"id": 2}]), patch.object(estimate.repositories, "list_manual_grouping_ready_trigger_frame_assets", return_value=rows), patch.object(workflow, "_check_motion_sensor_person_presence") as person_check, patch.object(estimate, "estimate_inputs", return_value={"note": ""}) as calculate:
            estimate.preview_grouping(Mock(), mode="now")
        person_check.assert_not_called()
        self.assertEqual([item["trigger_id"] for item in calculate.call_args.args[1][0]], [2, 3])

    def test_invalid_range_is_rejected_before_queries(self):
        db = Mock()
        with self.assertRaises(ValueError):
            estimate.preview_grouping(db, mode="range", start_time="2026-10-17T12:00", end_time="2026-10-17T11:00")
        db.execute.assert_not_called()

    def test_preview_never_calls_motion_person_ai(self):
        rows = []
        for number, cached in [(1, None), (2, 0), (3, 1)]:
            row = trigger(number)
            row.update(trigger_source="motion_sensor", motion_person_present=cached)
            row["trigger_frames"] = row.pop("frames")
            rows.append(row)
        with patch.object(estimate.repositories, "list_locations", return_value=[{"id": 2}]), patch.object(estimate.repositories, "list_manual_grouping_ready_trigger_frame_assets", return_value=rows), patch.object(workflow, "_check_motion_sensor_person_presence") as person_check, patch.object(estimate, "estimate_inputs", return_value={"note": ""}) as calculate:
            estimate.preview_grouping(Mock(), mode="now")
        person_check.assert_not_called()
        self.assertEqual([item["trigger_id"] for item in calculate.call_args.args[1][0]], [1, 3])

    def test_retry_preview_refreshes_frames_and_includes_new_window_triggers(self):
        rows = [trigger(1, identity=True), trigger(2, 1)]
        for row in rows:
            row["trigger_frames"] = row.pop("frames")
        batch = {"location_id": 2, "window_start": rows[0]["trigger_time"], "window_end": rows[1]["trigger_time"] + timedelta(seconds=1)}
        with patch.object(estimate.repositories, "get_grouping_batch", return_value=batch), patch.object(estimate.repositories, "list_grouping_estimate_triggers", return_value=[{**rows[0], "frame_payload": {"frames": []}}]), patch.object(estimate.repositories, "list_trigger_frame_assets_for_window", return_value=rows), patch.object(estimate, "estimate_inputs", return_value={"note": ""}) as calculate:
            estimate.preview_grouping(Mock(), mode="retry", batch_id=7)
        self.assertEqual([item["trigger_id"] for item in calculate.call_args.args[1][0]], [1, 2])
        self.assertTrue(all(item["frames"] for item in calculate.call_args.args[1][0]))


if __name__ == "__main__":
    unittest.main()
