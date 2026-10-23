from contextlib import ExitStack, nullcontext
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from app.services import detection_analysis as detection
from app.services import confidence_media, session_analysis, workflow_service


def answer(status="reasonable", cost=0.005):
    return {"model": detection.MODEL, "content": json.dumps({"status": status, "reason": "Visible evidence explanation.", "limitations": "Hidden goods remain unobserved."}),
            "usage": {"cost": cost}, "finish_reason": "stop"}


class DetectionAnalysisTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(detection, "current_model", return_value=detection.MODEL)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_decisions_are_not_confidence_gated(self):
        for value, session_status in (("reasonable", "not_detected"), ("suspicious", "detected"), ("insufficient_evidence", "need_review")):
            with self.subTest(value=value), patch.object(detection, "save") as save:
                result = detection._apply(Mock(), 7, {"script_run_id": 9}, {**answer(value), "confidence": 0.01})
                self.assertEqual(result["decision"], value)
                self.assertEqual(save.call_args.kwargs["session_status"], session_status)

    def test_invalid_and_truncated_responses_cannot_become_all_clear(self):
        for response in ({"content": '{}'}, answer("unknown"), {**answer(), "finish_reason": "length"}, {"content": '{"status":"reasonable"}'}):
            with self.subTest(response=response), self.assertRaises(ValueError):
                detection.parse_decision(response)

    def test_safe_group_is_never_queued(self):
        with patch.object(detection.repositories, "get_session") as get:
            self.assertIsNone(detection.queue_for_confidence(Mock(), 7, {"need_deep_analysis": False}))
            get.assert_not_called()

    def test_repeat_queue_preserves_completed_or_failed_attempt(self):
        for status in ("queued", "running", "success", "failed"):
            previous = {"status": status, "script_run_id": 10}
            with patch.object(detection.repositories, "get_session", return_value={"result_summary": {"detection_analysis": previous}}), patch.object(detection, "save") as save:
                result = detection.queue_for_confidence(Mock(), 7, {"id": 1, "need_deep_analysis": True})
                self.assertEqual(result, previous)
                save.assert_not_called()

    def test_queue_does_not_make_inline_api_request(self):
        with patch.object(detection.repositories, "get_session", return_value={"result_summary": {}}), patch.object(detection, "save") as save:
            result = detection.queue_for_confidence(Mock(), 7, {"id": 1, "need_deep_analysis": True})
            self.assertEqual(result["status"], "queued")
            self.assertEqual(result["attempt"], 0)
            save.assert_called_once()

    def test_retry_is_idempotent_for_queued_running_and_rejects_resolved(self):
        for status in ("queued", "running", "success"):
            previous = {"status": status, "decision": "reasonable"}
            with patch.object(detection, "session_lock", return_value=nullcontext(True)), patch.object(detection.repositories, "get_session", return_value={"result_summary": {"detection_analysis": previous}}), patch.object(detection, "save") as save:
                if status == "success":
                    with self.assertRaises(ValueError):
                        detection.retry(Mock(), 7)
                else:
                    self.assertEqual(detection.retry(Mock(), 7), previous)
                save.assert_not_called()

    def test_inconclusive_retry_explicitly_requests_a_new_billed_attempt(self):
        previous = {"status": "success", "decision": "insufficient_evidence", "script_run_id": 9}
        with patch.object(detection, "session_lock", return_value=nullcontext(True)), patch.object(detection.repositories, "get_session", return_value={"result_summary": {"detection_analysis": previous}}), patch.object(detection, "save"):
            result = detection.retry(Mock(), 7)
            self.assertEqual(result["script_run_id"], 9)
            self.assertTrue(result["new_attempt_requested"])
            self.assertEqual(result["status"], "queued")

    def test_failure_retry_preserves_old_run_for_answer_recovery(self):
        previous = {"status": "failed", "script_run_id": 9}
        with patch.object(detection, "session_lock", return_value=nullcontext(True)), patch.object(detection.repositories, "get_session", return_value={"result_summary": {"detection_analysis": previous}}), patch.object(detection, "save"):
            result = detection.retry(Mock(), 7)
            self.assertFalse(result["new_attempt_requested"])
            self.assertEqual(result["script_run_id"], 9)

    def test_streamed_actual_cost_and_raw_response_are_saved_before_validation(self):
        db = Mock()
        with patch.object(detection, "_write_journal"), patch.object(detection.repositories, "update_script_run_cost") as cost:
            detection._persist_response(db, 9, answer("invalid", 0.003), complete=True)
            db.commit.assert_called_once()
            log = json.loads(db.execute.call_args.args[1]["log"])
            self.assertEqual(log["response"]["usage"]["cost"], 0.003)
            cost.assert_called_once_with(db, 9, cost_amount=0.003, cost_currency="USD", cost_source="openrouter_actual")

    def test_partial_stream_does_not_become_a_recoverable_complete_answer(self):
        db = Mock()
        missing_path = Mock()
        missing_path.read_text.side_effect = FileNotFoundError
        patcher = patch.object(detection, "_journal_path", return_value=missing_path)
        patcher.start()
        self.addCleanup(patcher.stop)
        db.execute.return_value.mappings.return_value.first.return_value = {"stdout_log": json.dumps({"response": answer(), "response_complete": False})}
        # A stop finish marker makes a fully streamed answer recoverable even
        # if the process died before the final response persistence step.
        self.assertEqual(detection._saved_response(db, 9)["usage"]["cost"], 0.005)
        partial = answer()
        partial.pop("finish_reason")
        db.execute.return_value.mappings.return_value.first.return_value = {"stdout_log": json.dumps({"response": partial, "response_complete": False})}
        self.assertIsNone(detection._saved_response(db, 9))

    def test_mounted_journal_recovers_answer_after_database_persistence_failure(self):
        db = Mock()
        with tempfile.TemporaryDirectory() as directory, patch.object(detection.settings, "video_storage_dir", Path(directory)):
            db.execute.side_effect = RuntimeError("database unavailable")
            with self.assertRaisesRegex(RuntimeError, "database unavailable"):
                detection._persist_response(db, 9, answer("suspicious", 0.004), complete=True)
            db.execute.side_effect = None
            db.execute.return_value.mappings.return_value.first.return_value = {"stdout_log": ""}
            recovered = detection._saved_response(db, 9)
            self.assertEqual(recovered["usage"]["cost"], 0.004)
            self.assertEqual(detection.parse_decision(recovered)["decision"], "suspicious")

    def test_worker_operations_report_detection_sessions_not_legacy_video_jobs(self):
        db = Mock()
        db.execute.return_value.mappings.return_value.all.return_value = [{"location_id": 2, "queued_count": 3, "running_count": 1}]
        with patch.object(detection.repositories, "is_worker_paused", return_value=False):
            result = detection.worker_status(db)
        self.assertEqual(result["queued_count"], 3)
        self.assertEqual(result["running_count"], 1)
        self.assertEqual(result["max_global_workers"], 1)
        self.assertIn("detection_analysis", str(db.execute.call_args.args[0]))

    def test_media_ready_hit_queues_new_detection_but_never_legacy_analysis(self):
        row = {"id": 8, "batch_id": 3, "location_id": 2, "need_deep_analysis": True,
               "factor_payload": {"entry_trigger_ids": [1], "exit_trigger_ids": [2],
                                  "media_retrieval": {"status": "pending", "links": [
                                      {"video_asset_id": index, "section": role, "start": "2026-10-23T10:00:00", "end": "2026-10-23T10:05:00"}
                                      for index, role in ((1, "entry"), (2, "exit"), (3, "kiosk"))]}}}
        assets = [{"id": index, "status": "ready", "retrieved_at": "2026-10-23T10:06:00",
                   "metadata": {"retrieval_source": "confidence_evidence"}} for index in (1, 2, 3)]
        with (patch.object(detection, "ENABLED", True), patch.object(session_analysis, "PAUSED", True), patch.object(session_analysis, "no_transaction_check") as old,
              patch.object(workflow_service, "_ensure_session_for_confidence_group", return_value={"id": 7}), patch.object(confidence_media.repositories, "get_video_asset", side_effect=assets), patch.object(confidence_media.repositories, "update_session_grouping_link"), patch.object(confidence_media.repositories, "create_session_video_asset_link") as links, patch.object(confidence_media.repositories, "update_video_asset_status") as update, patch.object(detection, "queue_for_confidence") as queue, patch.object(confidence_media, "_save") as save):
            confidence_media._finish(Mock(), row)
            queue.assert_called_once()
            old.assert_not_called()
            self.assertEqual(links.call_count, 3)
            self.assertEqual(save.call_args.args[2]["session_analysis_status"], "detection_queued")
            self.assertTrue(all(not call.kwargs["metadata"]["analysis_eligible"] for call in update.call_args_list))


class DetectionRunTests(unittest.TestCase):
    def setUp(self):
        from app.services import detection_evidence
        self.db = Mock()
        self.analysis = {"status": "queued", "confidence_result_id": 8, "attempt": 0}
        self.session = {"id": 7, "entry_trigger_id": 1, "result_summary": {"detection_analysis": self.analysis}}
        self.row = {"id": 8, "batch_id": 3, "group_key": "1", "location_id": 2, "need_deep_analysis": True,
                    "factor_payload": {"media_retrieval": {"links": [{"video_asset_id": 9}]},
                                       "transaction_assignment": {"status": "complete", "unresolved_count": 0}}}
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.model = stack.enter_context(patch.object(detection, "current_model", return_value=detection.MODEL))
        stack.enter_context(patch.object(detection, "session_lock", return_value=nullcontext(True)))
        stack.enter_context(patch.object(detection.repositories, "get_session", return_value=self.session))
        stack.enter_context(patch.object(detection.repositories, "get_filter_confidence_result", return_value=self.row))
        self.create = stack.enter_context(patch.object(detection.repositories, "create_script_run_started", return_value=11))
        self.finish = stack.enter_context(patch.object(detection.repositories, "finish_script_run"))
        self.saved = stack.enter_context(patch.object(detection, "_saved_response", return_value=None))
        self.save = stack.enter_context(patch.object(detection, "save"))
        self.persist = stack.enter_context(patch.object(detection, "_persist_response"))
        self.build = stack.enter_context(patch.object(detection_evidence, "build_evidence", return_value=({}, [{"role": "kiosk", "image_url": "data:private"}])))
        self.api = stack.enter_context(patch.object(detection_evidence, "call_mimo", return_value=answer()))

    def test_one_new_paid_attempt_has_a_new_run_and_saves_answer_before_success(self):
        result = detection.run_session(self.db, 7)
        self.assertEqual(result["decision"], "reasonable")
        self.assertEqual(result["script_run_id"], 11)
        self.create.assert_called_once()
        self.assertEqual(self.create.call_args.kwargs["script_name"], "detection_analysis")
        self.api.assert_called_once()
        self.persist.assert_called_once_with(self.db, 11, answer(), complete=True)
        self.assertEqual(self.finish.call_args.kwargs["status"], "success")
        audit = next(call.args[1] for call in self.db.execute.call_args_list if "request_context" in str(call.args[0]))
        self.assertNotIn("data:private", audit["frames"])

    def test_dispatch_uses_current_saved_model_for_api_and_audit(self):
        self.model.return_value = "vendor/alternative-vision"
        detection.run_session(self.db, 7)
        self.assertEqual(self.create.call_args.kwargs["model_name"], "vendor/alternative-vision")
        self.assertEqual(self.api.call_args.kwargs["model_name"], "vendor/alternative-vision")
        self.assertEqual(self.create.call_args.kwargs["runner_payload"]["requested_model"], "vendor/alternative-vision")

    def test_already_succeeded_or_failed_is_not_sent_automatically(self):
        for status in ("success", "failed"):
            self.analysis["status"] = status
            detection.run_session(self.db, 7)
        self.api.assert_not_called()
        self.create.assert_not_called()

    def test_saved_answer_recovery_does_not_create_a_run_or_repeat_cost(self):
        self.analysis.update(script_run_id=10, retry_requested=True)
        self.saved.return_value = answer("suspicious")
        result = detection.run_session(self.db, 7)
        self.assertEqual(result["script_run_id"], 10)
        self.assertEqual(result["decision"], "suspicious")
        self.api.assert_not_called()
        self.create.assert_not_called()
        self.assertEqual(self.finish.call_args.args[1], 10)

    def test_explicit_unclear_retry_creates_new_run_retaining_previous_id(self):
        self.analysis.update(script_run_id=10, new_attempt_requested=True, decision="insufficient_evidence")
        result = detection.run_session(self.db, 7)
        self.saved.assert_not_called()
        self.api.assert_called_once()
        self.assertEqual(self.create.call_args.kwargs["runner_payload"]["previous_script_run_id"], 10)
        self.assertEqual(result["script_run_id"], 11)

    def test_lost_worker_never_automatically_repays(self):
        self.analysis.update(status="running", script_run_id=10)
        result = detection.run_session(self.db, 7)
        self.assertEqual(result["status"], "failed")
        self.assertIn("may have been billed", result["error"])
        self.api.assert_not_called()
        self.create.assert_not_called()

    def test_unresolved_receipt_ownership_waits_without_paid_request(self):
        self.row["factor_payload"]["transaction_assignment"] = {"status": "needs_review", "unresolved_count": 1}
        result = detection.run_session(self.db, 7)
        self.assertEqual(result["status"], "queued")
        self.assertIn("receipt ownership", result["message"])
        self.api.assert_not_called()
        self.create.assert_not_called()

    def test_invalid_answer_retains_raw_response_and_billing(self):
        self.api.return_value = answer("invalid")
        with self.assertLogs(detection.logger, level="ERROR"):
            result = detection.run_session(self.db, 7)
        self.assertEqual(result["status"], "failed")
        self.persist.assert_called_once_with(self.db, 11, answer("invalid"), complete=True)
        self.finish.assert_not_called()
        self.assertTrue(any("stderr_log" in str(call.args[0]) and "stdout_log" not in str(call.args[0]) for call in self.db.execute.call_args_list))


if __name__ == "__main__":
    unittest.main()
