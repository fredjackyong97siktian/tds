import json
import unittest
from unittest.mock import Mock, patch

from app.routers import sessions
from app.services import detection_analysis


class DetectionIssuesTests(unittest.TestCase):
    def test_query_includes_only_unresolved_analysis_and_active_explicit_retries(self):
        db = Mock()
        db.execute.return_value.scalar.return_value = 1
        db.execute.return_value.mappings.return_value.all.return_value = [{
            "session_id": 7, "location_id": 2, "updated_at": "2026-10-23T01:00:00+00:00",
            "analysis": json.dumps({"status": "success", "decision": "insufficient_evidence", "script_run_id": 9, "confidence_result_id": 8})}]
        with patch.object(detection_analysis.repositories, "is_worker_paused", return_value=False):
            result = detection_analysis.list_issues(db, limit=10, offset=20)
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["items"][0]["analysis"]["script_run_id"], 9)
        self.assertEqual(result["items"][0]["analysis"]["confidence_result_id"], 8)
        count_sql = str(db.execute.call_args_list[0].args[0])
        select_sql = str(db.execute.call_args_list[1].args[0])
        for sql in (count_sql, select_sql):
            self.assertIn("insufficient_evidence", sql)
            self.assertIn("'failed'", sql)
            self.assertIn("'queued','running'", sql)
            self.assertIn("retry_requested", sql)
            self.assertNotIn("from tds_script_run", sql)
            self.assertNotIn("stdout_log", sql)
        self.assertEqual(db.execute.call_args.args[1], {"limit": 10, "offset": 20})

    def test_empty_page_is_explicit_and_does_not_create_paid_work(self):
        db = Mock()
        db.execute.return_value.scalar.return_value = 0
        db.execute.return_value.mappings.return_value.all.return_value = []
        with patch.object(detection_analysis.repositories, "is_worker_paused", return_value=True), patch.object(detection_analysis, "run_session") as api:
            result = detection_analysis.list_issues(db, limit=100, offset=-2)
        self.assertEqual(result["items"], [])
        self.assertEqual(result["limit"], 50)
        self.assertEqual(result["offset"], 0)
        self.assertTrue(result["worker_paused"])
        api.assert_not_called()

    def test_route_forwards_pagination(self):
        db = Mock()
        with patch.object(detection_analysis, "list_issues", return_value={"items": [], "total": 0}) as query:
            sessions.get_detection_analysis_issues(limit=20, offset=40, db=db)
        query.assert_called_once_with(db, limit=20, offset=40)


if __name__ == "__main__":
    unittest.main()
