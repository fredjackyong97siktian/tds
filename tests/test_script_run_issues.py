import unittest
from unittest.mock import Mock, patch

from app import repositories
from app.routers import workflows


class ScriptRunIssueTests(unittest.TestCase):
    def test_summary_is_paginated_failed_only_and_does_not_fetch_full_logs(self):
        db = Mock()
        db.execute.return_value.scalar.return_value = 41
        rows = [{"script_run_id": 8, "script_name": "entry", "status": "failed", "issue_message": "error"}]
        with patch.object(repositories, "_fetch_all_dicts", return_value=rows):
            result = repositories.list_script_run_issues(db, limit=20, offset=20)
        self.assertEqual(result, {"items": rows, "total": 41})
        summary, count = db.execute.call_args_list
        sql = str(summary.args[0])
        self.assertIn("where status='failed'", sql)
        self.assertIn("left(stderr_log, 1000)", sql)
        self.assertNotIn("stdout_log", sql)
        self.assertNotIn("runner_payload", sql)
        self.assertEqual(summary.args[1], {"limit": 20, "offset": 20})
        self.assertIn("where status='failed'", str(count.args[0]))

    def test_repository_clamps_page_size_and_offset(self):
        db = Mock()
        db.execute.return_value.scalar.return_value = 0
        with patch.object(repositories, "_fetch_all_dicts", return_value=[]):
            result = repositories.list_script_run_issues(db, limit=10000, offset=-50)
        self.assertEqual(db.execute.call_args_list[0].args[1], {"limit": 100, "offset": 0})
        self.assertEqual(result["total"], 0)

    def test_route_forwards_pagination_without_running_any_script(self):
        db = Mock()
        with patch.object(repositories, "list_script_run_issues", return_value={"items": [], "total": 0}) as query:
            result = workflows.list_script_run_issues(limit=20, offset=40, db=db)
        query.assert_called_once_with(db, limit=20, offset=40)
        self.assertEqual(result["items"], [])


if __name__ == "__main__":
    unittest.main()
