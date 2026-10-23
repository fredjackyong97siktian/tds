from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

from app.services import grouping_result_summary as summary
from app import repositories


class GroupingSummaryTests(unittest.TestCase):
    def test_batch_page_uses_one_lookup_and_preserves_raw_audit(self):
        rows = [{"result_payload": {"groups": [{"entry": [1], "exit": [2]}], "unknown": [3, 4], "issues": [5]}},
                {"result_payload": {"open_entries": [6], "unknown": [4, 7], "excluded_non_customer_trigger_ids": [8]}}]
        original = deepcopy(rows)
        with patch.object(repositories, "list_confirmed_non_customer_trigger_ids", return_value={3, 4, 6}) as lookup:
            summary.annotate_batches(Mock(), rows)
        lookup.assert_called_once()
        self.assertEqual(lookup.call_args.args[1], [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(rows[0]["result_payload"]["excluded_non_customer_trigger_ids"], [3, 4])
        self.assertEqual(rows[1]["result_payload"]["excluded_non_customer_trigger_ids"], [4, 6, 8])
        self.assertEqual(rows[0]["result_payload"]["unknown"], original[0]["result_payload"]["unknown"])
        self.assertEqual(rows[0]["result_payload"]["issues"], [5])

    def test_empty_results_do_not_query(self):
        with patch.object(repositories, "list_confirmed_non_customer_trigger_ids") as lookup:
            summary.annotate_batches(Mock(), [{"result_payload": None}, {"result_payload": "invalid"}])
        lookup.assert_not_called()

    def test_trigger_references_are_distinct_and_support_object_ids(self):
        self.assertEqual(summary.result_trigger_ids({"groups": [{"entry": [1], "exit": [2]}],
            "open_entries": [{"trigger_id": 3}], "unknown": [4, 4, {"id": 5}, "bad", None]}), {1, 2, 3, 4, 5})
