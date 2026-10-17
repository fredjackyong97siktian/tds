import unittest
from unittest.mock import Mock, patch

from app.services import workflow_service as workflow


class ManualGroupIssueTests(unittest.TestCase):
    def test_issue_triggers_can_be_grouped_and_leave_issue_lists(self):
        batch = {"id": 3, "location_id": 2, "result_payload": {
            "groups": [], "open_entries": [], "unknown": [],
            "issues": [10, 11, 12], "frame_retrieval_issue_trigger_ids": [10, 12],
        }}
        items = [{"trigger_id": value, "frame_payload": {"frames": []}} for value in (10, 11)]
        with patch.object(workflow.repositories, "get_grouping_batch", return_value=batch), patch.object(workflow.repositories, "list_grouped_trigger_ids", return_value=set()), patch.object(workflow.repositories, "list_grouping_items", return_value=items), patch.object(workflow.repositories, "upsert_grouping_item") as save_item, patch.object(workflow.repositories, "update_grouping_batch") as update_batch, patch.object(workflow.repositories, "count_filter_confidence_results", return_value=0):
            result = workflow.add_manual_group_to_grouping_batch(Mock(), batch_id=3, entry_trigger_ids=[10], exit_trigger_ids=[11])
        summary = update_batch.call_args.args[2]["result_payload"]
        self.assertEqual(summary["issues"], [12])
        self.assertEqual(summary["frame_retrieval_issue_trigger_ids"], [12])
        self.assertEqual(result["group"]["entry"], [10])
        self.assertEqual(result["group"]["exit"], [11])
        self.assertEqual(save_item.call_count, 2)
        self.assertTrue(all(call.kwargs["status"] == "grouped" for call in save_item.call_args_list))


if __name__ == "__main__":
    unittest.main()
