import unittest
from unittest.mock import Mock, patch

from app import repositories
from app.routers import filtering, frames


class ConfidenceStatusFilterTests(unittest.TestCase):
    def test_detected_is_scoped_to_group_and_latest_session(self):
        with patch.object(repositories, "_column_exists", return_value=True):
            clause = repositories._confidence_status_clause(Mock(), "detected")
        self.assertIn("status_session.status = 'detected'", clause)
        self.assertIn("status_session.grouping_id = c.batch_id", clause)
        self.assertIn("newer_session.id > status_session.id", clause)
        self.assertIn("$.entry_trigger_ids", clause)
        self.assertIn("status_item.role = 'entry'", clause)
        self.assertIn("status_item.status <> 'deleted'", clause)

    def test_l1_filter_matches_existing_node_semantics(self):
        with patch.object(repositories, "_column_exists", return_value=True):
            clause = repositories._confidence_status_clause(Mock(), "l1")
        self.assertIn("coalesce(c.need_deep_analysis, 0) = 1 or", clause)
        self.assertIn("status_session.status = 'detected'", clause)

    def test_old_schema_has_no_detected_results_but_l1_still_works(self):
        with patch.object(repositories, "_column_exists", return_value=False):
            self.assertEqual(repositories._confidence_status_clause(Mock(), "detected"), "false")
            self.assertIn("c.need_deep_analysis", repositories._confidence_status_clause(Mock(), "l1"))

    def test_all_statuses_does_not_query_schema(self):
        with patch.object(repositories, "_column_exists") as exists:
            self.assertIsNone(repositories._confidence_status_clause(Mock(), None))
        exists.assert_not_called()

    def test_invalid_status_rejected(self):
        with self.assertRaises(ValueError):
            repositories._confidence_status_clause(Mock(), "invalid")

    def test_list_and_count_use_same_filter_before_pagination(self):
        db = Mock()
        db.execute.return_value.mappings.return_value.all.return_value = []
        db.execute.return_value.mappings.return_value.first.return_value = {"total": 12}
        with patch.object(repositories, "_column_exists", return_value=True):
            repositories.list_filter_confidence_results(db, limit=10, offset=10, batch_id=4,
                customer_identity=("phone", "60123"), status="detected")
            self.assertEqual(repositories.count_filter_confidence_results(db, batch_id=4,
                customer_identity=("phone", "60123"), status="detected"), 12)
        listing, counting = db.execute.call_args_list
        for call in (listing, counting):
            sql = str(call.args[0])
            self.assertIn("status_session.status = 'detected'", sql)
            self.assertIn("c.batch_id = :batch_id", sql)
            self.assertEqual(call.args[1]["customer_value"], "60123")
        self.assertLess(str(listing.args[0]).index("status_session.status"), str(listing.args[0]).index("limit :limit"))

    def test_routes_forward_status_to_list_and_count(self):
        with patch.object(repositories, "list_filter_confidence_results", return_value=[]) as listing, \
             patch.object(repositories, "count_filter_confidence_results", return_value=3) as counting:
            filtering.list_confidence_results(status="l1", db=Mock())
            self.assertEqual(filtering.count_confidence_results(status="detected", db=Mock()), {"total": 3})
        self.assertEqual(listing.call_args.kwargs["status"], "l1")
        self.assertEqual(counting.call_args.kwargs["status"], "detected")


class FrameFilterTests(unittest.TestCase):
    def test_open_and_closed_trigger_bounds_are_inclusive(self):
        for kwargs, present, absent in (
            ({"trigger_id_min": 100}, "fa.trigger_id >= :trigger_id_min", "trigger_id_max"),
            ({"trigger_id_max": 200}, "fa.trigger_id <= :trigger_id_max", "trigger_id_min"),
        ):
            clause, params = repositories._trigger_frame_asset_filters(**kwargs)
            self.assertIn(present, clause)
            self.assertNotIn(absent, clause)
            self.assertEqual(params, kwargs)
            self.assertIn("fa.status <> 'deleted'", clause)

    def test_frames_paginate_in_requested_direction(self):
        db = Mock()
        db.execute.return_value.mappings.return_value.all.return_value = []
        repositories.list_trigger_frame_assets(db, limit=10, offset=20, trigger_id_min=100, order="trigger_asc")
        sql, params = db.execute.call_args.args
        self.assertIn("order by fa.trigger_id asc, fa.id asc", str(sql))
        self.assertIn("offset :offset", str(sql))
        self.assertEqual(params, {"limit": 10, "offset": 20, "trigger_id_min": 100})

    def test_count_uses_same_filters_without_limit(self):
        db = Mock()
        db.execute.return_value.mappings.return_value.first.return_value = {"total": 999}
        self.assertEqual(repositories.count_trigger_frame_assets(db, trigger_id_max=300), 999)
        sql, params = db.execute.call_args.args
        self.assertIn("fa.trigger_id <= :trigger_id_max", str(sql))
        self.assertNotIn("limit", str(sql))
        self.assertEqual(params, {"trigger_id_max": 300})

    def test_sort_order_is_validated(self):
        db = Mock()
        with self.assertRaises(ValueError):
            repositories.list_trigger_frame_assets(db, order="invalid")
        db.execute.assert_not_called()
