import unittest
from sqlalchemy import create_engine, text

from app import repositories


class RetrievalFilterTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://")
        self.addCleanup(self.engine.dispose)
        self.connection = self.engine.connect()
        self.addCleanup(self.connection.close)
        self.connection.execute(text("create table runs (id integer, script_name text, model_name text)"))
        rows = [
            (1, "retrieve_video", "dahua_rtsp_playback:trigger_frames"),
            (2, "retrieve_video", "dahua_rtsp_playback:trigger_frame_asset"),
            (3, "retrieve_video", "dahua_rtsp_playback:identifier_frames"),
            (4, "retrieve_video", "dahua_rtsp_playback:kiosk_payment_frames"),
            (5, "retrieve_video", "worker_build_frame_job"),
            (6, "retrieve_video", "dahua_rtsp_playback:libx264"),
            (7, "retrieve_video", "worker_build_job"),
            (8, "retrieve_video", None),
            (9, "grouping", "openrouter-jev-1.14"),
            (10, "retrieve_frame", "future_frame_model"),
        ]
        self.connection.execute(text("insert into runs values (:id, :script_name, :model_name)"),
            [{"id": row[0], "script_name": row[1], "model_name": row[2]} for row in rows])

    def rows(self, script_name, model_name=None):
        filters, params = repositories._script_run_list_filters(script_name=script_name, script_type=None, model_name=model_name)
        sql = "select id from runs" + (f" where {' and '.join(filters)}" if filters else "") + " order by id"
        return [row[0] for row in self.connection.execute(text(sql), params)]

    def test_frame_filter_includes_all_legacy_captures_and_frame_build_errors(self):
        self.assertEqual(self.rows("retrieve_frame"), [1, 2, 3, 4, 5, 10])

    def test_video_filter_excludes_frames_but_keeps_video_build_errors(self):
        self.assertEqual(self.rows("retrieve_video"), [6, 7, 8])

    def test_filters_do_not_overlap_and_do_not_lose_any_retrieval(self):
        frames, videos = set(self.rows("retrieve_frame")), set(self.rows("retrieve_video"))
        self.assertFalse(frames & videos)
        self.assertEqual(frames | videos, {1, 2, 3, 4, 5, 6, 7, 8, 10})

    def test_model_filter_still_applies(self):
        self.assertEqual(self.rows("retrieve_frame", "dahua_rtsp_playback:kiosk_payment_frames"), [4])

    def test_no_script_filter_still_includes_everything(self):
        self.assertEqual(self.rows(None), list(range(1, 11)))


if __name__ == "__main__":
    unittest.main()
