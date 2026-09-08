ALTER TABLE sesamedb.tds_video_asset
    ADD COLUMN updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP AFTER created_at;

-- One-time immediate fix for whatever is stuck in 'retrieving' right now (e.g. video_asset_id=557),
-- since the ALTER above resets updated_at to "now" for existing rows and the stale-retry worker
-- logic would otherwise wait another retrieval_stale_seconds (15 min default) before reclaiming it.
UPDATE sesamedb.tds_video_asset SET status = 'not_retrieved' WHERE status = 'retrieving';
