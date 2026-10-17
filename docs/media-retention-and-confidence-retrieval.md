# Media Retention And Confidence Retrieval

## Daily Cleaner

`theft-cleaner-worker` runs the cleanup once daily at 04:00 Asia/Kuala_Lumpur.
The database stays in UTC. A restart after 04:00 catches up that day's run;
the last successful day and counts are stored in `tds_app_setting` under
`media_cleaner_last_success`. A MySQL named lock prevents two cleaners running
at the same time. Errors leave the run retryable.

The cutoff is one calendar month before the current time. The cleaner removes
old videos and images from private local storage and the configured Spaces
bucket. It checks both record age and physical file/object age. It also sweeps
derived media under registered trigger/session object prefixes and local
directories. It never deletes manifests, script logs, models, customer identity
tables, transaction records, groups, confidence results, or session records.
Expired media rows are marked `deleted` instead of deleting their audit history.
Legacy `retention_until` values do not override the monthly policy.

Protection includes detected sessions, sessions with a TDS System theft alert,
all triggers in their matched groups, and linked videos. Shared media URLs,
local paths and owner directories are protected too. Recent media and running
retrieval/analysis jobs are excluded. A failure to read the protection data
aborts cleanup before any files are removed.

Deploy with the existing Compose command, including the new cleaner service:

```bash
docker compose up -d --build
```

Preview only, without deleting objects/files or updating media rows:

```bash
docker compose exec theft-cleaner-worker python -m app.workers.cleaner_worker --dry-run
```

The configured Spaces key needs ListBucket, GetObject and DeleteObject access
for the application's own media prefix. The MySQL user needs SELECT on the
existing session/grouping/alert/script/media tables, UPDATE on `tds_video_asset`,
`tds_trigger_frame`, `tds_trigger_frame_asset`, and SELECT/INSERT/UPDATE on
`tds_app_setting`. No new tables, status constraints, or script names are needed.

## Confidence Evidence Before Sessions

Every newly scored completed group records `factor_payload.media_retrieval`
atomically with the confidence result. This applies whether L1 is required or
not, and works for scheduled, manual, self-grouped, and rerun confidence scoring.
Existing historical confidence results are not automatically backfilled.

The plan requests:

- Entry: 40 seconds before through 10 seconds after each entry trigger.
- Exit: 40 seconds before through 10 seconds after every exit trigger.
- Kiosk: the earliest entry through the last exit, including groups without
  transactions. This preserves evidence without deciding receipt ownership.

Asset rows start as `not_retrieved`. They reference triggers and the confidence
result, so the normal retrieval worker can retrieve them before a session exists.
Normal camera/NVR delay correction still applies. Matching existing full-video
assets are reused, and frame-only legacy assets are excluded from reuse.

The retrieval worker periodically finishes pending plans. It creates/links the
session only once all required videos have succeeded. Retrieval failures leave
the plan waiting, with failed video IDs; retry them in Confidence Result detail.
It never creates a partial session to bypass a failed entry, exit or kiosk clip.
The existing `session_creation_enabled` setting still controls session creation.

New assets are initially ineligible for automatic L1 analysis. After all media
are ready and the session is attached, only L1-required groups enable entrance
analysis. Safe groups get stored evidence and a session without paid analysis.
The retired kiosk analysis worker is not restored by this change.

Confidence Result detail shows Visit Videos, their retrieval state, playback
when ready, and retry for failed retrieval. The original confidence decision
and factor evidence remain unchanged.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests use temporary files and mocked storage/database connections. They never
delete from a live bucket or database.
