# Theft Detection API Docs

This document explains how the FastAPI server in `tds/` works, which API is used in the business flow, and which database each endpoint writes to.

## Database Assignment

- MySQL is used for transactional records:
  - `trigger_event`
  - `session`
  - `session_customer`
  - `video_asset`
  - `session_video_asset`
  - `kiosk_video_result`
  - `session_transaction`
  - `script_run`
- PostgreSQL is used for vector and gallery records:
  - `customer_gallery`
  - `active_gallery`

## Base Service

- FastAPI app entrypoint:
  - [app/main.py](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/app/main.py)
- Default local run:

```bash
cd /Users/fredjackyong/Documents/kebunapp/theft_detection/tds
uvicorn app.main:app --reload --port 8010
```

- Base URL:

```text
http://127.0.0.1:8010
```

- Swagger docs:

```text
http://127.0.0.1:8010/docs
```

## Important Behavior

Right now:

- `run-entry` waits until the Entry script finishes
- `run-kiosk` waits until the Kiosk script finishes
- both are synchronous API calls

That means the HTTP request stays open while the Python script is running.

## How Entry and Kiosk Are Run

The API does not call the old standalone `Entry.py` and `Kiosk.py` files directly.

Instead it calls:

- [DetectEntry.py](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/DetectEntry.py)
- [DetectKiosk.py](/Users/fredjackyong/Documents/kebunapp/theft_detection/tds/DetectKiosk.py)

Those runners:

1. configure model paths from `tds/models`
2. load shared gallery state from `shared_gallery_state.pkl`
3. import [Detect.py](/Users/fredjackyong/Documents/kebunapp/theft_detection/Detect.py)
4. run only the Entry logic or only the Kiosk logic
5. save updated shared state back to disk

## Recommended Flow

This section reflects the latest PDF flow, even where the current API still has gaps.

### `n8n` Entry Flow

1. `POST /api/v1/triggers`
2. wait 3 minutes
3. resolve entry method and whitelist result, then update `trigger_event`
   - there is no dedicated HTTP endpoint for this yet; current integration must update MySQL directly or add a future endpoint
4. if whitelisted:
   - mark the trigger as `whitelisted`
   - stop the flow
5. if there is no matched entry method:
   - stop the flow for now
6. use `GET /api/v1/workflows/triggers/{trigger_id}/video-ready-policy` to decide retry/issue behavior while waiting for the entrance CCTV clip
7. retrieve and save the entrance video
8. register the raw entrance video in `video_asset`
   - current route: `POST /api/v1/videos/triggers/{trigger_id}`
9. once the entrance video is `ready`, the Theft Detection System location worker picks it up

### Theft Detection System Flow Per Location

1. schedule poll per location
2. stop immediately if that location already has a `processing` entrance video
3. otherwise claim the latest or oldest `ready` entrance video for that location
4. run Entry script on that video
5. on `Detect Enter`:
   - create the `session`
   - create `session_customer`
   - save ReID image
   - write `customer_gallery`
   - write `active_gallery`
6. on `Detect Exit`:
   - find related session and customer
   - save ReID image
   - update `customer_gallery`
   - update `session_customer.leave_time`
   - if all customers have left, close the session
7. once a session is closed, kiosk processing runs in the server background
8. retrieve kiosk video between session `start_time` and `end_time`
9. run Kiosk script
10. insert POS transactions
11. finalize the session result

### Important Current Gaps

- the current implemented video route still uses `POST /api/v1/videos/sessions/{session_id}`, but the latest schema now models raw videos separately from `session_video_asset`
- the current implemented vector routes still use the old `gallery-runtime-state` path name even though the PostgreSQL table is now `active_gallery`
- there is not yet a dedicated endpoint for updating entry-method resolution on `trigger_event`
- `run-kiosk` is still an explicit API call today; the latest business flow expects kiosk processing to continue in the server background once the session closes

## Endpoint Guide

## 1. Health Check

### Request

```http
GET /health
```

### Response

```json
{
  "status": "ok",
  "service": "Theft Detection API",
  "version": "0.1.0"
}
```

## 2. Create Trigger

This endpoint creates only the `trigger_event` record. It does not create a session automatically.

### Request

```http
POST /api/v1/triggers
Content-Type: application/json
```

```json
{
  "location_id": 1,
  "aqara_event_id": "aqara-evt-001",
  "trigger_source": "aqara",
  "trigger_time": "2026-06-28T10:30:00+08:00",
  "raw_payload": {
    "device": "aqara-door-sensor",
    "action": "open"
  }
}
```

### Response

```json
{
  "id": 101,
  "location_id": 1,
  "status": "pending",
  "trigger_time": "2026-06-28T10:30:00+08:00"
}
```

## 3. Create Session

This endpoint creates the `session` record. In the latest intended flow, this should happen only after Entry detection confirms a real group entry, not immediately after the trigger.

Current implementation note:
- the live request model in `tds/app/schemas.py` still expects `trigger_id` and still allows `entry_video_url`
- the example below shows the target payload shape implied by the newer MySQL schema

### Request

```http
POST /api/v1/sessions
Content-Type: application/json
```

```json
{
  "entry_trigger_id": 101,
  "location_id": 1,
  "start_time": "2026-06-28T10:30:00+08:00"
}
```

### Response

```json
{
  "id": 55,
  "entry_trigger_id": 101,
  "location_id": 1,
  "status": "pending",
  "start_time": "2026-06-28T10:30:00+08:00",
  "end_time": null,
  "total_item_brought": 0,
  "transaction_total_items": 0
}
```

## 4. Add Or Update Session Customer

Use this when Entry or later automation knows a specific customer/person record for the session.

### Request

```http
POST /api/v1/sessions/55/customers
Content-Type: application/json
```

```json
{
  "person_id": 1,
  "enter_time": "2026-06-28T10:31:00+08:00",
  "kiosk_start_time": "2026-06-28T10:34:00+08:00",
  "leave_time": null,
  "match_status": "tracked"
}
```

## 5. Check Video Ready Policy

This endpoint helps `n8n` decide whether to retry video retrieval or stop the flow.

### Request

```http
GET /api/v1/workflows/triggers/101/video-ready-policy?created_time=2026-06-28T10:30:00+08:00&retries_used=1
```

### Meaning

- `created_time`: the original trigger or request time
- `retries_used`: how many retrieval attempts have already been used

### Response

```json
{
  "trigger_id": 101,
  "retries_used": 1,
  "retry_limit": 3,
  "wait_minutes_between_retries": 5,
  "next_retry_after": "2026-06-28T02:40:00+00:00",
  "should_mark_issue": false,
  "recommended_action": "retry_when_ready",
  "explanation": "Video is still within retry budget. Wait the suggested interval and check again."
}
```

When `should_mark_issue = true`, the downstream flow should stop and the trigger should be treated as an issue case.

## 6. Register A Video Record

Use this endpoint after a real video file has been retrieved, uploaded, or saved.

Important:
- the latest schema now separates raw `video_asset` rows from `session_video_asset` link rows
- videos are now stored under a private location/trigger/session folder structure
- the API stores a canonical private `file_path` and exposes access through an API route
- callers can still send an original `video_url` or source path, but the API keeps those in metadata

### Request

Raw trigger-owned video:

```http
POST /api/v1/videos/triggers/101
Content-Type: application/json
```

Session-linked video:

```http
POST /api/v1/videos/sessions/55
Content-Type: application/json
```

```json
{
  "section": "entrance",
  "video_url": "https://example.com/videos/1_ENTRY.mp4",
  "file_path": "/data/videos/1_ENTRY.mp4",
  "captured_start_time": "2026-06-28T10:30:00+08:00",
  "captured_end_time": "2026-06-28T10:35:00+08:00",
  "metadata": {
    "camera": "entry_cam_01"
  }
}
```

### Meaning

- `section` tells the system whether this video is `entrance` or `kiosk`
- raw `video_asset` rows are separated from session linkage in `session_video_asset`
- `video_url` is rewritten to a private API access route like `/api/v1/videos/assets/{id}/content`
- `file_path` is normalized into the private storage structure
- `captured_start_time` and `captured_end_time` help match this video to the session timeline
- `metadata` can store recorder or camera details

The same endpoint is used for both Entrance video and Kiosk video. The `section` value is what distinguishes them.

### Private Video Access

```http
GET /api/v1/videos/assets/{video_asset_id}/content
```

This streams the private local video file through FastAPI instead of exposing a public object URL.

## 7. Run Entry Script

This triggers `DetectEntry.py`.

### Request

```http
POST /api/v1/workflows/triggers/101/run-entry?session_id=55
Content-Type: application/json
```

```json
{
  "video_path": "/absolute/path/to/1_ENTRY.mp4"
}
```

### Optional Fields

```json
{
  "video_path": "/absolute/path/to/1_ENTRY.mp4",
  "output_dir": "/absolute/path/to/custom/output/logs/1_ENTRY",
  "gallery_state_path": "/absolute/path/to/custom/output/shared_gallery_state.pkl"
}
```

### Meaning

- `video_path` is the video file to process
- `output_dir` overrides the default output folder
- `gallery_state_path` overrides the shared cross-video gallery-state pickle
- `session_id` ties the script run to the correct session work directory and log records

### Behavior

1. FastAPI builds the session output paths
2. FastAPI runs `DetectEntry.py`
3. stdout and stderr are stored in `script_run`
4. the API returns only after the script finishes

In the latest target flow, this endpoint is called by the Theft Detection System worker after it claims a `ready` entrance video for a location.

## 8. Close Session

Use this after Entry/Exit logic determines the group has left.

### Request

```http
POST /api/v1/sessions/55/close?exit_trigger_id=202
```

### Meaning

- marks the session as `closed`
- writes the end time
- stores the `exit_trigger_id` if supplied
- in the latest business flow, closing the session is the handoff point to background kiosk processing

## 9. Retrieve Entrance Video

Use this before session creation, after the trigger exists and before Entry detection confirms a real group.

### Request

```http
POST /api/v1/workflows/triggers/101/retrieve-entrance-video
Content-Type: application/json
```

```json
{
  "start_time": "2026-06-28T10:30:00+08:00",
  "end_time": "2026-06-28T10:35:00+08:00",
  "location_id": 1
}
```

## 10. Retrieve Kiosk Video Window

Use this only after the session already exists.

### Request

```http
POST /api/v1/workflows/sessions/55/retrieve-kiosk-video
Content-Type: application/json
```

```json
{
  "start_time": "2026-06-28T10:31:00+08:00",
  "end_time": "2026-06-28T10:39:00+08:00",
  "location_id": 1
}
```

## 11. Run Kiosk Script

This triggers `DetectKiosk.py`.

### Request

```http
POST /api/v1/workflows/sessions/55/run-kiosk
Content-Type: application/json
```

```json
{
  "video_path": "/absolute/path/to/2_KIOSK.mp4"
}
```

### Meaning

- uses the same session work directory and shared gallery state as Entry
- reuses the learned person IDs where possible
- stores stdout and stderr in MySQL `script_run`

Use `POST /api/v1/videos/sessions/{session_id}` before this if you also want a persistent record of the kiosk video asset itself.

## 12. Add Transaction Record

Use this for real POS or receipt records that happened inside the session window.

### Request

```http
POST /api/v1/sessions/55/transactions
Content-Type: application/json
```

```json
{
  "receipt_number": "RCP-20260628-0001",
  "transaction_time": "2026-06-28T10:40:00+08:00",
  "total_items": 3,
  "total_amount": 15.5,
  "raw_payload": {
    "payment_method": "card"
  }
}
```

## 13. Finalize Session Result

This is the new endpoint that automatically sets the final session result to:

- `not_detected`
- `detected`
- `need_review`

### Request

```http
POST /api/v1/sessions/55/finalize
Content-Type: application/json
```

```json
{
  "kiosk_total_items": 4,
  "actual_items_brought": 4
}
```

### Decision Logic

- if `kiosk_total_items == transaction_total_items`, status becomes `not_detected`
- if `kiosk_total_items > transaction_total_items`, status becomes `detected`
- if `kiosk_total_items < transaction_total_items`, status becomes `need_review`

The endpoint calculates `transaction_total_items` automatically by summing the MySQL `session_transaction.total_items` rows for that session.

### Response

```json
{
  "session_id": 55,
  "status": "detected",
  "kiosk_total_items": 4,
  "transaction_total_items": 3,
  "actual_items_brought": 4,
  "result_summary": {
    "kiosk_total_items": 4,
    "transaction_total_items": 3,
    "actual_items_brought": 4,
    "difference": 1,
    "decision": "detected"
  }
}
```

## 13. Create Customer Gallery Record

This endpoint writes PostgreSQL vector/gallery records.

### Request

```http
POST /api/v1/vector/sessions/55/customer-gallery
Content-Type: application/json
```

```json
{
  "location_id": 1,
  "person_id": 1,
  "session_customer_id": 12,
  "image_url": "/app/session/locations/location_1/sessions/session_55/reid/session_customers/sc_12/entry/p1-front.jpg",
  "image_kind": "reid_view",
  "embedding_osnet": [0.11, 0.22, 0.33],
  "embedding_fashion": [0.44, 0.55, 0.66],
  "metadata": {
    "camera": "entry_cam_01",
    "image_path": "/app/session/locations/location_1/sessions/session_55/reid/session_customers/sc_12/entry/p1-front.jpg"
  }
}
```

## 14. List Customer Gallery Records

### Request

```http
GET /api/v1/vector/sessions/55/customer-gallery
```

This returns all PostgreSQL `customer_gallery` rows for the session.

### Private ReID Image Access

```http
GET /api/v1/vector/customer-gallery/{gallery_id}/image
```

This streams the private stored ReID image through FastAPI.

## 15. Upsert Gallery Runtime State

Use this to persist active-customer gallery state into PostgreSQL instead of only keeping it in a pickle file.

Important:
- the PostgreSQL table is now `active_gallery`
- each row represents one active customer at one location, keyed by `session_customer_id`

### Request

```http
PUT /api/v1/vector/locations/1/active-gallery/12
Content-Type: application/json
```

```json
{
  "session_id": 55,
  "session_customer_id": 12,
  "person_id": 1,
  "state_kind": "active_gallery",
  "state_payload": {
    "customer_gallery_ids": [5001, 5002],
    "primary_gallery_entry_id": 5001,
    "is_active": true
  },
  "metadata": {
    "source": "entry_runner",
    "location_id": 1
  }
}
```

## 16. Get Active Gallery State

### Request

```http
GET /api/v1/vector/locations/1/active-gallery/12/active_gallery
```

This returns the PostgreSQL `active_gallery` row for the given `location_id`, `session_customer_id`, and `state_kind`.

## Current Limitations

1. Entry and Kiosk runs are synchronous
2. Da Hua retrieval is not implemented yet
3. `tds` still depends on [Detect.py](/Users/fredjackyong/Documents/kebunapp/theft_detection/Detect.py)
4. `DetectEntry.py` and `DetectKiosk.py` still use a local pickle file for shared state unless your integration explicitly calls the vector endpoints
