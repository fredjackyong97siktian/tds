# Private Media Storage Layout

## Goal

Keep videos and ReID images:

- organized by `location`
- traceable by `trigger`, `session`, and `session_customer`
- private by default
- accessible only through API endpoints, not through public direct URLs

## Base Root

The API uses:

- `THEFT_API_VIDEO_STORAGE_DIR`

This directory is treated as the private media root.

Recommended value:

```text
/app/session
```

Inside that root, the API uses this structure:

```text
<video_storage_dir>/
  locations/
    location_<location_id>/
      triggers/
        trigger_<trigger_id>/
          entrance/
            raw/
              <video-file>
      sessions/
        session_<session_id>/
          videos/
            entrance/
              raw/
                <video-file>
            kiosk/
              raw/
                <video-file>
          reid/
            session_customers/
              sc_<session_customer_id>/
                entry/
                  <image-file>
                kiosk/
                  <image-file>
                exit/
                  <image-file>
          scripts/
            logs/
              <run-name>/
                ...
          output/
            ...
```

## Meaning

- `triggers/.../entrance/raw/`
  - raw entrance video before or independent of a confirmed session
- `sessions/.../videos/.../raw/`
  - raw video linked to a confirmed session
- `reid/session_customers/sc_<id>/...`
  - cropped ReID images for one `session_customer`
- `scripts/`
  - worker logs and gallery state files
- `output/`
  - script outputs that are not original evidence files

## API Rules

## Video Registration

When the API registers a video:

- it computes a canonical private `file_path`
- it stores any caller-supplied path as metadata only
- it returns an API access URL like:

```text
/api/v1/videos/assets/<video_asset_id>/content
```

That means:

- `video_asset.file_path` points to the private on-disk path
- `video_asset.video_url` can be the API access route instead of a public object URL

## ReID Image Access

For `customer_gallery`:

- the actual local image path should be stored in:
  - `metadata.image_path`
  - or `image_url` if you use it as a private local path

Images can then be accessed through:

```text
/api/v1/vector/customer-gallery/<gallery_id>/image
```

## Privacy Rule

Videos and ReID images should not be exposed as public bucket URLs.

Recommended access model:

1. store file privately on local disk or private object storage
2. save only internal storage path in DB/metadata
3. expose the file through authenticated API endpoints

## Database Mapping

## MySQL

- `video_asset.file_path`
  - canonical private local path
- `video_asset.video_url`
  - API access URL or another internal reference
- `session_video_asset`
  - session-to-video linkage

## PostgreSQL

- `customer_gallery.image_url`
  - optional logical reference
- `customer_gallery.metadata.image_path`
  - recommended private local path for ReID image files

## Current API Endpoints

### Private Video Access

```http
GET /api/v1/videos/assets/{video_asset_id}/content
```

### Private ReID Image Access

```http
GET /api/v1/vector/customer-gallery/{gallery_id}/image
```

## Notes

- the API validates that streamed files stay inside the configured private media root
- this prevents arbitrary file reads outside the storage directory
- if you later move to DigitalOcean Spaces private buckets, keep the same rule:
  - API issues access
  - media is not public by default

