from __future__ import annotations

import calendar
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse
from zoneinfo import ZoneInfo

from sqlalchemy import text

from .. import repositories
from ..config import settings
from ..spaces import _spaces_client, generate_public_object_url, is_spaces_configured
from ..storage import build_spaces_object_key

logger = logging.getLogger(__name__)
MEDIA_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".jpg", ".jpeg", ".png", ".webp", ".gif"}
ZONE = ZoneInfo("Asia/Kuala_Lumpur")


def month_ago(now):
    month = now.month - 1 or 12
    year = now.year - (now.month == 1)
    return now.replace(year=year, month=month, day=min(now.day, calendar.monthrange(year, month)[1]))


def scheduled_date(now):
    local = now.astimezone(ZONE)
    return local.date().isoformat() if local.hour >= 4 else None


def _json(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return {}
    return value or {}


def media_references(value):
    """Only our bucket URLs/keys and private media paths are deletion targets."""
    local, remote = set(), set()
    root = settings.video_storage_dir.resolve()
    root_key = build_spaces_object_key()
    prefix = root_key + "/" if root_key else ""
    public = urlparse(generate_public_object_url("")) if is_spaces_configured() else None
    def visit(item):
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, str):
            key = None
            if item.startswith("spaces://"):
                key = item[len("spaces://"):]
            elif prefix and item.startswith(prefix):
                key = item
            elif public and item.startswith(("https://", "http://")):
                parsed = urlparse(item)
                endpoint = urlparse(str(settings.spaces_endpoint_url))
                allowed_hosts = {public.netloc, f"{settings.spaces_bucket}.{endpoint.netloc}"}
                if parsed.netloc in allowed_hosts:
                    path = unquote(parsed.path)
                    base = public.path.rstrip("/") + "/"
                    key = path.removeprefix(base).lstrip("/") if parsed.netloc == public.netloc else path.lstrip("/")
            elif item.startswith("/"):
                path = Path(item).resolve()
                if root in path.parents and path.suffix.lower() in MEDIA_SUFFIXES:
                    local.add(path)
            if key and key.startswith(prefix or "location_") and ".." not in key.split("/") and Path(key).suffix.lower() in MEDIA_SUFFIXES:
                remote.add(key)
    visit(value)
    return local, remote


def _rows(db, sql, params=None):
    return [dict(row) for row in db.execute(text(sql), params or {}).mappings().all()]


def protected_owners(db, cutoff):
    table = repositories._table
    sessions = _rows(db, f"""
        select s.id, s.location_id, s.entry_trigger_id, s.exit_trigger_id, s.grouping_id
        from {table('session')} s
        where s.status = 'detected'
           or exists (select 1 from {table('thief_alert')} a
                      where a.method = 'TDS System' and a.detail = concat('Session ', s.id))
    """)
    session_ids = {int(row["id"]) for row in sessions}
    trigger_ids = {int(row[key]) for row in sessions for key in ("entry_trigger_id", "exit_trigger_id") if row.get(key) is not None}
    video_ids = set()
    for session in sessions:
        # Include every trigger in the same matched group, including multiple exits.
        items = _rows(db, f"""
            select distinct gi.trigger_id from {table('filter_grouping_item')} gi
            join {table('filter_grouping_item')} entry_item
              on entry_item.batch_id = gi.batch_id
             and cast(entry_item.group_key as binary) = cast(gi.group_key as binary)
            where gi.batch_id = :batch and entry_item.trigger_id = :entry
              and entry_item.role = 'entry'
        """, {"batch": session.get("grouping_id"), "entry": session.get("entry_trigger_id")})
        trigger_ids.update(int(row["trigger_id"]) for row in items)
        links = _rows(db, f"select video_asset_id from {table('session_video_asset')} where session_id=:id", {"id": session["id"]})
        video_ids.update(int(row["video_asset_id"]) for row in links)
    # Recent and in-flight files must also protect aliases under an old owner.
    recent = _rows(db, f"""
        select distinct va.id, va.trigger_id, sva.session_id from {table('video_asset')} va
        left join {table('session_video_asset')} sva on sva.video_asset_id = va.id
        where va.created_at >= :cutoff or va.retrieved_at >= :cutoff
           or va.status in ('retrieving', 'processing')
    """, {"cutoff": cutoff})
    video_ids.update(int(row["id"]) for row in recent)
    trigger_ids.update(int(row["trigger_id"]) for row in recent if row.get("trigger_id") is not None)
    session_ids.update(int(row["session_id"]) for row in recent if row.get("session_id") is not None)
    recent_frames = _rows(db, f"""
        select distinct fa.trigger_id from {table('trigger_frame_asset')} fa
        left join {table('trigger_frame')} f on f.frame_asset_id = fa.id
        where fa.created_at >= :cutoff or f.created_at >= :cutoff
           or fa.status in ('retrieving', 'processing')
    """, {"cutoff": cutoff})
    trigger_ids.update(int(row["trigger_id"]) for row in recent_frames)
    active = _rows(db, f"select session_id, trigger_id, runner_payload from {table('script_run')} where status in ('pending','running')")
    session_ids.update(int(row["session_id"]) for row in active if row.get("session_id") is not None)
    trigger_ids.update(int(row["trigger_id"]) for row in active if row.get("trigger_id") is not None)
    for run in active:
        payload = _json(run.get("runner_payload"))
        if isinstance(payload, dict):
            for key in ("trigger_ids", "entry_trigger_ids", "exit_trigger_ids"):
                values = payload.get(key)
                if isinstance(values, list):
                    trigger_ids.update(int(value) for value in values)
    grouping = _rows(db, f"""
        select distinct gi.trigger_id from {table('filter_grouping_item')} gi
        join {table('filter_grouping_batch')} b on b.id=gi.batch_id
        where b.status in ('pending','dispatching','running')
    """)
    trigger_ids.update(int(row["trigger_id"]) for row in grouping)
    return session_ids, trigger_ids, video_ids


def _owner_protected(value, sessions, triggers):
    for segment in str(value).replace("\\", "/").split("/"):
        kind, _, number = segment.partition("_")
        if number.isdigit() and ((kind == "session" and int(number) in sessions) or (kind == "trigger" and int(number) in triggers)):
            return True
    return False


def run_cleanup(db, now=None, dry_run=False):
    now = now or datetime.now(timezone.utc)
    cutoff_aware = month_ago(now.astimezone(ZONE)).astimezone(timezone.utc)
    cutoff = cutoff_aware.astimezone(timezone.utc).replace(tzinfo=None)
    table = repositories._table
    sessions, triggers, protected_videos = protected_owners(db, cutoff)
    stats = {"cutoff_utc": cutoff.isoformat(), "dry_run": dry_run, "files_deleted": 0, "objects_deleted": 0, "videos_expired": 0, "frames_expired": 0, "errors": 0}
    # Protect shared URLs/paths even if a different row owns the same object.
    protected_local, protected_remote = set(), set()
    videos = _rows(db, f"select * from {table('video_asset')} where status <> 'deleted'")
    links = _rows(db, f"select session_id, video_asset_id from {table('session_video_asset')}")
    session_links = {}
    for link in links:
        session_links.setdefault(int(link["video_asset_id"]), set()).add(int(link["session_id"]))
    changed = True
    while changed:
        before = len(protected_videos), len(sessions), len(triggers)
        for video in videos:
            if int(video["id"]) in protected_videos or video.get("trigger_id") in triggers or session_links.get(int(video["id"]), set()) & sessions:
                protected_videos.add(int(video["id"]))
                if video.get("trigger_id") is not None:
                    triggers.add(int(video["trigger_id"]))
                sessions.update(session_links.get(int(video["id"]), set()))
        changed = before != (len(protected_videos), len(sessions), len(triggers))
    for video in videos:
        if int(video["id"]) in protected_videos or video.get("trigger_id") in triggers or session_links.get(int(video["id"]), set()) & sessions:
            protected_videos.add(int(video["id"]))
            if video.get("trigger_id") is not None:
                triggers.add(int(video["trigger_id"]))
            sessions.update(session_links.get(int(video["id"]), set()))
            local, remote = media_references([video.get("file_path"), video.get("video_url"), _json(video.get("metadata"))])
            protected_local.update(local)
            protected_remote.update(remote)
    if triggers:
        from sqlalchemy import bindparam
        statement = text(f"select image_url from {table('trigger_frame')} where trigger_id in :ids")
        frames = db.execute(statement.bindparams(bindparam("ids", expanding=True)), {"ids": sorted(triggers)}).mappings().all()
        for frame in frames:
            local, remote = media_references(frame.get("image_url"))
            protected_local.update(local)
            protected_remote.update(remote)

    client = _spaces_client() if is_spaces_configured() else None
    remote_deleted, local_deleted = set(), set()

    def delete_remote(key, known_info=None):
        if key in protected_remote or _owner_protected(key, sessions, triggers):
            return False
        try:
            info = known_info or client.head_object(Bucket=settings.spaces_bucket, Key=key)
        except Exception as exc:
            if getattr(exc, "response", {}).get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
                return True
            raise
        if info["LastModified"] >= cutoff_aware:
            return False
        if not dry_run:
            client.delete_object(Bucket=settings.spaces_bucket, Key=key)
        stats["objects_deleted"] += 1
        remote_deleted.add(key)
        return True

    def delete_local(path):
        if path in protected_local or _owner_protected(path, sessions, triggers) or _owner_protected(path.resolve(), sessions, triggers) or path.is_symlink():
            return False
        if not path.exists():
            return True
        if datetime.fromtimestamp(path.stat().st_mtime, timezone.utc) >= cutoff_aware:
            return False
        if not dry_run:
            path.unlink()
        stats["files_deleted"] += 1
        local_deleted.add(path)
        return True

    def remove_refs(values):
        local, remote = media_references(values)
        ok = True
        for path in local:
            ok = (path in local_deleted or delete_local(path)) and ok
        for key in remote:
            ok = (client is not None and (key in remote_deleted or delete_remote(key))) and ok
        return ok

    # Never run against a live database as part of tests; dry-run is available via CLI.
    for video in videos:
        if int(video["id"]) in protected_videos or video["created_at"] >= cutoff or video["status"] in {"retrieving", "processing"}:
            continue
        try:
            if remove_refs([video.get("file_path"), video.get("video_url"), _json(video.get("metadata"))]):
                if not dry_run:
                    db.execute(text(f"update {table('video_asset')} set status='deleted' where id=:id"), {"id": video["id"]})
                    db.commit()
                stats["videos_expired"] += 1
        except Exception:
            db.rollback()
            stats["errors"] += 1
            logger.exception("Could not expire video_asset_id=%s", video["id"])

    last_id = 0
    while True:
        frames = _rows(db, f"""
            select f.* from {table('trigger_frame')} f
            join {table('trigger_frame_asset')} fa on fa.id = f.frame_asset_id
            where f.id > :id and f.created_at < :cutoff and f.status <> 'deleted'
              and fa.status not in ('retrieving','processing')
            order by f.id limit 200
        """, {"id": last_id, "cutoff": cutoff})
        if not frames:
            break
        for frame in frames:
            last_id = int(frame["id"])
            if int(frame["trigger_id"]) in triggers:
                continue
            try:
                if remove_refs(frame.get("image_url")):
                    if not dry_run:
                        db.execute(text(f"update {table('trigger_frame')} set status='deleted' where id=:id"), {"id": frame["id"]})
                        db.commit()
                    stats["frames_expired"] += 1
            except Exception:
                db.rollback()
                stats["errors"] += 1
                logger.exception("Could not expire frame_id=%s", frame["id"])

    # Include older derived clips/crops under registered owners, even if only
    # their source video or original frame is referenced by a media row.
    if client:
        for entity in ("trigger", "session"):
            last_owner = 0
            while True:
                owners = _rows(db, f"select id, location_id from {table('trigger_event' if entity == 'trigger' else 'session')} where id > :id and created_at < :cutoff order by id limit 200", {"id": last_owner, "cutoff": cutoff})
                if not owners:
                    break
                for owner in owners:
                    last_owner = int(owner["id"])
                    prefix = build_spaces_object_key(f"location_{owner['location_id']}", f"{entity}_{owner['id']}") + "/"
                    if _owner_protected(prefix, sessions, triggers):
                        continue
                    try:
                        for page in client.get_paginator("list_objects_v2").paginate(Bucket=settings.spaces_bucket, Prefix=prefix):
                            for obj in page.get("Contents", []):
                                key = obj["Key"]
                                if key not in remote_deleted and Path(key).suffix.lower() in MEDIA_SUFFIXES:
                                    delete_remote(key, obj)
                    except Exception:
                        stats["errors"] += 1
                        logger.exception("Could not clean media prefix %s", prefix)

    # Frame snapshots also have local temporary copies which have no DB path.
    # Sweep media only, under explicitly named trigger/session directories.
    root = settings.video_storage_dir.resolve()
    for base in (root / "locations", root / "tmp_video" / "locations"):
        if not base.exists():
            continue
        for location in base.glob("location_*"):
            if location.is_symlink():
                continue
            for owner in location.iterdir():
                if not owner.is_dir() or owner.is_symlink() or not owner.name.startswith(("trigger_", "session_")) or _owner_protected(owner, sessions, triggers):
                    continue
                for path in owner.rglob("*"):
                    if path.is_file() and not path.is_symlink() and path.suffix.lower() in MEDIA_SUFFIXES and root in path.resolve().parents:
                        try:
                            if path not in local_deleted:
                                delete_local(path)
                        except OSError:
                            stats["errors"] += 1
                            logger.exception("Could not expire local media %s", path)
    if not dry_run:
        from sqlalchemy import bindparam
        statement = text(f"""
            update {table('trigger_frame_asset')} fa set status='deleted'
            where fa.created_at < :cutoff and fa.status not in ('retrieving','processing','deleted')
              and fa.trigger_id not in :protected_ids
              and not exists (select 1 from {table('trigger_frame')} f where f.frame_asset_id=fa.id and f.status <> 'deleted')
        """).bindparams(bindparam("protected_ids", expanding=True))
        db.execute(statement, {"cutoff": cutoff, "protected_ids": sorted(triggers) or [-1]})
        db.commit()
    logger.info("Media retention completed: %s", stats)
    return stats
