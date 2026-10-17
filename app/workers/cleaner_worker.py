from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone

from sqlalchemy import text

from .. import repositories
from ..db import TransactionalSessionLocal
from ..services.media_retention import run_cleanup, scheduled_date

logger = logging.getLogger(__name__)
SETTING_KEY = "media_cleaner_last_success"


def tick(now=None, *, force=False, dry_run=False):
    now = now or datetime.now(timezone.utc)
    day = scheduled_date(now)
    if day is None and not force:
        return
    db = TransactionalSessionLocal()
    with db.get_bind().connect() as connection:
        acquired = connection.execute(text("select get_lock('tds_media_cleaner', 0)")).scalar() == 1
        try:
            if not acquired:
                return
            previous = repositories.get_app_setting(db, SETTING_KEY)
            if previous and not force and json.loads(previous).get("date") == day:
                return
            result = run_cleanup(db, now=now, dry_run=dry_run)
            if not dry_run and not result["errors"]:
                repositories.set_app_setting(db, SETTING_KEY, json.dumps({"date": day, "result": result}))
            return result
        finally:
            if acquired:
                connection.execute(text("select release_lock('tds_media_cleaner')"))
            db.close()


def main():
    parser = argparse.ArgumentParser(description="Remove non-theft media older than one calendar month, daily at 04:00 Asia/Kuala_Lumpur.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.once or args.dry_run:
        print(json.dumps(tick(force=True, dry_run=args.dry_run), default=str))
        return
    while True:
        try:
            tick()
        except Exception:
            logger.exception("Daily media cleaner failed; will retry")
        time.sleep(60)


if __name__ == "__main__":
    main()
