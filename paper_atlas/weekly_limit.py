"""Persistent, atomic app-wide budget for outbound AI requests."""
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

WEEKLY_LIMIT = 1000
TIMEZONE = "America/Chicago"


class WeeklyLimitReached(ValueError):
    def __init__(self, quota):
        self.quota = quota
        super().__init__(f"Weekly AI limit reached ({quota['limit']:,} requests). "
                         f"Resets {quota['resets_at']} ({TIMEZONE}). "
                         "Temporarily cached interpretations and results remain available until evicted or restarted.")


class WeeklyLimit:
    def __init__(self, path, clock=None):
        self.path = Path(path)
        self.clock = clock or (lambda: datetime.now(ZoneInfo(TIMEZONE)))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS ai_weekly_usage (week TEXT PRIMARY KEY, used INTEGER NOT NULL CHECK(used >= 0))")

    def connect(self):
        return sqlite3.connect(self.path, timeout=20)

    def _window(self):
        now = self.clock().astimezone(ZoneInfo(TIMEZONE))
        start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        return start.date().isoformat(), (start + timedelta(days=7)).isoformat()

    def _status(self, db, week, reset):
        row = db.execute("SELECT used FROM ai_weekly_usage WHERE week=?", (week,)).fetchone()
        used = row[0] if row else 0
        return {"limit": WEEKLY_LIMIT, "used": used, "remaining": max(0, WEEKLY_LIMIT-used),
                "week_start": week, "resets_at": reset, "timezone": TIMEZONE,
                "unit": "outbound_ai_requests"}

    def status(self):
        with self.connect() as db:
            return self._status(db, *self._window())

    def reserve(self):
        """Commit before network I/O. Failures/retries count; crashes cannot refund."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            week, reset = self._window()
            quota = self._status(db, week, reset)
            if quota["remaining"] == 0:
                raise WeeklyLimitReached(quota)
            db.execute("INSERT INTO ai_weekly_usage VALUES (?,1) ON CONFLICT(week) DO UPDATE SET used=used+1", (week,))
            return self._status(db, week, reset)
