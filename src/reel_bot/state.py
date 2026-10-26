"""SQLite record of which DMs the bot has already handled."""

import sqlite3
import time
from pathlib import Path


class State:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS handled (
                message_id TEXT PRIMARY KEY,
                thread_id TEXT,
                handled_at REAL,
                outcome TEXT,  -- sent | reacted | dry_run | declined | error
                reply TEXT
            );
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            """
        )

    def started_at(self) -> float:
        """When the bot first ran. Reels sent before this are ignored, so it doesn't reply to old DMs."""
        row = self.db.execute("SELECT value FROM meta WHERE key = 'started_at'").fetchone()
        if row:
            return float(row[0])
        now = time.time()
        with self.db:
            self.db.execute("INSERT INTO meta VALUES ('started_at', ?)", (str(now),))
        return now

    def is_handled(self, message_id: str) -> bool:
        return self.db.execute("SELECT 1 FROM handled WHERE message_id = ?", (message_id,)).fetchone() is not None

    def mark_handled(self, message_id: str, thread_id: str, outcome: str, reply: str = "") -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO handled VALUES (?, ?, ?, ?, ?)",
                (message_id, thread_id, time.time(), outcome, reply),
            )

    def recent_replies(self, limit: int = 30) -> list[str]:
        """Individual messages the bot sent (or would have, in dry run) most recently, newest first."""
        rows = self.db.execute(
            "SELECT reply FROM handled WHERE outcome IN ('sent', 'dry_run') AND reply != ''"
            " ORDER BY handled_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return list(dict.fromkeys(line for (reply,) in rows for line in reply.splitlines() if line.strip()))

    def sent_in_last_hour(self) -> int:
        cutoff = time.time() - 3600
        return self.db.execute(
            "SELECT COUNT(*) FROM handled WHERE outcome IN ('sent', 'reacted') AND handled_at > ?", (cutoff,)
        ).fetchone()[0]
