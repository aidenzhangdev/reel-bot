import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
TMP_DIR = ROOT / "tmp"
SESSION_FILE = ROOT / "session.json"
STATE_DB = DATA_DIR / "state.db"
LOCK_FILE = DATA_DIR / "bot.lock"
STYLE_FILE = DATA_DIR / "style.md"


@dataclass(frozen=True)
class Settings:
    ig_username: str
    ig_password: str
    my_name: str
    allowed_users: frozenset[str]  # lowercase usernames; "*" allows anyone
    dry_run: bool
    poll_seconds: int
    max_replies_per_hour: int
    text_trigger: str  # reply to texts whose first word is this; empty turns text replies off


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    allowed = {u.strip().lstrip("@").lower() for u in os.getenv("ALLOWED_USERS", "").split(",") if u.strip()}
    return Settings(
        ig_username=os.getenv("IG_USERNAME", ""),
        ig_password=os.getenv("IG_PASSWORD", ""),
        my_name=os.getenv("MY_NAME", "me"),
        allowed_users=frozenset(allowed),
        dry_run=os.getenv("DRY_RUN", "1") != "0",
        poll_seconds=max(10, int(os.getenv("POLL_SECONDS", "10"))),
        max_replies_per_hour=int(os.getenv("MAX_REPLIES_PER_HOUR", "20")),
        text_trigger=os.getenv("TEXT_TRIGGER", "bot").strip(),
    )
