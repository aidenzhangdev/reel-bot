"""Everything that talks to Instagram (via instagrapi's unofficial private API)."""

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from instagrapi import Client
from instagrapi.exceptions import (
    ChallengeRequired,
    FeedbackRequired,
    LoginRequired,
    PleaseWaitFewMinutes,
    RateLimitError,
)
from instagrapi.extractors import extract_media_v1
from instagrapi.types import DirectMessage

from .reel import ReelContent

# Errors that mean Instagram is suspicious of the account. Stop instead of retrying.
STOP_ERRORS = (ChallengeRequired, FeedbackRequired, LoginRequired, PleaseWaitFewMinutes, RateLimitError)

REEL_XMA_TYPES = {"xma_clip", "xma_media_share", "xma_reel_share"}


CONTEXT_MESSAGES = 6  # earlier texts in the chat to show Gemini when answering a text
TOP_COMMENTS = 5  # comments on a Reel to show Gemini, most liked first
MAX_COMMENT_LEN = 150
MIN_VIDEO_HEIGHT = 480  # smallest Reel resolution worth giving Gemini; it shrinks frames anyway


@dataclass
class Incoming:
    kind: str  # "reel" or "text"
    message_id: str
    thread_id: str
    sender: str  # username
    sent_at: float
    media_pk: str | None = None  # reels
    media_url: str | None = None  # reels
    note: str = ""  # reels: text the sender wrote right after sharing it
    text: str = ""  # texts: the message itself
    context: list[tuple[str, str]] = field(default_factory=list)  # texts: (who, what) before it, oldest first
    message: DirectMessage | None = None  # the DM itself, so the reply can quote it


def is_trigger(text: str | None, word: str) -> bool:
    """True if the message's first word is the trigger word, in any case ("Bot, ...", "BOT what")."""
    return bool(word and text) and re.match(rf"\W*{re.escape(word)}\b", text, re.IGNORECASE) is not None


def reel_source(msg: DirectMessage) -> tuple[str | None, str | None]:
    """Find the shared Reel in a DM. Returns (media_pk, url); both None if it isn't a Reel/video share."""
    if msg.clip is not None:
        return str(msg.clip.pk), None
    if msg.media_share is not None and msg.media_share.media_type == 2:  # 2 = video
        return str(msg.media_share.pk), None
    if msg.item_type in REEL_XMA_TYPES and msg.xma_share is not None:
        url = msg.xma_share.video_url
        if url.startswith("http") and ("/reel" in url or "/p/" in url):
            return None, url
    return None, None


def smallest_video(versions: list[dict]) -> str:
    """URL of the smallest video version at least MIN_VIDEO_HEIGHT tall, or the smallest one if none are."""
    by_size = sorted(versions, key=lambda v: v.get("height", 0) * v.get("width", 0))
    big_enough = [v for v in by_size if v.get("height", 0) >= MIN_VIDEO_HEIGHT]
    return (big_enough or by_size)[0]["url"]


class Instagram:
    def __init__(self, username: str, password: str, session_file: Path):
        self.cl = Client()
        self.cl.delay_range = [0.5, 1.5]  # random pause between API calls, like a person
        if session_file.exists():
            self.cl.load_settings(session_file)
        self.cl.login(username, password)
        self.cl.dump_settings(session_file)
        self.my_id = str(self.cl.user_id)

    def fetch_new(self, since_ts: float, allowed: frozenset[str], is_handled, trigger: str) -> list[Incoming]:
        """New Reels, plus texts starting with the trigger word, from allowed friends, oldest first."""
        found: list[Incoming] = []
        for thread in self.cl.direct_threads(amount=20, thread_message_limit=10):
            usernames = {str(u.pk): u.username.lower() for u in thread.users}
            msgs = thread.messages  # newest first
            used_as_note: set[str] = set()
            for i, msg in enumerate(msgs):
                if msg.timestamp.timestamp() < since_ts or str(msg.user_id) == self.my_id or is_handled(msg.id):
                    continue
                sender = usernames.get(str(msg.user_id), "")
                if "*" not in allowed and sender not in allowed:
                    continue
                pk, url = reel_source(msg)
                if pk or url:
                    # Text the same friend sent after the Reel (newer messages are earlier in the list).
                    after = [m for m in reversed(msgs[:i]) if m.text and str(m.user_id) == str(msg.user_id)]
                    used_as_note.update(m.id for m in after)
                    found.append(Incoming("reel", msg.id, thread.id, sender, msg.timestamp.timestamp(),
                                          media_pk=pk, media_url=url, note=" / ".join(m.text for m in after),
                                          message=msg))
                elif is_trigger(msg.text, trigger):
                    earlier = [m for m in msgs[i + 1 : i + 1 + CONTEXT_MESSAGES] if m.text]
                    context = [("you" if str(m.user_id) == self.my_id else usernames.get(str(m.user_id), "someone"), m.text)
                               for m in reversed(earlier)]
                    found.append(Incoming("text", msg.id, thread.id, sender, msg.timestamp.timestamp(),
                                          text=msg.text, context=context, message=msg))
            # A trigger text sent right after a Reel is already answered by the Reel reply.
            found = [f for f in found if not (f.kind == "text" and f.message_id in used_as_note)]
        return sorted(found, key=lambda f: f.sent_at)

    def download(self, pk: str | None, url: str | None, folder: Path) -> ReelContent:
        """Download a Reel, with its caption, author and top comments."""
        folder.mkdir(parents=True, exist_ok=True)
        if pk is None:
            pk = self.cl.media_pk_from_url(url)
        info, video_url = self._media_and_small_video(pk)
        if not video_url:
            raise ValueError(f"media {pk} has no video")
        # The video comes from a plain HTTP download, separate from the API session, so fetch comments meanwhile.
        with ThreadPoolExecutor(1) as pool:
            comments = pool.submit(self.top_comments, info.id)
            video = self.cl.video_download_by_url(video_url, folder=folder)
            return ReelContent(Path(video), info.caption_text or "", info.user.username or "", comments.result())

    def _media_and_small_video(self, pk: str):
        """The Reel's info plus the URL of a small version of its video.

        instagrapi's media_info always picks the biggest version (often 1080p, ~40MB), which is slow to download and
        too big to send to Gemini inline. Same single request, but we choose the version ourselves.
        """
        try:
            item = self.cl.private_request(f"media/{pk}/info/")["items"][0]
        except STOP_ERRORS:
            raise
        except Exception:
            item = None
        if not item or not item.get("video_versions"):
            info = self.cl.media_info(pk)
            return info, str(info.video_url or "")
        return extract_media_v1(item), smallest_video(item["video_versions"])

    def top_comments(self, media_id: str) -> list[str]:
        """The most liked comments on a Reel, from one request (the first page); [] if comments are off or it fails.

        `media_id` is the full "<pk>_<owner pk>" id. instagrapi's media_comments_chunk would look that up with an
        extra media_info call and then keep paging, which took ~7s.
        """
        try:
            raw = self.cl.private_request(f"media/{media_id}/comments/", params={"can_support_threading": "true"})
        except STOP_ERRORS:
            raise
        except Exception:
            return []
        top = [
            c for c in raw.get("comments") or []
            if (c.get("text") or "").strip() and not (c.get("parent_comment_id") or c.get("replied_to_comment_id"))
        ]
        top.sort(key=lambda c: c.get("comment_like_count") or 0, reverse=True)
        return [" ".join(c["text"].split())[:MAX_COMMENT_LEN] for c in top[:TOP_COMMENTS]]

    def send(self, thread_id: str, text: str, reply_to: DirectMessage | None = None) -> None:
        """Send a text, quoting `reply_to` like swiping to reply in the app."""
        self.cl.direct_send(text, thread_ids=[int(thread_id)], reply_to_message=reply_to)

    def react(self, thread_id: str, message_id: str, emoji: str) -> None:
        """Put an emoji reaction on a message, like double-tapping it in the app."""
        if not self.cl.direct_send_reaction(int(thread_id), int(message_id), emoji):
            raise RuntimeError("Instagram didn't accept the reaction")
