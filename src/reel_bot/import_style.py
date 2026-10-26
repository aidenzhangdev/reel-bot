"""Build data/style.md examples from an Instagram data export (Download your information → Messages, JSON or HTML).

    uv run import-style ~/Downloads/instagram-export.zip                 # list your group chats
    uv run import-style ~/Downloads/instagram-export.zip --chat "the boys"

Only messages *you* sent are kept, plus the caption of any Reel a friend shared right before you replied.
Friends' own messages and names never make it into style.md.
"""

import argparse
import html
import json
import re
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import STYLE_FILE

START = "<!-- imported-from-instagram:start -->"
END = "<!-- imported-from-instagram:end -->"
MAX_MESSAGES = 400
MAX_REACTIONS = 80
MAX_LEN = 280
REACTION_WINDOW = 15 * 60  # a reply within 15 min of a shared Reel counts as a reaction to it

# Instagram's auto-generated lines, not things you actually typed.
SYSTEM_TEXT = re.compile(
    r"sent an attachment|^reacted |^liked a message|changed the group|named the group|added .+ to the group"
    r"|left the group|started a (video|audio) chat|^you missed a|sent a voice message|unsent a message|shared a story",
    re.IGNORECASE,
)


@dataclass
class Chat:
    title: str
    participants: list[str]
    messages: list[dict]  # oldest first


def fix_text(s: str) -> str:
    """Instagram's JSON export writes UTF-8 text as if it were latin-1 ("ðŸ˜­" instead of "😭"). Undo that."""
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


FILE_PATTERN = re.compile(r"messages/inbox/([^/]+)/message_(\d+)\.(json|html)$")


def _read_files(export: Path) -> list[tuple[str, bytes]]:
    """(path, bytes) for every message_N.json/.html in an export zip or unzipped folder."""
    if export.is_file() and zipfile.is_zipfile(export):
        with zipfile.ZipFile(export) as z:
            return [(n, z.read(n)) for n in z.namelist() if FILE_PATTERN.search(n)]
    return [(p.as_posix(), p.read_bytes()) for p in export.rglob("message_*") if FILE_PATTERN.search(p.as_posix())]


def _parse_json(raw: bytes) -> tuple[str, list[dict]]:
    data = json.loads(raw)
    messages = []
    for m in data.get("messages", []):
        share = m.get("share") or {}
        messages.append({
            "sender": fix_text(m.get("sender_name", "")),
            "ts": m.get("timestamp_ms", 0) / 1000,
            "text": fix_text(m.get("content", "")).strip(),
            "link": share.get("link", ""),
            "caption": fix_text(share.get("share_text", "")),
            "owner": fix_text(share.get("original_content_owner", "")),
        })
    return fix_text(data.get("title", "")), messages


# The HTML export: one <div class="... _a6-g ..."> block per message, with the sender in an h2, the text and any
# shared post inside "_a6-p", and the time in "_a6-o". A shared post is caption, then owner, then the link.
HTML_BLOCK = '<div class="pam _3-95 _2ph- _a6-g uiBoxWhite noborder">'
HTML_SENDER = re.compile(r'<h2 class="[^"]*_2pim[^"]*">(.*?)</h2>', re.S)
HTML_TEXT = re.compile(r'_a6-p"><div><div></div><div>(.*?)</div>', re.S)
HTML_SHARE = re.compile(r'<div><div><div>([^<]*)</div><div>([^<]*)</div><div><a target="_blank" href="([^"]+)"')
HTML_TIME = re.compile(r'<div class="_3-94 _a6-o">([^<]*)</div>')
HTML_TITLE = re.compile(r"<title>(.*?)</title>", re.S)


def _clean_html(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", s)).strip()


def _parse_html_time(s: str) -> float:
    try:
        return datetime.strptime(s.strip(), "%b %d, %Y %I:%M %p").timestamp()
    except ValueError:
        return 0.0


def _parse_html(raw: bytes) -> tuple[str, list[dict]]:
    page = raw.decode("utf-8")
    title = HTML_TITLE.search(page)
    messages = []
    for block in page.split(HTML_BLOCK)[1:]:
        sender, time_ = HTML_SENDER.search(block), HTML_TIME.search(block)
        if not (sender and time_):
            continue  # page header, not a message
        text, share = HTML_TEXT.search(block), HTML_SHARE.search(block)
        messages.append({
            "sender": _clean_html(sender.group(1)),
            "ts": _parse_html_time(time_.group(1)),
            "text": _clean_html(text.group(1)) if text else "",
            "link": html.unescape(share.group(3)) if share else "",
            "caption": html.unescape(share.group(1)) if share else "",
            "owner": html.unescape(share.group(2)) if share else "",
        })
    return (_clean_html(title.group(1)) if title else ""), messages


def load_chats(export: Path) -> list[Chat]:
    files = _read_files(export)
    if not files:
        sys.exit(f"No messages found in {export}. Is this the zip from Instagram's \"Download your information\"?")
    by_thread: dict[str, dict[int, tuple[str, list[dict]]]] = {}
    for path, raw in files:
        thread, number, kind = FILE_PATTERN.search(path).groups()
        by_thread.setdefault(thread, {})[int(number)] = (_parse_json if kind == "json" else _parse_html)(raw)
    chats = []
    for thread, parts in by_thread.items():
        # Each file lists newest first, and higher-numbered files are older: walk them backwards for oldest first.
        messages = [m for n in sorted(parts, reverse=True) for m in reversed(parts[n][1])]
        title = parts[min(parts)][0] or thread
        chats.append(Chat(title, sorted({m["sender"] for m in messages if m["sender"]}), messages))
    return chats


def guess_me(chats: list[Chat]) -> str:
    """You're in every chat of your own export, so you've sent messages in more chats than anyone else."""
    sent_in = Counter(name for c in chats for name in c.participants)
    return sent_in.most_common(1)[0][0]


def _is_typed_message(m: dict) -> bool:
    text = m["text"]
    return bool(text) and not SYSTEM_TEXT.search(text) and not text.startswith("http") and len(text) <= MAX_LEN


def _shared_reel(m: dict) -> str | None:
    """A short description of a shared Reel/post, or None if this message isn't one."""
    if not re.search(r"instagram\.com/(reels?|p)/", m["link"]):
        return None
    caption = " ".join(m["caption"].split())[:120]
    return (f"@{m['owner']}: " if m["owner"] else "") + (caption or "(no caption)")


def extract(chats: list[Chat], me: str) -> tuple[list[str], list[tuple[str, str]]]:
    """Your messages (newest first, deduplicated) and (shared Reel, your reaction) pairs."""
    mine: list[str] = []
    reactions: list[tuple[str, str]] = []
    for chat in chats:
        pending: tuple[str, int] | None = None  # (reel description, timestamp) shared by a friend
        for m in chat.messages:
            sender, ts = m["sender"], m["ts"]
            if sender != me:
                reel = _shared_reel(m)
                if reel:
                    pending = (reel, ts)
                continue
            if not _is_typed_message(m):
                continue
            text = m["text"]
            mine.append(text)
            if pending and ts - pending[1] <= REACTION_WINDOW:
                reactions.append((pending[0], text))
            pending = None
    mine = list(dict.fromkeys(reversed(mine)))  # newest first, no duplicates
    reactions = list(dict.fromkeys(reversed(reactions)))
    return mine[:MAX_MESSAGES], reactions[:MAX_REACTIONS]


def render(mine: list[str], reactions: list[tuple[str, str]], source: str) -> str:
    lines = [START, f"## More real messages I've sent (imported from {source})"]
    lines += [f"- {t}" for t in mine]
    if reactions:
        lines += ["", "## More of my real reactions to Reels friends sent"]
        lines += [f"- friend sent: {reel} → me: {reply}" for reel, reply in reactions]
    lines.append(END)
    return "\n".join(lines)


def merge_into_style(existing: str, block: str) -> str:
    """Replace a previous import, or append one, leaving the hand-written parts alone."""
    if START in existing and END in existing:
        before, rest = existing.split(START, 1)
        after = rest.split(END, 1)[1]
        return before + block + after
    return existing.rstrip() + "\n\n" + block + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Add your real messages from an Instagram export to data/style.md.")
    p.add_argument("export", type=Path, help="the export .zip from Instagram, or the unzipped folder")
    p.add_argument("--chat", action="append", help="chat name to import from (repeatable); omit to list chats")
    p.add_argument("--me", help="your display name in the export (auto-detected if omitted)")
    args = p.parse_args()

    chats = load_chats(args.export.expanduser())
    me = args.me or guess_me(chats)

    if not args.chat:
        print(f"Found {len(chats)} chats. You appear to be \"{me}\" (use --me to change).\n")
        print(f"{'your msgs':>9}  chat")
        for c in sorted(chats, key=lambda c: -sum(m["sender"] == me for m in c.messages))[:30]:
            n = sum(m["sender"] == me for m in c.messages)
            kind = "group" if len(c.participants) > 2 else "dm"
            print(f"{n:>9}  {c.title}  ({kind})")
        print('\nRun again with --chat "<name>" to import one (repeat --chat for several).')
        return

    wanted = {name.lower() for name in args.chat}
    picked = [c for c in chats if c.title.lower() in wanted]
    missing = wanted - {c.title.lower() for c in picked}
    if missing:
        sys.exit(f"No chat named {', '.join(sorted(missing))}. Run without --chat to see the names.")

    mine, reactions = extract(picked, me)
    if not mine:
        sys.exit(f'Found no messages from "{me}" in that chat. Check the name with --me.')
    source = ", ".join(c.title for c in picked)
    STYLE_FILE.write_text(merge_into_style(STYLE_FILE.read_text(), render(mine, reactions, source)))
    print(f"Added {len(mine)} of your messages and {len(reactions)} Reel reactions to {STYLE_FILE}.")
    print("Your hand-written notes at the top were kept. Run again any time to refresh the import.")


if __name__ == "__main__":
    main()
