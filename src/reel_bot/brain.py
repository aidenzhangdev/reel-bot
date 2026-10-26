"""Ask Gemini to react to a Reel, or answer a text, the way I would text back."""

import os
import re
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from google import genai
from google.genai import errors, types

from .reel import ReelContent

DEFAULT_MODEL = "gemini-flash-latest"
BACKUP_MODEL = "gemini-flash-lite-latest"  # tried when the main model is overloaded or rate-limited
MAX_BURST = 3  # max separate messages per reply
MAX_ATTEMPTS = 3  # tries to get a reply that isn't a repeat or a copied example
UPLOAD_TIMEOUT = 60  # seconds to wait for Gemini to finish processing the video
INLINE_LIMIT = 18 * 1024 * 1024  # videos up to this size go inside the request (limit 20MB); bigger ones are uploaded
STYLE_PLACEHOLDER = "PASTE YOUR MESSAGES BELOW"
# What Gemini writes instead of a text when a Reel isn't worth a reply, and the emoji reaction it becomes.
REACTIONS = {"[heart]": "❤️", "[thumbs up]": "👍"}
EMOJI = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]\ufe0f?")
EMOJI_WINDOW = 8  # recent messages to check for an emoji the bot keeps leaning on
EMOJI_LIMIT = 2  # an emoji used in more than this many of them is left out of the next reply

SYSTEM = """You are texting as {name}. {situation}

How {name} texts is described below, with real messages they've sent. Learn the style from them: message length, capitalization, punctuation, slang, emoji use, and humor. These are examples of HOW they talk, not a list of replies to pick from.

<style>
{style}
</style>

Rules:
{task_rule}
- Write a new message. Never send one of the example messages word for word.
- Reply only with the message text. No quotes, no explanations, no "As {name}...".
- If {name} usually sends a few short texts in a row, put each one on its own line (at most {max_burst} lines). Otherwise reply on one line.
- Emoji: use at most one, only when it adds something, and vary them like {name} does. Most messages need none.
- Never say you're an AI."""

REEL_SITUATION = "A friend just sent {name} an Instagram Reel in DMs, and you reply exactly the way {name} would."
REEL_RULE = (
    "- Watch the whole video, sound included, so you get the joke. Then text back the way {name} would to a friend: "
    "a joke, a roast (of the friend or someone in it), a hot take, a callback, or what it reminds you of.\n"
    "- Don't describe or recap what happens in the video. The friend already watched it, so \"bro is doing X\" or "
    "\"he ate that in one bite\" is the boring reply. Think of a few angles and send the funniest: make it about the "
    "friend, someone you both know, yourself, or how absurd the whole thing is.\n"
    "- If there's genuinely nothing good to say and the friend didn't ask you anything, don't force it. Reply with "
    "exactly [heart] if the Reel is funny or good, or [thumbs up] if it's bland or has no real content. That becomes "
    "an emoji reaction on the Reel instead of a text. Never text back just an emoji.\n"
    "- The top comments (if any) are only there so you get the context: who this is, the reference, why people find "
    "it funny. Don't repeat, reword, or reuse the punchline of any comment. Use them as a jumping-off point and come "
    "up with your own take that none of the comments made."
)
TEXT_SITUATION = (
    "A friend just texted {name} in DMs, and you reply exactly the way {name} would. "
    'They start the message with "{trigger}" to get your attention; that word is just how they call you.'
)
TEXT_RULE = (
    "- Actually respond to what they said: answer the question, react to the news, roast them, whatever {name} "
    "would do. Earlier messages are only there for context."
)


def system_prompt(situation: str, task_rule: str, my_name: str, style: str, trigger: str = "") -> str:
    return SYSTEM.format(
        name=my_name,
        situation=situation.format(name=my_name, trigger=trigger),
        task_rule=task_rule.format(name=my_name),
        style=style,
        max_burst=MAX_BURST,
    )


@dataclass(frozen=True)
class Reaction:
    """React to the Reel with this emoji instead of texting back."""

    emoji: str


def blocks_repeats() -> bool:
    """BLOCK_REPEATS=0 in .env lets replies repeat old ones, copy style.md examples, or echo the Reel's comments."""
    return os.getenv("BLOCK_REPEATS", "1").strip() != "0"


class TryLater(Exception):
    """Gemini is overloaded or the free-tier limit was hit. The Reel should be retried on a later poll."""


def _is_temporary(e: errors.APIError) -> bool:
    return e.code == 429 or (e.code or 0) >= 500


@cache
def _client() -> genai.Client:
    # One quick retry, then let _generate switch to the backup model. The SDK default (5 tries, backing off
    # 1-8s each) can spend 15+ seconds on an overloaded model.
    retry = types.HttpRetryOptions(attempts=2, initial_delay=1, max_delay=2)
    return genai.Client(http_options=types.HttpOptions(retry_options=retry))  # reads GEMINI_API_KEY


def load_style(path: Path) -> str:
    style = path.read_text().strip()
    if any(line.strip() == STYLE_PLACEHOLDER for line in style.splitlines()):
        raise SystemExit(f"Fill in {path} with how you text before running the bot (see the instructions inside it).")
    return style


def split_reply(text: str) -> list[str]:
    """Clean the model's text into 1..MAX_BURST messages."""
    lines = [ln.strip().strip('"').strip() for ln in text.strip().splitlines()]
    return [ln for ln in lines if ln][:MAX_BURST]


def _norm(line: str) -> str:
    """Compare lines by their words only, so "cooked beyond repair 😭" and "Cooked beyond repair!" count as the same."""
    return " ".join(re.sub(r"[^\w\s]", "", line.lower()).split())


def drop_repeats(lines: list[str], recent: Sequence[str]) -> list[str]:
    """Remove lines the bot already sent recently. Emoji-only lines (no words) may repeat."""
    seen = {_norm(r) for r in recent} - {""}
    return [ln for ln in lines if _norm(ln) not in seen]


def overused_emoji(recent: Sequence[str]) -> set[str]:
    """Emoji in more than EMOJI_LIMIT of the last EMOJI_WINDOW messages sent, e.g. 😭 at the end of everything."""
    counts = Counter(e for line in recent[:EMOJI_WINDOW] for e in {m.rstrip("\ufe0f") for m in EMOJI.findall(line)})
    return {e for e, n in counts.items() if n > EMOJI_LIMIT}


def strip_emoji(lines: list[str], emoji: set[str]) -> list[str]:
    """Remove the given emoji from each line, dropping lines that were nothing but those emoji."""
    if not emoji:
        return lines
    out = []
    for line in lines:
        cleaned = EMOJI.sub(lambda m: "" if m.group().rstrip("\ufe0f") in emoji else m.group(), line)
        cleaned = " ".join(cleaned.split())
        if cleaned:
            out.append(cleaned)
    return out


def as_reaction(lines: list[str]) -> Reaction | None:
    """The Reaction if Gemini answered with one of the REACTIONS tokens instead of a text.

    A reply that's only emoji (like "🍔") is also a reaction: a heart on the Reel says the same thing, less awkwardly.
    """
    if len(lines) == 1 and lines[0].lower() in REACTIONS:
        return Reaction(REACTIONS[lines[0].lower()])
    if lines and not any(_norm(line) for line in lines):
        return Reaction(REACTIONS["[heart]"])
    return None


# Words too common to count as borrowing when a reply and a comment share them.
FILLER = set(
    "a an the is are was were be been am i im u you ur he she they it its this that these those and or but so to of "
    "in on at for with my me his her their we our not no just like bro bruh lol lmao lmfao fr ngl"
    .split()
)
BORROWED_RUN = 4  # words in a row shared with a comment
BORROWED_SHARE = 0.7  # share of a reply's non-filler words found in one comment
BORROWED_MIN_WORDS = 3  # non-filler words a reply needs before the share check applies


def borrows_from(line: str, comments: Sequence[str]) -> bool:
    """True if a reply line copies or rewords a comment, rather than making its own joke about the Reel."""
    words = _norm(line).split()
    content = [w for w in words if w not in FILLER]
    for comment in comments:
        theirs = _norm(comment).split()
        runs = {tuple(theirs[i : i + BORROWED_RUN]) for i in range(len(theirs) - BORROWED_RUN + 1)}
        if any(tuple(words[i : i + BORROWED_RUN]) in runs for i in range(len(words) - BORROWED_RUN + 1)):
            return True
        if len(content) >= BORROWED_MIN_WORDS and sum(w in set(theirs) for w in content) / len(content) >= BORROWED_SHARE:
            return True
    return False


def example_lines(style: str, min_words: int = 4) -> list[str]:
    """Your example messages from style.md that are long enough that sending one verbatim would be obvious copying."""
    lines = []
    for raw in style.splitlines():
        line = raw.strip()
        if not line.startswith("- "):
            continue
        line = line[2:].split(" → me: ", 1)[-1]  # for reaction examples, only your reply counts
        if len(_norm(line).split()) >= min_words:
            lines.append(line)
    return lines


def build_prompt(reel: ReelContent, friend: str, friend_note: str = "", recent: Sequence[str] = ()) -> str:
    parts = [f"Friend who sent it: {friend}"]
    if reel.author:
        parts.append(f"Reel posted by: @{reel.author}")
    parts.append(f"Caption: {reel.caption.strip() or '(none)'}")
    if reel.comments:
        parts.append("Top comments (background only, don't repeat or reword these):")
        parts.extend(f"- {c}" for c in reel.comments)
    if friend_note:
        parts.append(f"What {friend} said with it: {friend_note}")
    parts.extend(_recent_block(recent))
    parts.append("Write the reply.")
    return "\n".join(parts)


def _upload(path: Path) -> types.File:
    client = _client()
    f = client.files.upload(file=path)
    deadline = time.monotonic() + UPLOAD_TIMEOUT
    while f.state == types.FileState.PROCESSING:
        if time.monotonic() > deadline:
            client.files.delete(name=f.name)
            raise TimeoutError("Gemini took too long to process the video")
        time.sleep(0.5)
        f = client.files.get(name=f.name)
    if f.state != types.FileState.ACTIVE:
        client.files.delete(name=f.name)
        raise RuntimeError(f"Gemini couldn't process the video (state {f.state})")
    return f


def _generate(contents: list, system: str) -> str:
    """One reply from the main model, falling back to the backup model if it's busy."""
    models = list(dict.fromkeys([os.getenv("GEMINI_MODEL") or DEFAULT_MODEL, BACKUP_MODEL]))
    for model in models:
        try:
            response = _client().models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=system),
            )
            return response.text or ""
        except errors.APIError as e:
            if not _is_temporary(e):
                raise
            last_error = e
    raise TryLater(f"all models busy ({last_error.code} {last_error.status})") from last_error


def _recent_block(recent: Sequence[str]) -> list[str]:
    if not recent:
        return []
    return ["You already sent these recently. Don't reuse any of them:", *(f"- {r}" for r in recent)]


def _reply(
    media: list, prompt: str, system: str, banned: list[str], recent: Sequence[str], allow_reaction: bool = False,
    sources: Sequence[str] = (),
) -> list[str] | Reaction | None:
    """Generate until a reply has a line that isn't a repeat, a copied example, or lifted from `sources` (the Reel's
    comments), up to MAX_ATTEMPTS."""
    tired = overused_emoji(recent)
    if tired:
        prompt += f"\n\nYou've been overusing {' '.join(sorted(tired))}. Don't use it this time."
    for _ in range(MAX_ATTEMPTS):
        lines = split_reply(_generate([*media, prompt], system))
        if allow_reaction and (reaction := as_reaction(lines)):
            return reaction
        fresh = drop_repeats(strip_emoji(lines, tired), banned)
        borrowed = [ln for ln in fresh if borrows_from(ln, sources)]
        fresh = [ln for ln in fresh if ln not in borrowed]
        for ln in borrowed:
            print(f"  (dropped a line lifted from the comments: {ln})", flush=True)
        if fresh:
            return fresh
        if not lines:
            print("  (Gemini returned nothing, likely its safety filter)", flush=True)
            return None
        for ln in lines:
            if ln not in borrowed:
                print(f"  (rejected, repeat or copied style example: {ln})", flush=True)
        if borrowed:
            prompt += f"\n\nDon't use \"{borrowed[0]}\": that's lifted from the comments. Come up with your own take."
            continue
        prompt += (
            f"\n\nDon't send \"{lines[0]}\": that's copied from an example or something you already sent. "
            "Write a brand new reply."
        )
    return None


def react(
    reel: ReelContent, friend: str, my_name: str, style: str, friend_note: str = "", recent: Sequence[str] = ()
) -> list[str] | Reaction | None:
    """Reply to a Reel as a list of messages, a Reaction if there's nothing worth texting, or None if Gemini declined
    or could only repeat old replies.

    Unless BLOCK_REPEATS=0, lines that repeat something the bot sent lately (`recent`), copy one of the style.md
    examples word for word, or lift a comment are dropped. If nothing new is left it asks again, up to MAX_ATTEMPTS.
    """
    strict = blocks_repeats()
    system = system_prompt(REEL_SITUATION, REEL_RULE, my_name, style)
    prompt = build_prompt(reel, friend, friend_note, recent if strict else ())
    banned = [*recent, *example_lines(style), *reel.comments] if strict else []
    sources = reel.comments if strict else ()
    if reel.video.stat().st_size <= INLINE_LIMIT:
        # Sending the bytes with the request skips the upload and the wait for Gemini to process it (~3s).
        video = types.Part.from_bytes(data=reel.video.read_bytes(), mime_type="video/mp4")
        return _reply([video], prompt, system, banned, recent, allow_reaction=True, sources=sources)
    client = _client()
    try:
        uploaded = _upload(reel.video)
    except errors.APIError as e:
        if _is_temporary(e):
            raise TryLater(str(e)) from e
        raise
    try:
        return _reply([uploaded], prompt, system, banned, recent, allow_reaction=True, sources=sources)
    finally:
        client.files.delete(name=uploaded.name)


def build_text_prompt(
    message: str, friend: str, context: Sequence[tuple[str, str]] = (), recent: Sequence[str] = ()
) -> str:
    parts = []
    if context:
        parts.append("Earlier in this chat (oldest first):")
        parts.extend(f"{who}: {what}" for who, what in context)
        parts.append("")
    parts.append(f"New message from {friend}: {message}")
    parts.extend(_recent_block(recent))
    parts.append("Write the reply.")
    return "\n".join(parts)


def react_text(
    message: str, friend: str, my_name: str, style: str, trigger: str,
    context: Sequence[tuple[str, str]] = (), recent: Sequence[str] = (),
) -> list[str] | None:
    """Reply to a text message, with the same repeat/copy rules as react()."""
    strict = blocks_repeats()
    system = system_prompt(TEXT_SITUATION, TEXT_RULE, my_name, style, trigger)
    prompt = build_text_prompt(message, friend, context, recent if strict else ())
    return _reply([], prompt, system, [*recent, *example_lines(style)] if strict else [], recent)
