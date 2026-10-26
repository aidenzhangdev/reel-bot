"""Try the personality on one Reel without touching DMs.

    uv run try-reel https://www.instagram.com/reel/XXXX/      # logs in to the bot account to download it
    uv run try-reel ~/Downloads/some_video.mp4                 # local file, no Instagram needed
"""

import argparse
import shutil
from pathlib import Path

from . import brain
from .config import SESSION_FILE, STYLE_FILE, TMP_DIR, load_settings
from .reel import ReelContent


def main() -> None:
    p = argparse.ArgumentParser(description="Show the reply the bot would send for a Reel.")
    p.add_argument("source", help="Reel URL or path to a local video file")
    p.add_argument("--friend", default="a friend", help="name of the friend who sent it")
    p.add_argument("--note", default="", help="text the friend sent with it")
    p.add_argument("-n", type=int, default=1, help="how many different replies to generate")
    args = p.parse_args()

    s = load_settings()
    style = brain.load_style(STYLE_FILE)
    workdir = TMP_DIR / "try"
    try:
        local = Path(args.source).expanduser()
        if local.exists():
            content = ReelContent(local)
        else:
            from .ig import Instagram

            ig = Instagram(s.ig_username, s.ig_password, SESSION_FILE)
            content = ig.download(None, args.source, workdir)

        print(f"by @{content.author or '?'}")
        print(f"caption: {content.caption[:200] or '(none)'}")
        print("top comments:" if content.comments else "top comments: (none)")
        for c in content.comments:
            print(f"  - {c}")
        print()
        shown: list[str] = []
        for _ in range(args.n):
            reply = brain.react(content, args.friend, s.my_name, style, args.note, recent=shown)
            if isinstance(reply, brain.Reaction):
                print(f"-> (reacts {reply.emoji} instead of texting)")
                continue
            print("->", " | ".join(reply) if reply else "(Gemini declined)")
            shown += reply or []
    except brain.TryLater as e:
        print(f"Gemini is busy right now ({e}). Try again in a minute.")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
