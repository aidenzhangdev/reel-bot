"""Polling loop: watch the bot account's DMs and reply to Reels (and trigger-word texts) from allowed friends."""

import fcntl
import random
import shutil
import sys
import time
import traceback

from . import brain
from .config import LOCK_FILE, SESSION_FILE, STATE_DB, STYLE_FILE, TMP_DIR, load_settings
from .ig import STOP_ERRORS, Instagram
from .state import State


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def acquire_single_instance_lock():
    """Exit if another copy of the bot is already running. Two copies would both answer the same Reel."""
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("The bot is already running in another terminal. Stop that one first (Ctrl-C).")
    return lock  # keep the file open; the OS releases the lock when this process exits


def main() -> None:
    lock = acquire_single_instance_lock()  # noqa: F841
    s = load_settings()
    if not (s.ig_username and s.ig_password):
        sys.exit("Set IG_USERNAME and IG_PASSWORD in .env")
    if not s.allowed_users:
        sys.exit('Set ALLOWED_USERS in .env (comma-separated usernames, or "*" for anyone)')
    style = brain.load_style(STYLE_FILE)
    state = State(STATE_DB)
    since = state.started_at()

    log(f"Logging in as @{s.ig_username}...")
    try:
        ig = Instagram(s.ig_username, s.ig_password, SESSION_FILE)
    except STOP_ERRORS as e:
        sys.exit(f"Instagram blocked the login ({type(e).__name__}: {e}). Open the account in the Instagram app, clear any prompts, then try again.")
    mode = "DRY RUN (printing replies, not sending)" if s.dry_run else "LIVE (sending replies)"
    log(f"Watching DMs - {mode}. Ctrl-C to stop.")

    try:
        while True:
            try:
                incoming = ig.fetch_new(since, s.allowed_users, state.is_handled, s.text_trigger)
            except STOP_ERRORS as e:
                log(f"Instagram flagged the account ({type(e).__name__}: {e}). Stopping; check the app before restarting.")
                return
            except Exception:
                log("Couldn't fetch DMs, will retry:\n" + traceback.format_exc())
                incoming = []

            for r in incoming:
                if not s.dry_run and state.sent_in_last_hour() >= s.max_replies_per_hour:
                    log("Hit the hourly reply cap; the rest will wait for a later poll.")
                    break
                workdir = TMP_DIR / r.message_id
                try:
                    if r.kind == "reel":
                        log(f"Reel from @{r.sender}" + (f' ("{r.note}")' if r.note else ""))
                        reel = ig.download(r.media_pk, r.media_url, workdir)
                        reply = brain.react(reel, r.sender, s.my_name, style, r.note, state.recent_replies())
                    else:
                        log(f'Text from @{r.sender}: "{r.text}"')
                        reply = brain.react_text(
                            r.text, r.sender, s.my_name, style, s.text_trigger, r.context, state.recent_replies()
                        )
                except brain.TryLater as e:
                    log(f"Gemini is busy ({e}); will retry on a later poll.")
                    break
                except STOP_ERRORS as e:
                    log(f"Instagram flagged the account ({type(e).__name__}: {e}). Stopping.")
                    return
                except Exception:
                    log(f"Failed on this {r.kind}, skipping it:\n" + traceback.format_exc())
                    state.mark_handled(r.message_id, r.thread_id, "error")
                    continue
                finally:
                    shutil.rmtree(workdir, ignore_errors=True)

                if reply is None:
                    log("  Gemini declined or only repeated old replies; skipping.")
                    state.mark_handled(r.message_id, r.thread_id, "declined")
                    continue
                if isinstance(reply, brain.Reaction):
                    if s.dry_run:
                        log(f"  would react {reply.emoji}")
                        state.mark_handled(r.message_id, r.thread_id, "dry_run")
                        continue
                    try:
                        ig.react(r.thread_id, r.message_id, reply.emoji)
                    except STOP_ERRORS as e:
                        log(f"Instagram flagged the account while reacting ({type(e).__name__}: {e}). Stopping.")
                        return
                    except Exception:
                        log("Failed to react, skipping this reel:\n" + traceback.format_exc())
                        state.mark_handled(r.message_id, r.thread_id, "error", reply.emoji)
                        continue
                    state.mark_handled(r.message_id, r.thread_id, "reacted", reply.emoji)
                    log(f"  reacted {reply.emoji}")
                    continue
                text = "\n".join(reply)
                if s.dry_run:
                    log("  would reply: " + " | ".join(reply))
                    state.mark_handled(r.message_id, r.thread_id, "dry_run", text)
                    continue
                try:
                    for i, line in enumerate(reply):
                        if i:
                            time.sleep(random.uniform(0.5, 1.5))
                        ig.send(r.thread_id, line, reply_to=r.message if i == 0 else None)  # quote it once, up top
                except STOP_ERRORS as e:
                    log(f"Instagram flagged the account while sending ({type(e).__name__}: {e}). Stopping.")
                    return
                except Exception:
                    log(f"Failed to send, skipping this {r.kind}:\n" + traceback.format_exc())
                    state.mark_handled(r.message_id, r.thread_id, "error", text)
                    continue
                state.mark_handled(r.message_id, r.thread_id, "sent", text)
                log("  sent: " + " | ".join(reply))

            time.sleep(s.poll_seconds + random.uniform(0, 3))
    except KeyboardInterrupt:
        log("Stopped.")


if __name__ == "__main__":
    main()
