import time
from types import SimpleNamespace

import pytest

from reel_bot.brain import STYLE_PLACEHOLDER, build_prompt, drop_repeats, example_lines, load_style, split_reply
from reel_bot.ig import reel_source
from reel_bot.reel import ReelContent
from reel_bot.state import State


def msg(**kw):
    base = dict(clip=None, media_share=None, item_type="text", xma_share=None)
    return SimpleNamespace(**(base | kw))


@pytest.mark.parametrize(
    "m, expected",
    [
        (msg(item_type="clip", clip=SimpleNamespace(pk=123)), ("123", None)),
        (msg(item_type="media_share", media_share=SimpleNamespace(pk=7, media_type=2)), ("7", None)),
        (msg(item_type="media_share", media_share=SimpleNamespace(pk=7, media_type=1)), (None, None)),
        (
            msg(item_type="xma_clip", xma_share=SimpleNamespace(video_url="https://www.instagram.com/reel/ABC/")),
            (None, "https://www.instagram.com/reel/ABC/"),
        ),
        (msg(item_type="xma_clip", xma_share=SimpleNamespace(video_url="instagram://x")), (None, None)),
        (msg(item_type="text"), (None, None)),
    ],
)
def test_reel_source(m, expected):
    assert reel_source(m) == expected


def test_split_reply_cleans_and_caps():
    assert split_reply('"nah 😭"\n\n  bro really did that\nlmao\nfourth') == ["nah 😭", "bro really did that", "lmao"]
    assert split_reply("   ") == []


def test_build_prompt(tmp_path):
    text = build_prompt(ReelContent(tmp_path / "v.mp4", caption="cap", author="creator"), "sam", "look")
    assert "sam" in text and "@creator" in text and "cap" in text and "look" in text
    assert "(none)" in build_prompt(ReelContent(tmp_path / "v.mp4"), "sam")


def test_load_style_rejects_unfilled_template(tmp_path):
    f = tmp_path / "style.md"
    f.write_text(f"## msgs\n{STYLE_PLACEHOLDER}\n- hi")
    with pytest.raises(SystemExit):
        load_style(f)
    f.write_text("## msgs\n- hi")
    assert load_style(f) == "## msgs\n- hi"
    # Mentioning the placeholder in the instructions comment is fine.
    f.write_text(f'<!-- delete the line that says "{STYLE_PLACEHOLDER}" -->\n- hi')
    assert load_style(f).endswith("- hi")


def test_state(tmp_path):
    st = State(tmp_path / "s.db")
    t0 = st.started_at()
    assert State(tmp_path / "s.db").started_at() == t0  # persists across restarts
    assert not st.is_handled("m1")
    st.mark_handled("m1", "t1", "sent", "yo")
    st.mark_handled("m2", "t1", "dry_run", "yo")
    assert st.is_handled("m1") and st.is_handled("m2")
    assert st.sent_in_last_hour() == 1
    assert t0 <= time.time()


class FakeGemini:
    """Stands in for genai.Client: each model either returns text or fails with an HTTP code."""

    def __init__(self, behaviour):
        self.behaviour, self.calls, self.deleted = behaviour, [], []
        self.models = SimpleNamespace(generate_content=self._generate)
        self.files = SimpleNamespace(delete=lambda name: self.deleted.append(name))

    def _generate(self, model, contents, config):
        from google.genai import errors

        self.calls.append(model)
        result = self.behaviour[model]
        if isinstance(result, list):  # successive replies from the same model
            result = result.pop(0)
        if isinstance(result, int):
            raise (errors.ServerError if result >= 500 else errors.ClientError)(result, {"error": {"code": result, "status": "X", "message": "x"}})
        return SimpleNamespace(text=result)


@pytest.fixture
def fake_gemini(monkeypatch, tmp_path):
    from reel_bot import brain

    def install(behaviour):
        fake = FakeGemini(behaviour)
        monkeypatch.setattr(brain, "_client", lambda: fake)
        monkeypatch.setattr(brain, "_upload", lambda path: SimpleNamespace(name="files/1"))
        monkeypatch.delenv("GEMINI_MODEL", raising=False)
        monkeypatch.delenv("BLOCK_REPEATS", raising=False)  # filters on, whatever .env says
        monkeypatch.setattr(brain, "INLINE_LIMIT", -1)  # use the upload path unless a test opts in
        return fake

    (tmp_path / "v.mp4").write_bytes(b"fake video")
    return install, ReelContent(tmp_path / "v.mp4")


def test_react_falls_back_to_backup_model_when_busy(fake_gemini):
    from reel_bot import brain

    install, reel = fake_gemini
    fake = install({brain.DEFAULT_MODEL: 503, brain.BACKUP_MODEL: "lmao"})
    assert brain.react(reel, "sam", "me", "- yo") == ["lmao"]
    assert fake.calls == [brain.DEFAULT_MODEL, brain.BACKUP_MODEL]
    assert fake.deleted == ["files/1"]


def test_react_asks_to_try_later_when_all_models_busy(fake_gemini):
    from reel_bot import brain

    install, reel = fake_gemini
    fake = install({brain.DEFAULT_MODEL: 503, brain.BACKUP_MODEL: 429})
    with pytest.raises(brain.TryLater):
        brain.react(reel, "sam", "me", "- yo")
    assert fake.deleted == ["files/1"]


def test_react_does_not_retry_real_errors(fake_gemini):
    from google.genai import errors

    from reel_bot import brain

    install, reel = fake_gemini
    fake = install({brain.DEFAULT_MODEL: 400, brain.BACKUP_MODEL: "unused"})
    with pytest.raises(errors.ClientError):
        brain.react(reel, "sam", "me", "- yo")
    assert fake.calls == [brain.DEFAULT_MODEL]


def test_drop_repeats_ignores_case_punctuation_and_emoji():
    recent = ["this kid cooked beyond repair 😭", "🍟"]
    assert drop_repeats(["This kid cooked beyond repair!!", "bro what", "🍟"], recent) == ["bro what", "🍟"]


def test_recent_replies_splits_bursts_newest_first(tmp_path):
    st = State(tmp_path / "s.db")
    st.mark_handled("m1", "t", "sent", "old one")
    st.mark_handled("m2", "t", "error", "never sent")
    st.mark_handled("m3", "t", "sent", "new a\nnew b")
    assert st.recent_replies() == ["new a", "new b", "old one"]


def test_react_tells_gemini_about_recent_replies_and_retries_repeats(fake_gemini):
    from reel_bot import brain

    install, reel = fake_gemini
    fake = install({brain.DEFAULT_MODEL: ["cooked beyond repair 😭", "bro is cooked"], brain.BACKUP_MODEL: "unused"})
    seen_prompts = []
    original = fake._generate
    fake.models.generate_content = lambda model, contents, config: (seen_prompts.append(contents[1]), original(model, contents, config))[1]
    assert brain.react(reel, "sam", "me", "- yo", recent=["cooked beyond repair"]) == ["bro is cooked"]
    assert "- cooked beyond repair" in seen_prompts[0]
    assert "brand new reply" in seen_prompts[1]


def test_react_gives_up_if_it_only_repeats(fake_gemini):
    from reel_bot import brain

    install, reel = fake_gemini
    install({brain.DEFAULT_MODEL: ["cooked beyond repair", "Cooked beyond repair!", "cooked beyond repair 😭"], brain.BACKUP_MODEL: "unused"})
    assert brain.react(reel, "sam", "me", "- yo", recent=["cooked beyond repair"]) is None


def test_second_copy_of_bot_refuses_to_start(tmp_path, monkeypatch):
    import subprocess
    import sys

    from reel_bot import main

    monkeypatch.setattr(main, "LOCK_FILE", tmp_path / "bot.lock")
    held = main.acquire_single_instance_lock()
    code = (
        "import sys; from pathlib import Path; from reel_bot import main; "
        f"main.LOCK_FILE = Path({str(tmp_path / 'bot.lock')!r}); main.acquire_single_instance_lock()"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode != 0 and "already running" in result.stderr
    held.close()


def test_example_lines_only_long_ones_and_only_my_side():
    style = "## How I text\n- all lowercase\n- bro how\n- u werent even there bro\n- friend sent: @x: some long caption here → me: go touch grass already"
    assert example_lines(style) == ["u werent even there bro", "go touch grass already"]


def test_react_rejects_replies_copied_from_style_examples(fake_gemini):
    from reel_bot import brain

    install, reel = fake_gemini
    style = "- u werent even there bro\n- bro how"
    install({brain.DEFAULT_MODEL: ["U werent even there bro!", "bro how", "bro the choreo is so off"], brain.BACKUP_MODEL: "unused"})
    assert brain.react(reel, "sam", "me", style) == ["bro how"]  # short lines may repeat; long copies are rejected


@pytest.mark.parametrize(
    "text, expected",
    [("bot what should i eat", True), ("BOT!!", True), ("  Bot, help", True), ("bot", True),
     ("botz are cool", False), ("yo bot", False), ("", False), (None, False)],
)
def test_is_trigger(text, expected):
    from reel_bot.ig import is_trigger

    assert is_trigger(text, "bot") is expected
    assert is_trigger(text, "") is False  # blank trigger turns text replies off


def make_ig(threads, my_id="1"):
    from reel_bot.ig import Instagram

    ig = Instagram.__new__(Instagram)  # skip the real login
    ig.my_id = my_id
    ig.cl = SimpleNamespace(direct_threads=lambda **kw: threads)
    return ig


def dm(id, user_id, ts, text=None, **kw):
    from datetime import datetime, timezone

    base = dict(clip=None, media_share=None, item_type="text", xma_share=None)
    return SimpleNamespace(id=id, user_id=user_id, text=text, timestamp=datetime.fromtimestamp(ts, timezone.utc), **(base | kw))


def test_fetch_new_finds_trigger_texts_with_context_and_skips_the_rest():
    users = [SimpleNamespace(pk=2, username="Sam")]
    msgs = [  # newest first, like Instagram
        dm("m5", "2", 500, "bot what do u think"),
        dm("m4", "1", 400, "nah"),  # the bot itself
        dm("m3", "2", 300, "should i get the jordans"),
        dm("m2", "2", 200, "hello bot"),  # trigger not first: ignored
        dm("m1", "2", 50, "bot old message"),  # before the bot started
    ]
    ig = make_ig([SimpleNamespace(id="t1", users=users, messages=msgs)])
    found = ig.fetch_new(100, frozenset({"sam"}), lambda _: False, "bot")
    assert [(f.kind, f.message_id) for f in found] == [("text", "m5")]
    # Context is the earlier chat, oldest first, including messages from before the bot started.
    assert found[0].context == [
        ("sam", "bot old message"), ("sam", "hello bot"), ("sam", "should i get the jordans"), ("you", "nah")
    ]


def test_fetch_new_trigger_text_right_after_a_reel_is_answered_by_the_reel_reply():
    users = [SimpleNamespace(pk=2, username="sam")]
    msgs = [dm("m2", "2", 300, "bot look at this"), dm("m1", "2", 200, item_type="clip", clip=SimpleNamespace(pk=9))]
    ig = make_ig([SimpleNamespace(id="t1", users=users, messages=msgs)])
    found = ig.fetch_new(100, frozenset({"*"}), lambda _: False, "bot")
    assert [(f.kind, f.message_id, f.note) for f in found] == [("reel", "m1", "bot look at this")]


def test_react_text_sends_context_and_no_video(fake_gemini):
    from reel_bot import brain

    install, _ = fake_gemini
    fake = install({brain.DEFAULT_MODEL: "get the jordans bro", brain.BACKUP_MODEL: "unused"})
    seen = []
    original = fake._generate
    fake.models.generate_content = lambda model, contents, config: (seen.append((contents, config)), original(model, contents, config))[1]
    reply = brain.react_text("bot should i", "sam", "me", "- yo", "bot", context=[("sam", "jordans or dunks")])
    assert reply == ["get the jordans bro"]
    contents, config = seen[0]
    assert len(contents) == 1 and "sam: jordans or dunks" in contents[0] and "bot should i" in contents[0]
    assert '"bot"' in config.system_instruction and "Watch the whole video" not in config.system_instruction


def test_react_turns_reaction_tokens_into_emoji_reactions(fake_gemini):
    from reel_bot import brain

    install, reel = fake_gemini
    install({brain.DEFAULT_MODEL: ["[heart]", "[Thumbs Up]"], brain.BACKUP_MODEL: "unused"})
    assert brain.react(reel, "sam", "me", "- yo") == brain.Reaction("❤️")
    assert brain.react(reel, "sam", "me", "- yo") == brain.Reaction("👍")


def test_text_replies_never_become_reactions(fake_gemini):
    from reel_bot import brain

    install, _ = fake_gemini
    install({brain.DEFAULT_MODEL: "[heart]", brain.BACKUP_MODEL: "unused"})
    assert brain.react_text("bot yo", "sam", "me", "- yo", "bot") == ["[heart]"]


def test_overused_emoji_is_stripped_and_mentioned(fake_gemini):
    from reel_bot import brain

    install, reel = fake_gemini
    fake = install({brain.DEFAULT_MODEL: "bro is not him 😭\n😭", brain.BACKUP_MODEL: "unused"})
    seen = []
    original = fake._generate
    fake.models.generate_content = lambda model, contents, config: (seen.append(contents[1]), original(model, contents, config))[1]
    recent = ["a 😭", "b 😭😭", "c 😂", "d 😭", "e"]
    assert brain.react(reel, "sam", "me", "- yo", recent=recent) == ["bro is not him"]
    assert "overusing 😭" in seen[0]


def test_overused_emoji_needs_more_than_the_limit():
    from reel_bot.brain import overused_emoji, strip_emoji

    assert overused_emoji(["a 😭", "b 😭", "c ❤️", "d ❤", "e ❤️"]) == {"❤"}
    assert strip_emoji(["love it ❤️ 😂"], {"❤"}) == ["love it 😂"]


def test_emoji_only_reel_reply_becomes_a_heart(fake_gemini):
    from reel_bot import brain

    install, reel = fake_gemini
    install({brain.DEFAULT_MODEL: "🍔", brain.BACKUP_MODEL: "unused"})
    assert brain.react(reel, "sam", "me", "- yo") == brain.Reaction("❤️")


def test_fetch_new_keeps_the_dm_so_replies_can_quote_it():
    users = [SimpleNamespace(pk=2, username="sam")]
    reel, text = dm("m1", "2", 200, item_type="clip", clip=SimpleNamespace(pk=9)), dm("m2", "3", 300, "bot yo")
    ig = make_ig([SimpleNamespace(id="t1", users=users + [SimpleNamespace(pk=3, username="kai")], messages=[text, reel])])
    found = ig.fetch_new(100, frozenset({"*"}), lambda _: False, "bot")
    assert [f.message for f in found] == [reel, text]


def test_send_quotes_the_message_it_replies_to():
    sent = []
    ig = make_ig([])
    ig.cl.direct_send = lambda text, thread_ids, reply_to_message=None: sent.append((text, thread_ids, reply_to_message))
    original = dm("m1", "2", 200, "bot yo")
    ig.send("42", "yo", reply_to=original)
    assert sent == [("yo", [42], original)]


def test_top_comments_most_liked_first_skips_replies_and_caps():
    from reel_bot import ig as ig_mod

    def c(text, likes, reply_to=None):
        return {"text": text, "comment_like_count": likes, "parent_comment_id": reply_to}

    ig = make_ig([])
    page = [c("ok", 1), c("lmao", 50), c("reply", 999, "x"), c("  ", 70), *[c(f"c{i}", 10 + i) for i in range(6)]]
    requested = []
    ig.cl.private_request = lambda path, params: (requested.append(path), {"comments": page})[1]
    assert ig.top_comments("1_2") == ["lmao", "c5", "c4", "c3", "c2"]
    assert len(ig.top_comments("1_2")) == ig_mod.TOP_COMMENTS
    assert requested[0] == "media/1_2/comments/"  # one request, by the full id we already have

    def boom(path, params):
        raise RuntimeError("comments are off")

    ig.cl.private_request = boom
    assert ig.top_comments("1_2") == []


def test_build_prompt_includes_top_comments(tmp_path):
    text = build_prompt(ReelContent(tmp_path / "v.mp4", comments=["he thinks hes him", "the shoes 💀"]), "sam")
    assert "Top comments" in text and "- he thinks hes him" in text and "- the shoes 💀" in text


def test_borrows_from_catches_copies_and_rewords_but_not_own_takes():
    from reel_bot.brain import borrows_from

    comments = ["he really thinks he's the main character", "the guy in the back stole the show"]
    assert borrows_from("He really thinks he's the main character 😭", comments)  # copy
    assert borrows_from("bro thinks hes the main character", comments)  # 4 words in a row
    assert borrows_from("guy in back stole show", comments)  # reworded: most of its words come from one comment
    assert not borrows_from("bro this is u at every party", comments)  # its own take
    assert not borrows_from("bro he is so cooked", comments)  # only common words in common
    assert not borrows_from("lmao", comments)


def test_react_retries_replies_lifted_from_comments(fake_gemini):
    from reel_bot import brain

    install, plain = fake_gemini
    reel = ReelContent(plain.video, comments=["the guy in the back stole the show"])
    fake = install({brain.DEFAULT_MODEL: ["guy in the back stole the show fr", "send this to max"], brain.BACKUP_MODEL: "unused"})
    seen = []
    original = fake._generate
    fake.models.generate_content = lambda model, contents, config: (seen.append(contents[1]), original(model, contents, config))[1]
    assert brain.react(reel, "sam", "me", "- yo") == ["send this to max"]
    assert "the guy in the back stole the show" in seen[0]
    assert "lifted from the comments" in seen[1]


def test_small_videos_go_inline_without_an_upload(fake_gemini, monkeypatch):
    from reel_bot import brain

    install, reel = fake_gemini
    fake = install({brain.DEFAULT_MODEL: "send this to max", brain.BACKUP_MODEL: "unused"})
    monkeypatch.setattr(brain, "INLINE_LIMIT", 1024)
    monkeypatch.setattr(brain, "_upload", lambda path: pytest.fail("should not upload a small video"))
    seen = []
    original = fake._generate
    fake.models.generate_content = lambda model, contents, config: (seen.append(contents), original(model, contents, config))[1]
    assert brain.react(reel, "sam", "me", "- yo") == ["send this to max"]
    video = seen[0][0]
    assert video.inline_data.data == b"fake video" and video.inline_data.mime_type == "video/mp4"
    assert fake.deleted == []


def test_download_fetches_comments_alongside_the_video(tmp_path):
    ig = make_ig([])
    ig._media_and_small_video = lambda pk: (
        SimpleNamespace(id="9_2", caption_text="cap", user=SimpleNamespace(username="creator")), "https://cdn/v.mp4"
    )
    ig.cl.video_download_by_url = lambda url, folder: folder / "v.mp4"
    ig.cl.private_request = lambda path, params: {"comments": [{"text": "lol", "comment_like_count": 3}]}
    reel = ig.download("9", None, tmp_path)
    assert (reel.video, reel.caption, reel.author, reel.comments) == (tmp_path / "v.mp4", "cap", "creator", ["lol"])


def test_smallest_video_prefers_the_smallest_that_is_still_clear_enough():
    from reel_bot.ig import smallest_video

    def v(h):
        return {"height": h, "width": h * 9 // 16, "url": f"{h}p"}

    assert smallest_video([v(1080), v(360), v(720), v(480)]) == "480p"
    assert smallest_video([v(360), v(240)]) == "240p"


def test_download_uses_a_small_video_version_and_falls_back_to_media_info(monkeypatch):
    import reel_bot.ig as ig_mod

    ig = make_ig([])
    raw = {"height": 480, "width": 270, "url": "https://cdn/480.mp4"}
    ig.cl.private_request = lambda path: {"items": [{"video_versions": [raw, {**raw, "height": 1080, "url": "big"}]}]}
    ig.cl.media_info = lambda pk: pytest.fail("should not need media_info")
    extracted = SimpleNamespace(caption_text="cap", user=SimpleNamespace(username="creator"))
    monkeypatch.setattr(ig_mod, "extract_media_v1", lambda item: extracted)
    assert ig._media_and_small_video("9") == (extracted, "https://cdn/480.mp4")

    info = SimpleNamespace(video_url="https://cdn/best.mp4")
    ig.cl.private_request = lambda path: {"items": [{}]}  # no video_versions
    ig.cl.media_info = lambda pk: info
    assert ig._media_and_small_video("9") == (info, "https://cdn/best.mp4")


def test_block_repeats_off_lets_anything_through(fake_gemini, monkeypatch):
    from reel_bot import brain

    install, plain = fake_gemini
    reel = ReelContent(plain.video, comments=["the guy in the back stole the show"])
    fake = install({brain.DEFAULT_MODEL: ["the guy in the back stole the show", "cooked beyond repair"], brain.BACKUP_MODEL: "unused"})
    seen = []
    original = fake._generate
    fake.models.generate_content = lambda model, contents, config: (seen.append(contents[-1]), original(model, contents, config))[1]
    monkeypatch.setenv("BLOCK_REPEATS", "0")
    assert brain.react(reel, "sam", "me", "- yo", recent=["x"]) == ["the guy in the back stole the show"]
    assert brain.react_text("bot yo", "sam", "me", "- yo", "bot", recent=["cooked beyond repair"]) == ["cooked beyond repair"]
    assert all("already sent these" not in p for p in seen)
