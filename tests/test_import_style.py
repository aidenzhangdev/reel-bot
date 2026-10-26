import json
import zipfile

from reel_bot import import_style as imp


def mojibake(s: str) -> str:
    """Encode text the broken way Instagram's export does."""
    return s.encode("utf-8").decode("latin-1")


def write_export(tmp_path):
    me, sam, jo = "Alex", "Sam", "Jo"
    group = {
        "title": "the boys",
        "participants": [{"name": me}, {"name": sam}, {"name": jo}],
        "messages": [  # Instagram lists newest first
            {"sender_name": me, "timestamp_ms": 9_000_000, "content": "https://example.com/link"},
            {"sender_name": me, "timestamp_ms": 8_000_000, "content": mojibake("bro fell 😭")},
            {"sender_name": sam, "timestamp_ms": 7_990_000, "content": "Sam sent an attachment.",
             "share": {"link": "https://www.instagram.com/reel/XYZ/", "share_text": "guy slips on ice", "original_content_owner": "funnyvids"}},
            {"sender_name": jo, "timestamp_ms": 5_000_000, "content": "jo private message"},
            {"sender_name": me, "timestamp_ms": 4_000_000, "content": "Reacted 😂 to your message"},
            {"sender_name": me, "timestamp_ms": 3_000_000, "content": "nah thats wild"},
            {"sender_name": me, "timestamp_ms": 2_000_000, "content": "nah thats wild"},
        ],
    }
    dm = {"title": "Sam", "participants": [{"name": me}, {"name": sam}],
          "messages": [{"sender_name": me, "timestamp_ms": 1, "content": "dm message"}]}
    path = tmp_path / "export.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("your_instagram_activity/messages/inbox/theboys_123/message_1.json", json.dumps(group))
        z.writestr("your_instagram_activity/messages/inbox/sam_456/message_1.json", json.dumps(dm))
    return path


def test_loads_zip_and_guesses_me(tmp_path):
    chats = imp.load_chats(write_export(tmp_path))
    assert sorted(c.title for c in chats) == ["Sam", "the boys"]
    assert imp.guess_me(chats) == "Alex"


def test_extract_keeps_only_my_typed_messages_and_reel_reactions(tmp_path):
    chats = [c for c in imp.load_chats(write_export(tmp_path)) if c.title == "the boys"]
    mine, reactions = imp.extract(chats, "Alex")
    assert mine == ["bro fell 😭", "nah thats wild"]  # fixed emoji, no links/system lines/dupes, newest first
    assert reactions == [("@funnyvids: guy slips on ice", "bro fell 😭")]
    assert "jo private message" not in imp.render(mine, reactions, "the boys")


def test_merge_replaces_previous_import_and_keeps_notes():
    notes = "## How I text\n- lowercase"
    once = imp.merge_into_style(notes, imp.render(["a"], [], "x"))
    twice = imp.merge_into_style(once, imp.render(["b"], [], "x"))
    assert twice.startswith(notes) and "- b" in twice and "- a" not in twice
    assert twice.count(imp.START) == 1
