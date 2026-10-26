# reel-bot

An Instagram DM bot that replies to Reels your friends send it, in *your* texting style.

Friends DM a Reel to a bot account. The bot watches the video (sound included) with Google Gemini, reads the caption and top comments for context, and texts back the way you would: a joke, a roast, a hot take. If there's nothing worth saying, it just reacts ❤️ or 👍. It can also answer texts that start with a trigger word.

Your style comes from `data/style.md`: a few notes on how you text plus real messages you've sent. `import-style` can fill it automatically from an Instagram data export of a group chat.

## How it works
- **Instagram:** [instagrapi](https://github.com/subzeroid/instagrapi) (the unofficial private API) polls the bot account's DMs every ~10s.
- **Reels:** the bot downloads a small version of the video, fetches the top 5 comments, and sends it all to Gemini (free tier), which writes the reply. Replies quote the message they answer.
- **Texts:** messages whose first word is `TEXT_TRIGGER` get a reply, using the last few messages in the chat as context.
- **Guardrails:** an allow-list of who it replies to, an hourly reply cap, a single-instance lock, a dry-run mode, optional no-repeat filtering, and it stops right away if Instagram flags the account.

## Setup
Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```sh
uv sync
cp .env.example .env               # fill in the bot account login and a free Gemini key
cp data/style.example.md data/style.md   # then fill in how you text
```

Use a **burner Instagram account** for the bot. Automating Instagram is against its terms, and the account could get flagged.

## Usage
```sh
uv run import-style ~/Downloads/instagram-export.zip               # list your chats
uv run import-style ~/Downloads/instagram-export.zip --chat "name" # add your messages from one chat to style.md

uv run try-reel https://www.instagram.com/reel/XXXX/ -n 3   # preview replies to one Reel, sends nothing
uv run reel-bot                                            # run the bot (DRY_RUN=1 only prints replies)
```

`import-style` keeps only messages *you* sent. Friends' messages and names never go into `style.md`.

## Tests
```sh
uv run pytest
```

## Privacy
`.env` (logins and API key), `session.json` (Instagram session), `data/style.md` (your messages) and `data/state.db` are git-ignored and never leave your machine.
