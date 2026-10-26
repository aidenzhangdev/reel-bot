"""What we know about a downloaded Reel. Gemini watches the video itself, so no preprocessing is needed."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ReelContent:
    video: Path
    caption: str = ""
    author: str = ""
    comments: list[str] = field(default_factory=list)  # top comments, most liked first
