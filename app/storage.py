"""
Cross-session memory, v1: one JSON file per (friend, poet) on local disk.

This is the seam §09 of the project brief flags for a later swap to
Supabase - get_state/save_state are the only two functions the rest of
the app calls, so replacing the body of this file with Supabase reads/
writes later doesn't touch app.py or llm.py at all.

Good enough today for "I come back tomorrow and it remembers": running
locally, or on any host with a persistent disk, a friend's chat and
memory summary survive a restart. Streamlit Community Cloud's disk is
NOT guaranteed to survive a redeploy - that's exactly the gap Supabase
closes later.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

STORE_DIR = Path(__file__).resolve().parent.parent / "data" / "memory"

EMPTY_STATE: dict[str, Any] = {
    "transcript": [],       # [{"role": "user"|"poet", "text": str}, ...]
    "used_poem_ids": [],    # ids already quoted, so they don't repeat
    "current_topic": None,
    "last_open_question": None,
}


def _safe(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name, flags=re.UNICODE).strip("_") or "friend"


def _path(user: str, poet: str) -> Path:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    return STORE_DIR / f"{_safe(user)}__{_safe(poet)}.json"


def load_state(user: str, poet: str) -> dict[str, Any]:
    p = _path(user, poet)
    if not p.exists():
        return json.loads(json.dumps(EMPTY_STATE))  # deep copy
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return json.loads(json.dumps(EMPTY_STATE))


def reset_state(user: str, poet: str) -> dict[str, Any]:
    """Same name, same poet, clean slate - for the "شروع گفتگوی تازه" button.
    Overwrites the saved file too, so it doesn't come back on the next
    load_state() call."""
    fresh = json.loads(json.dumps(EMPTY_STATE))
    save_state(user, poet, fresh)
    return fresh


def save_state(user: str, poet: str, state: dict[str, Any]) -> None:
    _path(user, poet).write_text(
        json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def list_friends() -> list[str]:
    if not STORE_DIR.exists():
        return []
    seen = set()
    for p in STORE_DIR.glob("*__*.json"):
        seen.add(p.stem.split("__")[0])
    return sorted(seen)


def summarize_transcript(transcript: list[dict], max_turns: int = 16) -> str:
    """Plain-text stand-in for the persona doc's structured MEMORY
    SUMMARIZER PROMPT (CURRENT_TOPIC / USER_EXPERIENCES / OPEN_HYPOTHESES
    / ...). Keeping the last N turns verbatim is simpler to ship for v1 -
    one LLM call per message instead of two. Worth upgrading to the real
    structured summarizer once conversations regularly run long."""
    recent = transcript[-max_turns:]
    lines = []
    for turn in recent:
        speaker = "سالک" if turn["role"] == "user" else "شاعر"
        lines.append(f"{speaker}: {turn['text']}")
    return "\n".join(lines) if lines else "(این اولین پیام گفتگوست.)"
