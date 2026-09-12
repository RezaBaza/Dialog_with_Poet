"""
Composes CORE_SYSTEM_PROMPT + PERSONA + RUNTIME_INSTRUCTION + VERIFIED_POEMS
(the formula from Salek_Poet_Agent_Personas_FA, section 1) and calls the
model. Two providers are wired up - see PROVIDER below for which one is
actually active. The prompt composition above generate_stream() doesn't
change no matter which provider is picked.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

POETS = {
    "molana": {"title": "مولانا", "persona_file": "persona_molana.txt"},
    "hafez": {"title": "حافظ", "persona_file": "persona_hafez.txt"},
    "saadi": {"title": "سعدی", "persona_file": "persona_saadi.txt"},
    "attar": {"title": "عطار", "persona_file": "persona_attar.txt"},
    "khayyam": {"title": "خیام", "persona_file": "persona_khayyam.txt"},
}

# v1 only wires retrieval up for Rumi (see pipeline/build_corpus.py and
# the project brief §06) - the other four personas are ready to go the
# moment their corpora are built the same way.
ACTIVE_POETS = ["molana"]


def _read(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


CORE_PROMPT = _read("core_system_prompt.txt")
RUNTIME_TEMPLATE = _read("runtime_template.txt")


def format_verified_poems(poems: list) -> str:
    if not poems:
        return "(خالی - هیچ شعر معتبری بازیابی نشده، مستقیم نقل نکن)"
    blocks = []
    for p in poems:
        kind = "نثر" if p.is_prose else "شعر"
        blocks.append(
            f"- [{kind} | {p.book_title} | {p.full_title}]\n{p.text}"
        )
    return "\n\n".join(blocks)


def build_system_prompt(poet_key: str, topic: str, conversation_summary: str,
                         last_open_question: str, user_message: str,
                         verified_poems: list) -> str:
    persona = _read(POETS[poet_key]["persona_file"])
    runtime = RUNTIME_TEMPLATE.format(
        poet=POETS[poet_key]["title"],
        topic=topic or "(آزاد)",
        user_message=user_message or "(خالی - اولین نوبت گفتگوست)",
        conversation_summary=conversation_summary,
        last_open_question=last_open_question or "(ندارد)",
        verified_poems=format_verified_poems(verified_poems),
    )
    return (
        f"{CORE_PROMPT}\n\n"
        f"=== PERSONA: {POETS[poet_key]['title']} ===\n{persona}\n\n"
        f"=== RUNTIME INSTRUCTION ===\n{runtime}"
    )


class LLMError(RuntimeError):
    pass


def friendly_error_message(e: Exception) -> str:
    """Turns an SDK exception into short Persian text for the chat UI - the
    raw exception (a nested dict of quota metrics, retry URLs, etc.) was
    getting dumped straight into the conversation. The real error still
    goes to the console via print(), just not to the person chatting."""
    print(f"[llm error] {type(e).__name__}: {e}", file=sys.stderr)

    if isinstance(e, LLMError):
        return str(e)
    text = str(e).lower()
    if "resource_exhausted" in text or "429" in text or "rate_limit" in text or "overloaded" in text:
        return (
            "سقف استفاده‌ی فعلی این مدل پر شده. چند دقیقه دیگه دوباره امتحان کن."
        )
    return "یه مشکل موقت در ارتباط با مدل پیش اومد. یه بار دیگه امتحان کن."


def generate(system_prompt: str, user_message: str) -> str:
    """Blocking convenience wrapper around generate_stream() - see there for
    which provider actually handles the call."""
    return "".join(generate_stream(system_prompt, user_message))


def _provider() -> str:
    # OpenAI first when its key is present - added to try a model reputed
    # to be strong on Farsi at low cost. Claude next (added earlier to get
    # past Gemini's 20-requests/day free-tier cap). Gemini last as the
    # original fallback. Removing any key just falls through to the next.
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    return "gemini"


def generate_stream(system_prompt: str, user_message: str):
    """Yields text chunks as they arrive. Dispatches to whichever provider
    _provider() picks; the prompt itself (built by build_system_prompt) is
    identical either way."""
    provider = _provider()
    if provider == "openai":
        yield from _generate_stream_openai(system_prompt, user_message)
    elif provider == "claude":
        yield from _generate_stream_claude(system_prompt, user_message)
    else:
        yield from _generate_stream_gemini(system_prompt, user_message)


def _generate_stream_openai(system_prompt: str, user_message: str):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise LLMError(
            "OPENAI_API_KEY تنظیم نشده. آن را در فایل .env یا در "
            "Streamlit secrets قرار بده (به README مراجعه کن)."
        )
    import openai

    client = openai.OpenAI(api_key=api_key)
    # gpt-4o-mini: cheap ($0.15/$0.60 per 1M tokens in/out) and solidly
    # multilingual, which is why it's the default "try a cheap OpenAI model
    # strong on Farsi" pick. Override with OPENAI_MODEL to compare others.
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    got_any = False
    with client.chat.completions.stream(
        model=model_name,
        max_completion_tokens=3000,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message or "(سالک هنوز چیزی ننوشته - گفتگو را باز کن)"},
        ],
    ) as stream:
        for event in stream:
            if event.type == "content.delta" and event.delta:
                got_any = True
                yield event.delta
    if not got_any:
        raise LLMError("پاسخی از مدل دریافت نشد.")


def _generate_stream_claude(system_prompt: str, user_message: str):
    api_key = os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMError(
            "CLAUDE_API_KEY تنظیم نشده. آن را در فایل .env یا در "
            "Streamlit secrets قرار بده (به README مراجعه کن)."
        )
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    # Haiku 4.5 by default: cheap enough (~$1/$5 per 1M tokens in/out) that
    # a friends-group pilot costs a few dollars a month at most, per the
    # project brief's §09 estimate. Set CLAUDE_MODEL=claude-sonnet-5 for
    # better persona/tone quality once that's worth paying more for.
    model_name = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
    got_any = False
    with client.messages.stream(
        model=model_name,
        max_tokens=3000,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": user_message or "(سالک هنوز چیزی ننوشته - گفتگو را باز کن)",
        }],
    ) as stream:
        for text in stream.text_stream:
            if text:
                got_any = True
                yield text
    if not got_any:
        raise LLMError("پاسخی از مدل دریافت نشد.")


def _generate_stream_gemini(system_prompt: str, user_message: str):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise LLMError(
            "GEMINI_API_KEY تنظیم نشده. آن را در فایل .env یا در "
            "Streamlit secrets قرار بده (به README مراجعه کن)."
        )
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        # This is the actual fix for the ~30-40s wait: gemini-3.6-flash
        # reasons silently before emitting any visible text by default, even
        # with streaming on - nothing appears until that finishes. This
        # persona doesn't need heavy reasoning (the CORE prompt's own
        # step-by-step algorithm already does that work), so turn it down.
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
        # We never give the model tools to call - disabling this skips an
        # SDK-side capability probe on every request.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        # A 700-token cap here was tried as a latency backstop and had to be
        # reverted: this model's internal "thinking" tokens apparently count
        # against the same max_output_tokens budget, so on turns where it
        # reasoned more, the visible reply got cut off mid-sentence with no
        # error - worse than the slow turns it was meant to fix. 3000 leaves
        # wide headroom for thinking + a ~220-word reply, while still
        # guarding against a truly runaway generation.
        max_output_tokens=3000,
    )
    model_name = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
    got_any = False
    for chunk in client.models.generate_content_stream(
        model=model_name,
        contents=user_message or "(سالک هنوز چیزی ننوشته - گفتگو را باز کن)",
        config=config,
    ):
        if chunk.text:
            got_any = True
            yield chunk.text
    if not got_any:
        raise LLMError("پاسخی از مدل دریافت نشد.")


_QUOTE_PATTERN = re.compile(r"«([^«»]+)»|\"([^\"]+)\"", re.DOTALL)


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def validate_quotes(reply: str, verified_poems: list) -> str:
    """App-level backstop for STRICT QUOTE MODE (persona doc, section 9):
    the prompt tells the model never to fabricate a quote, but we caught it
    doing exactly that in a real session (a fake-sounding line inside «»
    that wasn't in VERIFIED_POEMS). Don't trust the model to police itself -
    strip the quote marks off anything that isn't an exact (whitespace-
    normalized) substring of the poems actually retrieved this turn. The
    underlying words stay - it just stops being presented as a citation."""
    haystack = _normalize_for_match(" ".join(p.text for p in verified_poems)) if verified_poems else ""

    def _check(match: re.Match) -> str:
        quoted = match.group(1) or match.group(2)
        if quoted and _normalize_for_match(quoted) in haystack:
            return match.group(0)
        print(f"[quote validator] stripped unverified quote: {quoted[:80]!r}", file=sys.stderr)
        return quoted

    return _QUOTE_PATTERN.sub(_check, reply)
