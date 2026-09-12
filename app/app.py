"""
دیالوگ با شاعر - نسخه‌ی اول (فقط مولانا)
Run: streamlit run app/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import storage
from app.llm import (
    ACTIVE_POETS,
    POETS,
    build_system_prompt,
    friendly_error_message,
    generate_stream,
    validate_quotes,
)
from app.retrieval import PoemIndex

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Streamlit secrets -> env, so llm.py's os.environ.get keeps working either way.
# Locally there's no secrets.toml at all (you're using .env instead) - st.secrets
# raises in that case rather than acting like an empty dict, so this is expected
# to no-op on your machine and only actually do something once deployed with
# secrets configured on Streamlit Community Cloud.
# This used to copy only GEMINI_API_KEY - a leftover from before Claude and
# OpenAI existed as providers - so an OpenAI/Claude secret on Streamlit Cloud
# was silently never picked up. Now copies every key llm.py's _provider()
# actually checks.
_SECRET_KEYS = (
    "GEMINI_API_KEY", "GEMINI_MODEL",
    "CLAUDE_API_KEY", "ANTHROPIC_API_KEY", "CLAUDE_MODEL",
    "OPENAI_API_KEY", "OPENAI_MODEL",
)
try:
    for _k in _SECRET_KEYS:
        if _k in st.secrets:
            os.environ.setdefault(_k, st.secrets[_k])
except Exception:
    pass

TOPICS = [
    "تعلق", "عشق و محبت", "ترس", "نیاز به تایید", "خشم و رنجش",
    "مخالفت دیگران", "تنهایی", "مرگ", "معنای زندگی", "شادی و لذت",
    "کنترل", "غرور", "انتظار و توقع", "شک و ایمان", "کارما",
    "انتخاب و اختیار", "رابطه", "بخشش", "سکوت و کلام", "هویت",
    "شکست", "حضور", "تقلید و استقلال فکری",
]

st.set_page_config(page_title="دیالوگ با شاعر", page_icon="🪶", layout="centered")

st.html(
    """<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700&display=swap">
<style>
/* Font: scoped to actual text containers only - NOT div/span broadly.
   Streamlit's chat-message avatars and some widget icons are rendered
   via a ligature icon font (a span containing literal text like "face"
   that the icon font turns into a glyph); an earlier, broader version of
   this rule caught those spans too and broke the ligature, which is why
   the avatar icons showed as raw text like "face" / "art_..." instead of
   icons. [data-testid="stWidgetLabel"] is the actual wrapper Streamlit
   puts widget labels (like "اسمت چیه؟") in - plain "label" selectors
   were missing it, which is why those stayed in the default font. */
html, body, .stApp,
.stMarkdown, .stMarkdown p, .stMarkdown li,
.stChatMessage [data-testid="stMarkdownContainer"],
.stChatMessage [data-testid="stMarkdownContainer"] p,
.stChatMessage [data-testid="stMarkdownContainer"] li,
.stCaption, h1, h2, h3, h4,
.stButton button, .stTextInput input,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
div[data-testid="stChatInput"] textarea {
  font-family: 'Vazirmatn', Tahoma, sans-serif !important;
}

/* Only the reading content flows RTL - Streamlit's own widget chrome
   (dropdown arrows, buttons) is left in its normal layout direction on
   purpose. Flipping the whole .stApp direction was tried first and broke
   things because Streamlit's internal flex layouts assume LTR. */
.stChatMessage [data-testid="stMarkdownContainer"],
.stChatMessage [data-testid="stMarkdownContainer"] p,
.stChatMessage [data-testid="stMarkdownContainer"] li,
.stMarkdown p, .stMarkdown li,
.stCaption, h1, h2, h3, h4,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
div[data-testid="stChatInput"] textarea,
.stTextInput input,
.disclosure {
  direction: rtl;
  text-align: right;
}

.disclosure {
  font-size: 0.8rem; color: #8d8267; border-top: 1px solid #ddd0a8;
  padding-top: 8px; margin-top: 18px;
}

/* Sidebar on the right, attempt #2. Attempt #1 flipped .stApp's direction
   wholesale and broke icon ligatures + widget internals. This time nothing's
   "direction" changes at all - just the visual ORDER of the sidebar vs. the
   main content within their flex container, plus a position override in
   case the sidebar is fixed-positioned rather than a flex child in this
   Streamlit version. st.sidebar itself is untouched functionally, so
   st.chat_input's pinned-to-bottom behavior (broken by the st.columns
   experiment before) isn't at risk here.

   Confirmed working on desktop, but forcing flex-direction: row
   unconditionally broke mobile: Streamlit's own responsive layout normally
   stacks the sidebar as a collapsible drawer below/above the main content
   on narrow screens, and overriding that to a side-by-side row squeezed
   both panes so tight that text wrapped one character per line. Scoping
   this to wider screens only, so mobile keeps Streamlit's native stacking. */
@media (min-width: 768px) {
  [data-testid="stAppViewContainer"] { display: flex; flex-direction: row; }
  [data-testid="stSidebar"] { order: 2; right: 0 !important; left: auto !important; }
  [data-testid="stMain"], section.main { order: 1; }
}
</style>"""
)

# Ganjoor's own portrait illustration for Rumi - the same image used on
# ganjoor.net/moulavi - rather than a generic robot emoji for the assistant
# avatar. Note the domain is api.ganjoor.net, not ganjoor.net; the latter
# 404s even though it's what the poet metadata's ImageUrl field implies.
POET_AVATARS = {
    "molana": "https://api.ganjoor.net/api/ganjoor/poet/image/moulavi.gif",
}


@st.cache_resource
def get_index() -> PoemIndex:
    return PoemIndex()


def poet_label(key: str) -> str:
    suffix = "" if key in ACTIVE_POETS else "  (به‌زودی)"
    return POETS[key]["title"] + suffix


# Settings panel is back in st.sidebar. The right-side-via-st.columns
# experiment caused two real bugs, not just a cosmetic miss: st.chat_input
# only reliably keeps its pinned-to-bottom behavior across reruns when
# it's at the top level of the script or inside st.sidebar - nested in
# st.columns it isn't a well-supported combination, which is almost
# certainly why the input box drifted up/down and why a reply went missing
# after the second message. Sidebar means the panel sits on the left
# again; revisiting a right-side panel is worth doing later as a narrow,
# separately-tested CSS change, not bundled with everything else.
with st.sidebar:
    st.markdown("### دیالوگ با شاعر")
    friend_name = st.text_input("اسمت چیه؟", value=st.session_state.get("friend_name", ""))
    poet_key = st.selectbox(
        "شاعر", options=list(POETS.keys()), format_func=poet_label,
        index=list(POETS.keys()).index("molana"),
        disabled=not friend_name,
    )
    if poet_key not in ACTIVE_POETS:
        st.info("این شاعر هنوز در نسخه‌ی اول فعال نیست - فعلا فقط مولانا.")
    topic = st.selectbox("موضوع (اختیاری)", options=["(آزاد)"] + TOPICS)
    if friend_name and st.button("شروع گفتگوی تازه"):
        # Same name, same poet, but a clean slate - resets both this
        # session's copy and the saved file, so coming back later with the
        # same name doesn't drag in a conversation you meant to leave behind.
        st.session_state[f"state::{friend_name}::{poet_key}"] = storage.reset_state(friend_name, poet_key)
        st.rerun()
    st.markdown(
        '<div class="disclosure">این یک شبیه‌سازی هوش‌مصنوعی است، نه خودِ شاعر. '
        "پاسخ‌ها از دل آثار واقعی او ساخته می‌شوند، اما جای مشاوره‌ی روان‌شناختی "
        "یا پزشکی را نمی‌گیرند.</div>",
        unsafe_allow_html=True,
    )

if not friend_name:
    st.title("دیالوگ با شاعر 🪶")
    st.write("برای شروع، اسمت رو توی پنل کناری بنویس.")
    st.stop()

st.session_state["friend_name"] = friend_name

if poet_key not in ACTIVE_POETS:
    st.title(f"گفت‌وگو با {POETS[poet_key]['title']}")
    st.warning("این شاعر هنوز آماده نیست. فعلا مولانا را انتخاب کن.")
    st.stop()

state_key = f"state::{friend_name}::{poet_key}"
if state_key not in st.session_state:
    st.session_state[state_key] = storage.load_state(friend_name, poet_key)
state = st.session_state[state_key]

if topic != "(آزاد)":
    state["current_topic"] = topic

st.title(f"گفت‌وگو با {POETS[poet_key]['title']}")

poet_avatar = POET_AVATARS.get(poet_key)
for turn in state["transcript"]:
    role = "user" if turn["role"] == "user" else "assistant"
    avatar = poet_avatar if role == "assistant" else None
    with st.chat_message(role, avatar=avatar):
        st.write(turn["text"])

if not state["transcript"]:
    st.caption("گفتگو رو با یه پیام شروع کن، یا خالی بفرست تا خودش شروع کنه.")

user_message = st.chat_input("پیامت به مولانا...")

if user_message is not None:
    if user_message.strip():
        state["transcript"].append({"role": "user", "text": user_message})
        with st.chat_message("user"):
            st.write(user_message)

    with st.chat_message("assistant", avatar=poet_avatar):
        with st.spinner("..."):
            index = get_index()
            query = user_message or (state.get("current_topic") or "")
            poems = index.search(
                query,
                topic=state.get("current_topic"),
                top_k=3,
                exclude_ids=set(state["used_poem_ids"]),
            )
            system_prompt = build_system_prompt(
                poet_key=poet_key,
                topic=state.get("current_topic") or "",
                conversation_summary=storage.summarize_transcript(state["transcript"]),
                last_open_question=state.get("last_open_question") or "",
                user_message=user_message or "",
                verified_poems=poems,
            )
        # Manual streaming loop instead of st.write_stream: when the model
        # call fails before yielding anything (a 429, a bad key, ...),
        # st.write_stream appears to swallow that rather than re-raising it
        # to our except block - the symptom was a blank assistant bubble
        # forever, with the real error never shown even though our error
        # handling below was already correct. Doing our own accumulation
        # with st.empty() means we're never depending on how write_stream
        # handles a failure internally. The placeholder text also means
        # something visible appears immediately, instead of nothing at all
        # during the (sometimes 10-40s) wait for the first token.
        placeholder = st.empty()
        placeholder.markdown("...")
        reply = ""
        failed = False
        try:
            for chunk in generate_stream(system_prompt, user_message or ""):
                reply += chunk.replace("\n", "  \n")
                placeholder.markdown(reply + " ▌")
            # App-level check, not just prompt-level: strip the quote marks
            # off anything that isn't an exact match in the poems retrieved
            # this turn, so a fabricated "quote" can never slip through even
            # if the model's own anti-fabrication instructions fail - which
            # we saw happen once already in a real conversation.
            checked = validate_quotes(reply, poems)
            if checked != reply:
                reply = checked
                placeholder.markdown(reply)
        except Exception as e:  # network/quota/model error - never crash the chat
            failed = True
            reply = f"⚠️ {friendly_error_message(e)}"
            placeholder.markdown(reply)

        if poems and not failed:
            with st.expander("شعرهای مرتبط که در این پاسخ در دسترس بودند"):
                for p in poems:
                    st.markdown(f"**{p.full_title}** — [مشاهده در گنجور]({p.url})")
                    st.caption(p.text.replace("\n", "  \n"))

    # A failed attempt isn't real conversation - don't save it into the
    # poet's memory (it would otherwise get replayed back as context on the
    # next turn) or into used_poem_ids (nothing was actually quoted). The
    # user's own message above is already saved either way. Also skip the
    # rerun on failure: st.rerun() immediately starts a fresh script run,
    # which would clear the just-shown error before anyone could read it,
    # since it was never added to state["transcript"].
    if not failed:
        state["transcript"].append({"role": "poet", "text": reply})
        state["used_poem_ids"].extend(p.id for p in poems)
        storage.save_state(friend_name, poet_key, state)
        st.rerun()
