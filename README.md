# دیالوگ با شاعر — v1 (مولانا)

A Persian self-inquiry chat with Rumi, grounded in his real Divan-e Shams,
Masnavi, Fihi ma Fihi, and Majales-e Sab'e (pulled from the official
[ganjoor/ganjoor-data](https://github.com/ganjoor/ganjoor-data) export).
Full design/scope background is in `project-brief.html`; the persona
engineering is in `Salek_Poet_Agent_Personas_FA.docx`.

## What's actually built

- `pipeline/build_corpus.py` — pulls Rumi's 4 books out of the sparse-cloned
  Ganjoor export and flattens them into `data/corpus/rumi.jsonl` (6,329
  chunks, already committed — you don't need to re-run this unless you're
  adding a poet or the source data changes).
- `app/retrieval.py` — BM25 lexical search over that corpus, with a Persian
  topic → keyword expansion table so retrieval finds relevant couplets even
  when the abstract topic word itself never appears. `MIN_SCORE` gates out
  poems that aren't genuinely relevant, so most turns correctly surface none.
- `prompts/` — the CORE system prompt and all five poet personas from
  `Salek_Poet_Agent_Personas_FA.docx`, with four gaps patched in: an explicit
  "always reply in Persian" rule, a written self-harm safety response, a
  user-facing disclosure string, and Khayyam's attribution-uncertainty note.
- `app/llm.py` — composes CORE + PERSONA + RUNTIME + VERIFIED_POEMS exactly
  per the persona doc's formula, and calls Gemini (free tier by default).
- `app/storage.py` — one JSON file per (friend, poet) so a conversation
  picks back up correctly tomorrow. This is the seam for swapping in
  Supabase later (see the project brief, §09) — nothing else in the app
  would need to change.
- `app/app.py` — the Streamlit chat UI: name picker, topic list from the
  persona doc, RTL layout, disclosure banner, an expander showing which
  real poems were available for each reply.

Only Rumi is wired up (`ACTIVE_POETS` in `app/llm.py`). The other four
personas are fully drafted and ready — adding one is a data-pull + corpus
-build step, not new code (see "Adding a poet" below).

## What was simplified for v1 — and why

- **BM25, not semantic embeddings.** No model download, runs in
  milliseconds, easy to debug. The project brief's Qwen3-Embedding
  suggestion is a drop-in upgrade later if lexical search starts missing
  things a semantic search would catch.
- **Plain-text conversation summary, not the persona doc's full structured
  memory** (CURRENT_TOPIC / USER_EXPERIENCES / OPEN_HYPOTHESES / ...). v1
  keeps the last 16 turns verbatim - one LLM call per message instead of
  two. Worth upgrading once conversations regularly run long enough that
  raw history gets unwieldy.
- **JSON files, not Supabase**, for persistence — genuinely works (a
  conversation survives a restart), but only on a host with a real
  persistent disk. Streamlit Community Cloud's disk isn't guaranteed to
  survive a redeploy - that's what the Supabase swap in the brief closes.

## Setup

1. **Get a free Gemini API key**: [aistudio.google.com](https://aistudio.google.com/app/apikey) → "Create API key". No card required.
2. Copy `.env.example` to `.env` and paste the key in.
3. Install dependencies (whichever Python environment you'll run this with):
   ```
   pip install -r requirements.txt
   ```
4. Run it:
   ```
   streamlit run app/app.py
   ```

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (`vendor/` and `.env` are gitignored on purpose —
   `data/corpus/rumi.jsonl` is already committed, so Streamlit Cloud doesn't
   need the Ganjoor clone at all).
2. On share.streamlit.io, point a new app at this repo, main file `app/app.py`.
3. In the app's **Settings → Secrets**, add:
   ```
   GEMINI_API_KEY = "your_key_here"
   ```
4. Set the app to **private** with an email allowlist if you want it
   restricted to your friends (Community Cloud's free tier supports one
   private app).

## Adding a poet (Hafez, Saadi, Attar, or Khayyam)

Their personas are already in `prompts/`. To bring one online:

1. Sparse-checkout their folder from `ganjoor-data` the same way Rumi's was
   pulled (see git history / the commands used for `poets/moulavi`), for
   the specific books discussed in the project brief §12 — e.g. for Attar:
   Divān, Manteq al-Tayr, Mosibat-Nāmeh, Tazkirat al-Awliyā.
2. Copy `pipeline/build_corpus.py`, adjust `BOOKS` and the output filename
   for that poet, run it.
3. Add the poet to `ACTIVE_POETS` in `app/llm.py`, and give `app/retrieval.py`
   a way to pick the right corpus file per poet (right now it's hardcoded
   to `rumi.jsonl` — this is the one real code change needed, everything
   else was written generically).
4. For Khayyam specifically: tag the Mirafzali/Hedayat quatrain sources
   with a different confidence than the plain Robā'iyāt set (brief, §12) —
   the persona prompt already expects and uses that distinction.

## Known gaps before sharing with friends

- No login/auth beyond a name text field — fine for 5-6 trusted friends,
  not beyond that.
- The self-harm safety response in `prompts/core_system_prompt.txt`
  includes an Iranian emergency-line number (اورژانس اجتماعی ۱۲۳) that
  should be double-checked against a current, authoritative source before
  this goes in front of anyone.
- No spend cap set on the Gemini account yet — worth doing before wider use
  (project brief §11).
