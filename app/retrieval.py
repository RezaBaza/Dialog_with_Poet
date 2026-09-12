"""
Poem retrieval layer: turns (topic, user message) into a short list of
VERIFIED_POEMS the persona prompt is allowed to quote from.

v1 uses BM25 (lexical search) rather than semantic embeddings - no
model download, runs in milliseconds over ~6k chunks, and is easy to
reason about ("did the right words show up") while the persona/prompt
side of the project is still being tuned. Swapping in Qwen3-Embedding
later means replacing PoemIndex.search's internals; callers don't change.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "corpus" / "rumi.jsonl"

# Persian character/diacritic normalization so "كتاب" and "کتاب",
# or a word with/without tashkeel, are treated as the same token.
_ARABIC_DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۭ]")
_CHAR_MAP = str.maketrans({
    "ي": "ی", "ك": "ک", "ة": "ه", "أ": "ا", "إ": "ا", "ؤ": "و",
    "ئ": "ی", "‌": " ", "‏": " ", "‎": " ",
})
_PUNCT = re.compile(r"[^\w؀-ۿ\s]", re.UNICODE)


def normalize_fa(text: str) -> str:
    text = _ARABIC_DIACRITICS.sub("", text)
    text = text.translate(_CHAR_MAP)
    text = _PUNCT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


# Very light Persian affix stripping. Conversational Persian is almost
# always conjugated ("می‌ترسم" = "I fear") while the corpus's own topical
# words appear as bare stems ("ترس" = fear) - without this, BM25 (exact
# token matching) misses obviously relevant poems purely on morphology.
# Not linguistically precise, but applied identically to the corpus at
# index time and to every query, so what matters for matching - internal
# consistency - holds even where a specific stem is technically wrong.
_PREFIXES = ("نمی", "می")
_SUFFIXES = ("های", "هایی", "ترین", "تری", "تر", "اند", "یم", "ید", "ند", "ای", "ها", "ام", "اش", "م", "ی", "د", "ه")


def _stem(word: str) -> str:
    if len(word) <= 3:
        return word
    for pre in _PREFIXES:
        if word.startswith(pre) and len(word) - len(pre) >= 3:
            word = word[len(pre):]
            break
    for suf in _SUFFIXES:  # already longest-first
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            word = word[: -len(suf)]
            break
    return word


def tokenize(text: str) -> list[str]:
    return [_stem(w) for w in normalize_fa(text).split()]


# Rough Persian topic -> vocabulary expansion, seeded from the 23
# self-inquiry topics in Salek_Poet_Agent_Personas_FA (section 10).
# Not exhaustive - a heuristic to help lexical search reach relevant
# couplets even when the topic word itself never appears verbatim.
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "تعلق": ["وابسته", "بند", "دلبستگی", "گرفتار", "اسیر", "دلبسته"],
    "عشق و محبت": ["عشق", "محبت", "عاشق", "معشوق", "دلداده"],
    "ترس": ["ترس", "بیم", "هراس", "خوف", "وحشت"],
    "نیاز به تایید": ["تایید", "نام و ننگ", "ریا", "خودنمایی"],
    "خشم و رنجش": ["خشم", "رنج", "کینه", "غضب", "دلگیر"],
    "مخالفت دیگران": ["خصم", "دشمن", "مخالف", "طعنه", "ملامت"],
    "تنهایی": ["تنها", "فراق", "غربت", "هجران", "جدایی"],
    "مرگ": ["مرگ", "مردن", "اجل", "فنا", "رحلت"],
    "معنای زندگی": ["زندگی", "هستی", "وجود", "جان", "عمر"],
    "شادی و لذت": ["شادی", "سرور", "نشاط", "طرب", "خوشی"],
    "کنترل": ["اختیار", "تدبیر", "تقدیر", "دست", "زمام"],
    "غرور": ["غرور", "تکبر", "خودبینی", "منیت", "نخوت"],
    "انتظار و توقع": ["انتظار", "توقع", "امید", "آرزو", "چشم داشت"],
    "شک و ایمان": ["شک", "یقین", "ایمان", "گمان", "باور"],
    "کارما": ["عمل", "سزا", "جزا", "کشت", "درو"],
    "انتخاب و اختیار": ["اختیار", "انتخاب", "جبر", "راه"],
    "رابطه": ["یار", "دوست", "وصال", "همدم", "رفیق"],
    "بخشش": ["بخشش", "عفو", "گذشت", "کرم"],
    "سکوت و کلام": ["خاموش", "سکوت", "سخن", "کلام", "حرف"],
    "هویت": ["من", "خود", "نفس", "هستی من", "خویشتن"],
    "شکست": ["شکست", "ناکامی", "خطا", "لغزش"],
    "حضور": ["حضور", "اکنون", "دم", "لحظه", "امروز"],
    "تقلید و استقلال فکری": ["تقلید", "پیروی", "استقلال", "خودرای"],
}

MIN_SCORE = 0.5  # BM25 scores below this mean "no real overlap" -> return nothing

# Divan-e Shams is Rumi's most iconic, most quotable work (and what most
# people mean by "a Rumi poem") - a small score boost means it wins ties
# against Masnavi/prose, without ever beating a Masnavi passage that's
# genuinely the stronger match on its own lexical score.
BOOK_BOOST = {"shams": 1.15}


@dataclass
class Poem:
    id: int
    book_key: str
    book_title: str
    is_prose: bool
    title: str
    full_title: str
    url: str
    summary: str
    text: str
    score: float


class PoemIndex:
    def __init__(self, corpus_path: Path = CORPUS_PATH):
        self.records: list[dict] = []
        with corpus_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))
        tokenized = [
            tokenize(f"{r['title']} {r['summary']} {r['text']}")
            for r in self.records
        ]
        self.bm25 = BM25Okapi(tokenized)

    def search(
        self,
        query: str,
        topic: str | None = None,
        top_k: int = 3,
        exclude_ids: set[int] | None = None,
    ) -> list[Poem]:
        exclude_ids = exclude_ids or set()
        query_stems = set(tokenize(query))
        expansion_terms = list(TOPIC_KEYWORDS.get(topic, []))
        if not topic:
            # No topic picked from the dropdown - infer candidates from the
            # message itself, since free-form chat is the common case and
            # the literal topic word (e.g. "ترس") rarely appears in its
            # bare dictionary form inside a real sentence ("می‌ترسم").
            for t, kws in TOPIC_KEYWORDS.items():
                if query_stems & set(tokenize(" ".join([t, *kws]))):
                    expansion_terms.extend(kws)
        tokens = tokenize(query) + tokenize(" ".join(expansion_terms))
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            (
                (score * BOOK_BOOST.get(self.records[i]["book_key"], 1.0), i)
                for i, score in enumerate(scores)
                if self.records[i]["id"] not in exclude_ids
            ),
            reverse=True,
        )
        out: list[Poem] = []
        for score, i in ranked[:top_k]:
            if score < MIN_SCORE:
                break
            r = self.records[i]
            out.append(Poem(score=round(float(score), 2), **{k: r[k] for k in
                ("id", "book_key", "book_title", "is_prose", "title",
                 "full_title", "url", "summary", "text")}))
        return out


if __name__ == "__main__":
    import sys

    idx = PoemIndex()
    q = " ".join(sys.argv[1:]) or "تنهایی و فراق"
    print(f"query: {q}\n")
    for p in idx.search(q, top_k=3):
        print(f"[{p.score}] {p.full_title}  ({p.url})")
        print(p.text[:200].replace("\n", " / "))
        print()
