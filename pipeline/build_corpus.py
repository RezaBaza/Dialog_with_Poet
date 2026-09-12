"""
Pull Rumi's four Ganjoor books (Divan-e Shams, Masnavi, Fihi ma Fihi,
Majales-e Sab'e) out of the sparse-checked-out ganjoor-data export and
flatten them into one clean JSONL corpus: data/corpus/rumi.jsonl

Each line is one retrievable chunk = one Ganjoor poem/section file,
with its display text reconstructed from the Verses array (hemistich
pairs for poetry, paragraphs for prose) and its real ganjoor.net URL
kept for citation.

Run: python pipeline/build_corpus.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POET_DIR = ROOT / "vendor" / "ganjoor-data" / "poets" / "moulavi"
OUT_PATH = ROOT / "data" / "corpus" / "rumi.jsonl"

BOOKS = {
    "shams": "دیوان شمس",
    "masnavi": "مثنوی معنوی",
    "fhmfh": "فیه ما فیه",
    "7m": "مجالس سبعه",
}

# Fihi ma Fihi and Majales-e Sab'e are Rumi's own prose (discourses /
# sermons) rather than verse - useful downstream if the app ever wants
# to weight or label prose differently from poetry.
PROSE_BOOKS = {"fhmfh", "7m"}


def reconstruct_text(verses: list[dict]) -> tuple[str, bool]:
    """Group verses by CoupletIndex (preserving order), join each group's
    lines, then join groups. Returns (text, is_prose)."""
    groups: dict[int, list[str]] = {}
    order: list[int] = []
    is_prose = False
    for v in verses:
        idx = v.get("CoupletIndex", v.get("VOrder"))
        if idx not in groups:
            groups[idx] = []
            order.append(idx)
        groups[idx].append(v.get("Text", "").strip())
        if v.get("Position") == "Paragraph":
            is_prose = True
    blocks = ["\n".join(t for t in groups[idx] if t) for idx in order]
    sep = "\n\n" if is_prose else "\n"
    return sep.join(b for b in blocks if b), is_prose


def load_poem(path: Path, book_key: str, book_title: str) -> dict | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    verses = d.get("Verses") or []
    if not verses:
        return None
    text, is_prose = reconstruct_text(verses)
    if not text.strip():
        return None
    full_url = d.get("FullUrl", "")
    return {
        "id": d.get("Id"),
        "book_key": book_key,
        "book_title": book_title,
        "is_prose": is_prose,
        "title": d.get("Title", ""),
        "full_title": d.get("FullTitle", ""),
        "url": f"https://ganjoor.net{full_url}" if full_url else "",
        "summary": d.get("PoemSummary", "") or "",
        "text": text,
        "n_verses": len(verses),
    }


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for book_key, book_title in BOOKS.items():
        book_dir = POET_DIR / book_key
        if not book_dir.exists():
            print(f"!! missing book dir: {book_dir}")
            continue
        files = [p for p in book_dir.rglob("*.json") if p.name != "_cat.json"]
        n_ok = 0
        for p in files:
            rec = load_poem(p, book_key, book_title)
            if rec:
                records.append(rec)
                n_ok += 1
        print(f"{book_key} ({book_title}): {n_ok}/{len(files)} chunks")

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(records)} chunks -> {OUT_PATH}")


if __name__ == "__main__":
    main()
