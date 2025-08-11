"""
normalize_numbers.py
~~~~~~~~~~~~~~~~~~~~
Adds **numeric normalisation** on top of the cleaned transcripts.

Rules implemented (short recap)
------------------------------
* *Language Identification*: text inside a matched pair of double underscores `__` is
  considered **French**; everything else **Bambara**.  Markers can frame a full
  phrase, so we segment on `__(.*?)__` boundaries and treat each slice
  independently.

* *Bambara segment*
  * Thousands separator Eg. 70.000 → 70000 (remove dots when >1 digits after)  
  * x.y with **one** decimal digit (e.g. 102.3) → `number_to_bambara(x)` +
    " tomi " + `number_to_bambara(y)`.
  * Integer → `number_to_bambara(n)`.
  * Leading‑zero numbers (e.g. 01): pronounce each digit as an individual
    digit joined by " ni " (fu ni kelen …).

* *French segment*
  * Same thousands‑dot removal (70.000 → 70000 / 35.000.000 -> 35000000).
  * Integer → spelled out in French words (hyphen rules).  Uses `num2words` if
    available; otherwise a light custom converter up to 999 999.
  * Leading‑zero numbers: spell each digit (zéro un …).

* Extra hygiene*
  * Numbers glued to text or punctuation get temporary spaces inserted to make
    them capturable (e.g. `__12heures__` → `__ 12 heures __`).  Spaces are
    collapsed at the end.

Outputs
-------
Re‑writes the manifest with normalised numbers and regenerates.

Usage::
    pip install num2words

    python normalize_numbers.py cleaned.step3.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, List

# Try to leverage num2words for French – fall back to custom
try:
    from num2words import num2words  # type: ignore

    def number_to_french(n: int) -> str:
        return num2words(n, lang="fr")
except ModuleNotFoundError:  # minimal fallback

    def _below_hundred_fr(n: int) -> str:
        basics = [
            "zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf",
            "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize",
        ]
        dixaines = [
            "", "dix", "vingt", "trente", "quarante", "cinquante", "soixante",
            "soixante‑dix", "quatre‑vingt", "quatre‑vingt‑dix",
        ]
        if n < 17:
            return basics[n]
        if n < 20:
            return "dix‑" + basics[n - 10]
        if n < 70:
            d, u = divmod(n, 10)
            if u == 1 and d not in {8}:  # vingt‑et‑un, trente‑et‑un …
                return dixaines[d] + "‑et‑un"
            return dixaines[d] + ("‑" + basics[u] if u else "")
        if n < 80:
            return "soixante‑" + _below_hundred_fr(n - 60)
        if n < 100:
            return "quatre‑vingt" + ("‑" + basics[n - 80] if n != 80 else "s")
        raise ValueError

    def number_to_french(n: int) -> str:
        if n < 100:
            return _below_hundred_fr(n)
        if n < 1000:
            c, r = divmod(n, 100)
            head = "cent" if c == 1 else _below_hundred_fr(c) + " cent"
            if r == 0:
                return head + ("s" if c > 1 else "")
            return head + " " + number_to_french(r)
        if n < 1_000_000:
            th, r = divmod(n, 1000)
            head = "mille" if th == 1 else number_to_french(th) + " mille"
            return head + (" " + number_to_french(r) if r else "")
        # Simplistic million handling
        mil, r = divmod(n, 1_000_000)
        head = "un million" if mil == 1 else number_to_french(mil) + " millions"
        return head + (" " + number_to_french(r) if r else "")

# Bambara converter provided by user (slightly adjusted for 0‑digit list spelling)
UNITS_BAM = [
    "fu", "kelen", "fila", "saba", "naani", "duuru", "wɔɔrɔ", "wolonwula", "seegin", "kɔnɔntɔn",
]


def number_to_bambara(n: int) -> str:
    """Recursive Bambara number to words (supports up to millions)."""
    units = UNITS_BAM
    tens = [
        "", "tan", "mugan", "bi saba", "bi naani", "bi duuru", "bi wɔɔrɔ", "bi wolonfila", "bi seegin", "bi kɔnɔntɔn",
    ]
    if n == 0:
        return "fu"
    if n < 10:
        return units[n]
    if n < 100:
        if n < 20:
            return tens[n // 10]
        return tens[n // 10] + (" ni " + number_to_bambara(n % 10) if n % 10 else "")
    if n < 1000:
        prefix = "kɛmɛ" if n < 200 else "kɛmɛ " + number_to_bambara(n // 100)
        return prefix + (" ni " + number_to_bambara(n % 100) if n % 100 else "")
    if n < 1_000_000:
        return "waa " + number_to_bambara(n // 1000) + (" ni " + number_to_bambara(n % 1000) if n % 1000 else "")
    return "milyɔn " + number_to_bambara(n // 1_000_000) + (" ni " + number_to_bambara(n % 1_000_000) if n % 1_000_000 else "")

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------
RE_CS_BLOCK = re.compile(r"__.*?__", re.S)
RE_NUMBER   = re.compile(r"\d+(?:\.\d+)?")
RE_TAG     = re.compile(r"<([^<>\s]+?)>")

# ---------------------------------------------------------------------------
# Normalisation per segment
# ---------------------------------------------------------------------------

def normalize_number_token(token: str, lang: str) -> str:
    """Return the normalised form of a numeric token for *lang*.

    * Detects radio frequencies (one decimal digit).
    * Collapses multi‑dot thousands separators.
    * Handles leading‑zero sequences.
    * Delegates to full converters for everything else.
    """
    # --- radio frequency ---------------------------------------------------
    if token.count(".") == 1:
        left, right = token.split(".")
        if len(right) == 1:
            if lang == "bam":
                return f"{number_to_bambara(int(left))} tomi {number_to_bambara(int(right))}"
            return f"{number_to_french(int(left))} point {number_to_french(int(right))}"

    # --- thousands separators ---------------------------------------------
    if "." in token:
        parts = token.split(".")
        if all(len(p) == 3 for p in parts[1:]):
            token = "".join(parts)  # e.g., 35.000.000 → 35000000
        else:
            # rare fallback – just remove first dot
            token = parts[0] + "".join(parts[1:])

    # --- leading zeros -----------------------------------------------------
    if token.startswith("0") and len(token) > 1:
        digits = [int(d) for d in token]
        if lang == "bam":
            return " ni ".join(UNITS_BAM[d] for d in digits)
        return " ".join(number_to_french(d) for d in digits)

    # --- plain integer -----------------------------------------------------
    n = int(token)
    return number_to_bambara(n) if lang == "bam" else number_to_french(n)


def normalize_segment(seg: str, lang: str) -> str:

    def _replace(match: re.Match) -> str:
        return normalize_number_token(match.group(0), lang)

    seg = RE_NUMBER.sub(_replace, seg)
    seg = re.sub(r"\s{2,}", " ", seg).strip()
    return seg


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------

def normalise_numbers(text: str) -> str:
    if not text or not RE_NUMBER.search(text):
        return text

    out_parts: List[str] = []
    last_index = 0
    for match in RE_CS_BLOCK.finditer(text):
        # Outside block = Bambara
        outside = text[last_index:match.start()]
        normalized = normalize_segment(outside, "bam")
        print(f"Normalising Bambara Block: {outside!r} --> {normalized!r}")
        out_parts.append(normalized)
        # Inside block = French (keep markers)
        content = match.group(0)
        inner = content.strip("_")  # remove leading/trailing underscores
        normalized_inner = normalize_segment(inner, "fr")
        print(f"Normalising French Block: {inner!r} → {normalized_inner!r}")
        out_parts.append(f"__{normalized_inner}__")
        last_index = match.end()
    # tail
    normalized = normalize_segment(text[last_index:], 'bam')
    print(f"Normalising Bambara Block: {text[last_index:]!r} --> {normalized!r}")
    out_parts.append(normalized)

    result = " ".join(part for part in out_parts if part)
    return re.sub(r"\s{2,}", " ", result).strip()

# ---------------------------------------------------------------------------
# I/O helpers (reuse pattern)
# ---------------------------------------------------------------------------

def iter_manifest(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def write_manifest(records: Iterable[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Normalise digits into Bambara / French words depending on CS markers")
    p.add_argument("--manifest", type=Path, help="Input cleaned manifest")
    p.add_argument("--text-key", type=str, default="text", help="Key for text field (default: 'text')")
    p.add_argument("-o", "--output", type=Path, help="Output manifest (default: <input>)")

    args = p.parse_args()

    in_path: Path = args.manifest
    out_path: Path = args.output or in_path
    text_key = args.text_key

    processed_records = []
    for rec in iter_manifest(in_path):
        txt = rec.get(text_key)
        norm = normalise_numbers(txt)
        rec = {
            "audio_filepath": rec.get("audio_filepath"),
            "duration": rec.get("duration"),
            "text": norm,
        }

        processed_records.append(rec)


    write_manifest(processed_records, out_path)

    print(f"✅ Number‑normalised manifest → {out_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
