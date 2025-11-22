"""
normalize_text.py
~~~~~~~~~~~~~~~~~
Final stage of transcript normalization for ASR training:
• Remove punctuation and symbols (configurable keep set)
• Collapse multiple whitespaces
• Lowercase non-tag text (optional)
• Optionally keep CS markage underscores
• KEEP <TAG> SPANS EXACTLY AS-IS (never lowercased/altered)

Usage:
    python normalize_text.py manifest.jsonl [-o out.jsonl] [--lower] [--keep-cs]
"""

import re
import json
import argparse
from pathlib import Path
from typing import Iterable

import bambara_normalizer as bm_norm


TAG_RE = re.compile(r"<[^<>]+?>")  # match any <...> tag once previous steps normalized tags


def _process_chunk(chunk: str, normalizer: bm_norm.BambaraASRNormalizer,
                   do_lower_case: bool, keep_cs_markage: bool) -> str:
    """Normalize a chunk that is GUARANTEED to contain no <TAG>."""
    # Build keep set from normalizer defaults
    keep = normalizer.remove_symbols.__defaults__[0]
    # We do NOT add '<>' here because tags are stripped out before normalization
    if keep_cs_markage:
        keep += "_"

    chunk = normalizer.remove_symbols(s=chunk, keep=keep)
    chunk = re.sub(r"\s+", " ", chunk)
    if do_lower_case:
        chunk = chunk.lower()
    return chunk


def normalize_text(text: str, do_lower_case: bool = False,
                   keep_cs_markage: bool = False) -> str:
    if not text:
        return ""

    normalizer = bm_norm.BambaraASRNormalizer()

    out_parts = []
    last = 0
    for m in TAG_RE.finditer(text):
        # Process the non-tag text before this tag
        if m.start() > last:
            out_parts.append(
                _process_chunk(text[last:m.start()], normalizer, do_lower_case, keep_cs_markage)
            )
        # Append the tag EXACTLY as-is
        out_parts.append(m.group(0))
        last = m.end()
    # Tail after last tag
    if last < len(text):
        out_parts.append(
            _process_chunk(text[last:], normalizer, do_lower_case, keep_cs_markage)
        )

    # Join and clean up spacing around tags
    normalized = "".join(out_parts)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def iter_manifest(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_manifest(records: Iterable[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Final text normalization step for Bambara ASR data"
    )
    parser.add_argument("--manifest", type=Path, help="Input JSONL manifest (expects a 'text' field)")
    parser.add_argument("-o", "--output", type=Path, help="Output manifest path (default: same path)")
    parser.add_argument("--text-key", type=str, default="text",
                        help="Key for text field in manifest (default: 'text')")
    parser.add_argument("--lower", action="store_true", help="Lowercase non-tag text")
    parser.add_argument("--keep-cs", action="store_true", help="Keep code-switch markers (underscores)")
    args = parser.parse_args()

    in_path = args.manifest
    out_path = args.output or in_path
    text_field = args.text_key

    processed = []
    for rec in iter_manifest(in_path):
        txt = rec.get(text_field)
        rec["text"] = normalize_text(txt, do_lower_case=args.lower, keep_cs_markage=args.keep_cs)
        processed.append(rec)

    write_manifest(processed, out_path)
    print(f"✅ Normalized manifest → {out_path}")

if __name__ == "__main__":
    main()
