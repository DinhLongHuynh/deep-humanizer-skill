#!/usr/bin/env python
"""Build a calibration corpus (JSONL for calibrate.py) from two folders of texts.

    human/   your own writing (pre-2023 or written without AI help), one text per .txt/.md file
    ai/      LLM texts written to the same topics, one per file, with the SAME file name as the
             human text it matches (a suffix _ai is also accepted: post1.md <-> post1_ai.md)

Matched files share a `group`, so calibrate.py never trains and tests on the same topic. Long
texts are cut into paragraph-aligned chunks of about --chunk-words words (the default bank uses
150-400-word documents); chunks keep their file's group. Use --chunk-words 0 to keep whole texts.

Usage:
    python make_corpus.py --human my_writing/ --ai ai_versions/ --register blog --out my_corpus.jsonl
    python calibrate.py my_corpus.jsonl --out ~/.deep-humanizer/banks/blog --cache ~/.deep-humanizer/cache/blog
    python check.py draft.md --reference ~/.deep-humanizer/banks/blog --profile blog
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hz_common import read_text, strip_markup, words  # noqa: E402
import voice_profile as vp  # noqa: E402


def stem(p: Path) -> str:
    return re.sub(r"[_-]ai$", "", p.stem, flags=re.I)


def docs(folder: Path, label: str, register: str, chunk_words: int) -> list[dict]:
    out = []
    for f in vp.sample_files(folder):
        text, _ = vp.normalise_paragraphs(read_text(str(f)))
        parts = vp.chunk_text(text, chunk_words) if chunk_words else [strip_markup(text).strip()]
        for i, part in enumerate(parts):
            if len(words(part)) < 80:
                continue
            out.append({"id": f"{label}_{stem(f)}_{i}", "text": part, "label": label, "group": stem(f),
                        "register": register, "source_file": f.name})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human", required=True, help="folder of your own texts")
    ap.add_argument("--ai", required=True, help="folder of matched LLM texts (same file names)")
    ap.add_argument("--register", default="personal")
    ap.add_argument("--chunk-words", type=int, default=350)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    H = docs(Path(a.human).expanduser(), "human", a.register, a.chunk_words)
    A = docs(Path(a.ai).expanduser(), "ai", a.register, a.chunk_words)
    hg, ag = {d["group"] for d in H}, {d["group"] for d in A}
    rows = H + A
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {"out": a.out, "human_docs": len(H), "ai_docs": len(A), "human_files": len(hg), "ai_files": len(ag),
              "matched_topics": len(hg & ag), "human_only": sorted(hg - ag)[:10], "ai_only": sorted(ag - hg)[:10]}
    warn = []
    if min(len(H), len(A)) < 30:
        warn.append("Fewer than 30 documents in a class: quantiles and AUROCs will be noisy (about 100 per class is "
                    "stable).")
    if len(hg | ag) < 5:
        warn.append("Fewer than 5 topic groups: grouped cross-validation needs more topics.")
    if not hg & ag:
        warn.append("No file names match between the folders: topics are not paired, so the bank may learn topic "
                    "instead of authorship.")
    report["warnings"] = warn
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
