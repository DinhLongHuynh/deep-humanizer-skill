#!/usr/bin/env python
"""Layer 1 - surface tells.

Counts lexical, phrasal, punctuation and formatting habits that are
over-represented in instruction-tuned LLM prose, normalised per 1,000 words,
and returns the exact spans so the rewrite can target them.

These are the cheapest signals and the least reliable on their own: a
careful human can produce any of them, and removing all of them does not
move probability- or embedding-level detectors much (see
references/research_digest.md). Use them as a to-do list, not as a verdict.

Usage:
    python surface_scan.py draft.md [--json out.json] [--vocab ../assets/ai_vocabulary.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hz_common import ASSETS, dump_json, paragraphs, read_text, split_sentences, strip_markup, words  # noqa: E402

# (family, weight, regex). Weight 3 = act on a single sighting, 2 = strong in
# company, 1 = weak alone. Families mirror references/surface_patterns.md.
PATTERNS: list[tuple[str, int, str]] = [
    # staging instead of stating
    ("not_x_but_y", 3, r"\bnot (?:just|only|merely|simply|about)\b[^.?!\n]{0,90}?\bbut\b"),
    ("not_x_but_y", 3, r"\b(?:it|this|that)(?:'s| is) not\b[^.?!\n]{0,70}?[,;:—–]\s*(?:it|this|that)(?:'s| is)\b"),
    ("not_x_but_y", 3, r"\b(?:does not|doesn't|is not|isn't) (?:mean|about)\b[^.?!\n]{0,90}?[.;]\s*(?:It|This)(?:'s| is| means)\b"),
    ("staged_opener", 3, r"(?:^|(?<=[.!?]\s))(?:Let's (?:dive|explore|break|unpack|take a (?:closer )?look)|Here's (?:the thing|what you need|why)|The thing is|Without further ado|Honestly\?|Real talk|Let's be honest)"),
    ("deep_saying", 3, r"\b(?:at its (?:core|heart)|the real (?:question|issue|story)|what (?:really|truly) matters|the heart of the matter|the deeper (?:issue|question)|is the (?:language|currency|architecture) of)\b"),
    ("arguing_with_no_one", 2, r"\b(?:To be clear,|Don't get me wrong|This is not to say|I'm not saying|One might be tempted|A tempting approach would be|You might think)\b"),
    ("closer_line", 3, r"(?:^|\n)\s*(?:That(?:'s| is) (?:the real (?:win|point|story)|what matters|the point)|Let that sink in|Read that again|And that changes everything|The message (?:was|is) clear)[.!]?\s*(?:\n|$)"),
    # chat and draft leftovers
    ("chatbot_residue", 3, r"\b(?:I hope this helps|Great question|Certainly!|Of course!|Absolutely!|I'd be happy to|Feel free to|Let me know if|Would you like me to|As an AI|Here is (?:a|an|the) (?:revised|rewritten|summary|overview))\b"),
    ("knowledge_limit", 3, r"\b(?:as of my (?:last|latest) (?:update|training)|based on (?:the )?available information|not (?:widely|extensively) documented|while specific details)\b"),
    # inflation
    ("inflated_significance", 2, r"\b(?:stands? as a testament|(?:a )?testament to|pivotal (?:role|moment)|plays? an? (?:crucial|key|pivotal|vital|significant) role|underscor(?:es|ing) the importance|reflects? a broader|(?:lasting|enduring) legacy|setting the stage|evolving landscape|indelible mark|continues? to thrive|rich (?:cultural )?(?:heritage|tapestry|history)|in today's (?:fast-paced|digital|ever-changing))\b"),
    ("ing_rider", 2, r",\s+(?:highlighting|underscoring|emphasi[sz]ing|ensuring|reflecting|symboli[sz]ing|showcasing|fostering|cementing|solidifying|demonstrating|illustrating|paving the way|contributing to)\b"),
    ("copula_avoidance", 1, r"\b(?:serves|served|stands|stood|functions|acts) as (?:a|an|the)\b|\bboasts\b"),
    ("signpost_transition", 1, r"(?:^|(?<=[.!?]\s))(?:Additionally|Furthermore|Moreover|In conclusion|Overall|Ultimately|Importantly|Notably|In summary|Crucially|Consequently),"),
    ("hedge_stack", 1, r"\b(?:could|might|may) potentially\b|\bit(?:'s| is) (?:important|worth|crucial) (?:to note|noting)\b"),
    ("vague_authority", 2, r"\b(?:experts|researchers|critics|observers|studies) (?:say|argue|believe|suggest|note|have noted)\b"),
    ("sales_language", 2, r"\b(?:nestled|breathtaking|must-visit|stunning|in the heart of|diverse array|seamless(?:ly)?|cutting-edge|game-chang(?:er|ing)|unlock(?:s|ing)? the (?:power|potential))\b"),
]

FORMAT_PATTERNS: list[tuple[str, int, str]] = [
    ("bold_label_list", 2, r"(?m)^\s*(?:[-*+]|\d+[.)])\s+\*\*[^*\n]+\*\*\s*[:\-–—]"),
    ("emoji_or_arrow", 2, "[\u2192\u27a1\u2705\u2728\U0001F300-\U0001FAFF]"),
    ("title_case_heading", 1, r"(?m)^#{1,6}\s+(?:[A-Z][a-z]+\s+){2,}[A-Z][a-z]+\s*$"),
]

FALLBACK_VOCAB = [
    "delve", "intricate", "tapestry", "testament", "underscore", "pivotal", "meticulous", "showcase",
    "commendable", "realm", "landscape", "notably", "garner", "bolster", "multifaceted", "nuanced",
    "embark", "foster", "unwavering", "seamless", "holistic", "leverage", "paramount", "robust",
    "comprehensive", "crucial", "enhance", "vibrant", "invaluable", "noteworthy",
]


def load_vocab(path: str | None) -> list[str]:
    """Single words and phrases marked as over-represented in LLM text.
    Accepts the research file (dict with 'items'), a {'terms': [...]} dict or a plain list."""
    p = Path(path) if path else ASSETS / "ai_vocabulary.json"
    if not p.exists():
        return FALLBACK_VOCAB
    data = json.loads(p.read_text())
    if isinstance(data, dict) and "items" in data:
        n_src = {t: d.get("n_sources", 1) for t, d in data.get("by_term", {}).items()}
        # common function-like words (Kobak 'common excess' set) and LLM-vs-LLM markers (Sun et al.) are
        # too frequent in human prose to count unless several independent sources list them
        weak = ("common excess", "LLM-vs-LLM")
        items = [it for it in data["items"] if it.get("direction") == "over" and it.get("unit") in ("word", "phrase")
                 and not re.search(r"[\[\]]|\bX\b", it["term"])
                 and (n_src.get(it["term"], 1) >= 2 or not any(w in it.get("list", "") for w in weak))]
    elif isinstance(data, dict) and "terms" in data:
        items = data["terms"]
    else:
        items = data
    terms = [(it["term"] if isinstance(it, dict) else it).lower().strip() for it in items]
    return sorted({t for t in terms if t})


def load_trigrams(path: str | None = None) -> list[tuple[str, ...]]:
    """Stopword-removed trigrams (Antislop) for skip-gram matching."""
    p = Path(path) if path else ASSETS / "ai_vocabulary.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    if not (isinstance(data, dict) and "items" in data):
        return []
    return [tuple(it["term"].lower().split()) for it in data["items"] if it.get("unit") == "trigram"]


STOPWORDS = set("""a an the and or but if of at by for with about against between into through during before after
above below to from up down in out on off over under again further then once here there when where why how all any both
each few more most other some such no nor not only own same so than too very s t can will just don should now i me my we
our you your he him his she her it its they them their what which who whom this that these those am is are was were be
been being have has had having do does did doing her hers herself himself itself""".split())

OBFUSCATION_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff\u00a0\u0430\u0435\u043e\u0440\u0441\u0445\u0455\u0456\u0458]")


def vocab_regex(term: str) -> str:
    """Match a lemma plus regular inflections (delve -> delves, delved, delving)."""
    t = re.escape(term)
    if " " in term:
        return rf"\b{t}\b"
    stem = t[:-1] if term.endswith("e") else t
    return rf"\b(?:{t}|{stem}(?:s|es|d|ed|ing|ingly|ly|ment|ments)?)\b"


def scan(text: str, vocab: list[str]) -> dict:
    raw = text
    prose = strip_markup(text)
    n_words = max(1, len(words(prose)))
    per_k = 1000.0 / n_words
    hits = []
    fam_counts: dict[str, int] = {}
    weighted = 0.0
    for fam, w, rx in PATTERNS:
        for m in re.finditer(rx, prose, flags=re.I if fam not in ("staged_opener", "signpost_transition") else 0):
            hits.append({"family": fam, "weight": w, "match": m.group(0).strip()[:120], "start": m.start()})
            fam_counts[fam] = fam_counts.get(fam, 0) + 1
            weighted += w
    for fam, w, rx in FORMAT_PATTERNS:
        for m in re.finditer(rx, raw):
            hits.append({"family": fam, "weight": w, "match": m.group(0).strip()[:120], "start": m.start()})
            fam_counts[fam] = fam_counts.get(fam, 0) + 1
            weighted += w

    # AI-overrepresented vocabulary
    vocab_hits: dict[str, int] = {}
    low = prose.lower()
    for term in vocab:
        c = len(re.findall(vocab_regex(term), low))
        if c:
            vocab_hits[term] = c
    vocab_total = sum(vocab_hits.values())

    # stopword-removed trigram cliches (Antislop) and character-level obfuscation
    content = [w.lower() for w in words(prose) if w.lower() not in STOPWORDS]
    grams = set(zip(content, content[1:], content[2:]))
    tri_hits = [" ".join(t) for t in load_trigrams() if t in grams]
    obf = len(OBFUSCATION_RE.findall(raw))

    # punctuation profile
    em = prose.count("\u2014")
    en_dash = len(re.findall(r"\s\u2013\s", prose))
    dbl = len(re.findall(r"\s--\s", prose))
    curly = len(re.findall("[\u201c\u201d\u2018\u2019]", raw))
    semis = prose.count(";")
    colons = len(re.findall(r":(?!//)", prose))
    bold = len(re.findall(r"\*\*[^*]+\*\*", raw))

    # triads "X, Y, and Z"
    triads = len(re.findall(r"\b[\w-]+(?: [\w-]+)?, [\w-]+(?: [\w-]+)?,? and [\w-]+\b", prose))

    # paragraph-level: one-sentence closers after a longer paragraph
    paras = paragraphs(prose)
    closers = 0
    for prev, cur in zip(paras, paras[1:]):
        if len(words(cur)) <= 12 and len(words(prev)) >= 30 and len(split_sentences(cur)) == 1:
            closers += 1

    # sentence-opening repetition
    sents = split_sentences(prose)
    firsts = [s.split()[0].lower().strip("\"'(") for s in sents if s.split()]
    rep_open = sum(1 for a, b in zip(firsts, firsts[1:]) if a == b)

    weighted += 1.0 * (em + en_dash + dbl) + 0.5 * vocab_total + 0.5 * triads + 2.0 * closers + 2.0 * len(tri_hits)
    result = {
        "n_words": n_words,
        "n_sentences": len(sents),
        "surface_score_per_1k": round(weighted * per_k, 2),
        "family_counts": fam_counts,
        "rates_per_1k": {
            "ai_vocab": round(vocab_total * per_k, 2),
            "em_dash": round((em + en_dash + dbl) * per_k, 2),
            "semicolon": round(semis * per_k, 2),
            "colon": round(colons * per_k, 2),
            "bold": round(bold * per_k, 2),
            "triad": round(triads * per_k, 2),
            "pattern_hits": round(len(hits) * per_k, 2),
        },
        "counts": {
            "em_or_spaced_dash": em + en_dash + dbl, "curly_quotes": curly, "one_line_closers": closers,
            "repeated_sentence_openers": rep_open, "triads": triads, "ai_vocab_tokens": vocab_total,
            "cliche_trigrams": len(tri_hits), "obfuscation_chars": obf,
        },
        "cliche_trigram_hits": tri_hits,
        "integrity_warning": ("text contains zero-width, non-breaking or Cyrillic look-alike characters; normalise "
                              "(NFKC) and remove them - they game detectors and are not part of humanizing") if obf else None,
        "ai_vocab_hits": dict(sorted(vocab_hits.items(), key=lambda kv: -kv[1])),
        "hits": sorted(hits, key=lambda h: h["start"]),
    }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="text/markdown file, or - for stdin")
    ap.add_argument("--json", help="write JSON here instead of stdout")
    ap.add_argument("--vocab", help="vocabulary JSON (default assets/ai_vocabulary.json)")
    a = ap.parse_args()
    dump_json(scan(read_text(a.path), load_vocab(a.vocab)), a.json)


if __name__ == "__main__":
    main()
