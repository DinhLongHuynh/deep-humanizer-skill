#!/usr/bin/env python
"""Paragraph- and document-level structure scan.

The other layers score words and sentences. Readers and current detectors also
react to how paragraphs are built: a run of sentences that each report one
finding at the same pace, no reaction from the writer, a tidy summary line at
the end, paragraphs of equal size, and several answers that all follow the same
arc. Sentence-level edits leave these in place (session finding, November 2026:
28 of 66 sentences edited, three detectors at ~0, tropa-mini and ZeroGPT still
at ~1.0).

For every paragraph this script reports
  fact_share      share of sentences carrying a finding (digit, %, citation,
                  et al., 2+ acronyms, quoted span)
  stance_per_100  writer-stance markers (epistemic, evaluative, experiential)
  first_person    I / my / me / we / our per 100 words
  sent_len_cv     sentence-length spread inside the paragraph
  summary_closer  last sentence looks like a wrap-up line
  writer_share    share of sentences where the writer speaks (first person or an
                  epistemic adverb such as probably / actually); reported stance
                  of others ("respondents felt") does not count
  detectors       held-out detector p(AI) on the paragraph alone (>= 60 words),
                  and, for every detector that flags the whole document, a
                  drop-one attribution: the document is re-scored without each
                  paragraph. Paragraph-alone scores often stay low while the
                  document is flagged (tropa-mini needs context), so the drop-one
                  delta is the one to use for locating the driver.
and flags paragraphs that combine these into a report-like run. With several
files it also checks whether all documents share one arc (same opener, the
personal paragraph always in the same slot).

Thresholds are heuristics from the literature summarised in
references/research_digest.md and this session's failure case; they are NOT
calibrated on the reference bank. Use them to locate paragraphs, not as a score.

Usage
  python structure_scan.py doc.md [doc2.md ...] [--detectors tropa_mini,desklib] [--no-detectors] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hz_common import dump_json, paragraphs, read_text, split_sentences, strip_markup, word_count  # noqa: E402

STANCE = re.compile(
    r"\b(think|thought|guess|doubt|expect(?:ed)?|surpris\w*|confus\w*|lik(?:e|ed)|disliked|agree|disagree|wonder\w*|"
    r"honestly|actually|probably|maybe|perhaps|seems?|seemed|apparently|not sure|unsure|to me|for me|in my view|"
    r"I would|I'd|stuck with|struck|realis\w*|realiz\w*|noticed|felt|feel|worr\w*|glad|strange|odd|"
    r"careful|oversell|bet|hard to|easy to|clearly|of course|bad example|poor example)\b", re.I)
FIRST_PERSON = re.compile(r"\b(I|I'm|I've|I'd|me|my|mine|we|our|us)\b")
WRITER_ADV = re.compile(r"\b(probably|maybe|perhaps|actually|honestly|of course|clearly|surely|apparently|frankly)\b", re.I)
ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{1,}s?\b")
CITATION = re.compile(r"\bet al\.|\([A-Z][A-Za-z\-]+[^()]{0,40}\d{4}[a-z]?\)")
CLOSER = re.compile(
    r"^(In short|Overall|In summary|To sum up|Ultimately|In the end|All in all|Taken together|In a way|"
    r"This (?:shows|means|highlights|suggests|is why)|That is (?:why|the same|what)|Reading the two together|"
    r"Together,)|\b(which is (?:why|the same)|the same \w+ (?:that|as) )", re.I)


def is_fact_sentence(s: str) -> bool:
    if re.search(r"\d", s) or "%" in s or CITATION.search(s) or re.search(r"[\"\u201c][^\"\u201d]{8,}[\"\u201d]", s):
        return True
    return len(set(ACRONYM.findall(s)) - {"I"}) >= 2


def scan_paragraph(p: str) -> dict:
    sents = split_sentences(p)
    lens = [len(s.split()) for s in sents]
    n = max(1, word_count(p))
    cv = statistics.pstdev(lens) / statistics.mean(lens) if len(lens) >= 2 else None
    fact_share = sum(is_fact_sentence(s) for s in sents) / max(1, len(sents))
    stance = len(STANCE.findall(p)) * 100 / n
    fp = len(FIRST_PERSON.findall(p)) * 100 / n
    closer = bool(sents and CLOSER.search(sents[-1]))
    writer = sum(bool(FIRST_PERSON.search(s) or WRITER_ADV.search(s)) for s in sents) / max(1, len(sents))
    flags = []
    if len(sents) >= 4 and fact_share >= 0.5 and writer <= 0.2:
        flags.append("report-like: a run of findings and the writer never speaks")
    if len(sents) >= 4 and cv is not None and cv < 0.25:
        flags.append(f"even rhythm (sentence-length CV {cv:.2f})")
    if closer:
        flags.append("summary closer: ends on a wrap-up line")
    return {"text": p, "n_words": n, "n_sent": len(sents), "sent_len_cv": None if cv is None else round(cv, 2),
            "fact_share": round(fact_share, 2), "stance_per_100": round(stance, 2),
            "first_person_per_100": round(fp, 2), "writer_share": round(writer, 2), "summary_closer": closer,
            "opener": " ".join(sents[0].split()[:3]) if sents else "", "flags": flags}


def scan(text: str, detector_names=None, run_detectors: bool = True, min_words_det: int = 60, thr: float = 0.5,
         full_scores: dict | None = None, skip: set | None = None) -> dict:
    """full_scores: {detector: p_ai} for the whole document (computed here if None).
    skip: detectors not to attribute (e.g. uninformative on human controls)."""
    paras = [p for p in paragraphs(strip_markup(text)) if word_count(p) >= 15]
    rows = [scan_paragraph(p) for p in paras]
    attribution = {}
    if run_detectors and rows:
        import detectors
        names = detector_names or detectors.DEFAULT_OPEN
        for r in rows:
            if r["n_words"] >= min_words_det:
                out = detectors.run_open(r["text"], names)
                r["detectors"] = {k: round(v["p_ai"], 3) for k, v in out.items() if "p_ai" in v}
        if full_scores is None:
            full_scores = {k: v["p_ai"] for k, v in detectors.run_open(text, names).items() if "p_ai" in v}
        high = [k for k, v in full_scores.items() if v >= thr and k not in (skip or set())]
        if high and len(rows) >= 2:
            for i, r in enumerate(rows):
                rest = "\n\n".join(x["text"] for j, x in enumerate(rows) if j != i)
                out = detectors.run_open(rest, high)
                r["drop_one"] = {k: round(full_scores[k] - out[k]["p_ai"], 3) for k in high if "p_ai" in out[k]}
            for k in high:
                deltas = [r.get("drop_one", {}).get(k, 0.0) for r in rows]
                top = max(range(len(rows)), key=lambda j: deltas[j])
                attribution[k] = {"doc_p_ai": round(full_scores[k], 3), "drop_one_delta": deltas, "driver": top + 1}
                if deltas[top] >= 0.1:
                    rows[top]["flags"].append(f"drives {k}: removing it lowers p(AI) by {deltas[top]:.2f}")
                else:
                    rows[top]["flags"].append(f"largest {k} contribution, but no single paragraph drives it "
                                              f"(max drop {deltas[top]:.2f}): the signal is spread across the text")
    doc_flags = []
    lens = [r["n_words"] for r in rows]
    if len(lens) >= 3 and statistics.pstdev(lens) / statistics.mean(lens) < 0.15:
        doc_flags.append("paragraphs of near-equal length")
    tot = sum(lens) or 1
    stance_doc = sum(r["stance_per_100"] * r["n_words"] for r in rows) / tot
    fact_doc = sum(r["fact_share"] * r["n_sent"] for r in rows) / max(1, sum(r["n_sent"] for r in rows))
    if stance_doc < 0.5:
        doc_flags.append(f"almost no writer stance ({stance_doc:.2f} markers per 100 words)")
    sents = split_sentences(strip_markup(text))
    first = [s.split()[0].lower() for s in sents if s.split()]
    rep = sum(a == b for a, b in zip(first, first[1:])) / max(1, len(first) - 1)
    if rep > 0.25:
        doc_flags.append(f"consecutive sentences open with the same word ({rep:.0%})")
    return {"paragraphs": rows, "attribution": attribution, "doc": {"n_paragraphs": len(rows), "stance_per_100": round(stance_doc, 2),
                                        "fact_share": round(fact_doc, 2), "repeated_opener_share": round(rep, 2),
                                        "flags": doc_flags}}


def shared_arc(scans: dict) -> list[str]:
    """Cross-document template check: do all documents follow the same arc?"""
    if len(scans) < 3:
        return []
    out = []
    slots = []
    for s in scans.values():
        P = s["paragraphs"]
        if not P:
            continue
        i = max(range(len(P)), key=lambda j: P[j]["first_person_per_100"])
        slots.append("first" if i == 0 else "last" if i == len(P) - 1 else "middle")
    if slots and len(set(slots)) == 1:
        out.append(f"the most personal paragraph is in the same slot ({slots[0]}) in every document")
    closers = [s["paragraphs"][-1]["summary_closer"] for s in scans.values() if s["paragraphs"]]
    if closers and all(closers):
        out.append("every document ends on a summary line")
    w = [s["paragraphs"][0]["opener"].split()[0].lower() for s in scans.values() if s["paragraphs"] and s["paragraphs"][0]["opener"]]
    if w and len(set(w)) == 1:
        out.append(f"every document opens with '{w[0]}'")
    return out


def markdown(scan_res: dict) -> list[str]:
    L = ["## Structure (paragraph level; heuristic, not calibrated)", ""]
    d = scan_res["doc"]
    L.append(f"Document: {d['n_paragraphs']} paragraphs, stance {d['stance_per_100']} per 100 words, "
             f"fact-sentence share {d['fact_share']}, repeated openers {d['repeated_opener_share']:.0%}")
    for f in d["flags"]:
        L.append(f"- DOC: {f}")
    L.append("")
    L += ["| # | words | sentences | fact share | writer share | len CV | drop-one delta | alone (p AI) | flags |",
          "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(scan_res["paragraphs"], 1):
        det = ", ".join(f"{k} {v:.2f}" for k, v in r.get("detectors", {}).items()) or "-"
        d1 = ", ".join(f"{k} {v:+.2f}" for k, v in r.get("drop_one", {}).items()) or "-"
        L.append(f"| {i} | {r['n_words']} | {r['n_sent']} | {r['fact_share']} | {r['writer_share']} | "
                 f"{r['sent_len_cv'] if r['sent_len_cv'] is not None else '-'} | {d1} | {det} | {'; '.join(r['flags']) or '-'} |")
    L.append("")
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--detectors", default=None, help="comma list of open detectors (default: detectors.DEFAULT_OPEN)")
    ap.add_argument("--no-detectors", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    names = a.detectors.split(",") if a.detectors else None
    scans = {p: scan(read_text(p), names, not a.no_detectors) for p in a.paths}
    for p, s in scans.items():
        print(f"# {p}\n")
        print("\n".join(markdown(s)))
    arc = shared_arc(scans)
    if arc:
        print("## Shared arc across documents\n")
        print("\n".join(f"- {x}" for x in arc))
    if a.json:
        dump_json({"scans": scans, "shared_arc": arc}, a.json)


if __name__ == "__main__":
    main()
