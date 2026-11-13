#!/usr/bin/env python
"""Meaning-preservation guard: did the rewrite keep every claim and add none?

Distribution-level rewriting pushes the text away from the model's most
probable phrasing; the risk is that it also drifts away from the source's
content or picks up invented specifics. This script checks both directions:

* coverage    - every original sentence has a close semantic match in the
                rewrite (sentence or adjacent-sentence window), else it is listed as LOST
* additions   - every rewrite sentence is supported by some original window,
                else it is listed as UNSUPPORTED (possible fabrication)
* hard facts  - numbers, percentages, years, citations, URLs, quoted strings
                and named entities must match as sets (MISSING / ADDED)
* optional NLI (--nli) - entailment of each original sentence by its best
                rewrite window, using a small cross-encoder. Every contradiction
                above 0.5 is listed; none are summarised away.

Restructure mode (--restructure) is for paragraph-level rewrites, where the
default sentence alignment reports reordered or merged sentences as LOST. It
matches each original sentence against windows of up to three rewrite
sentences, runs NLI, and classifies every original sentence as
  KEPT         entailed (or close match without NLI)
  CONFLICT     NLI contradiction > 0.5 (always listed with its probability)
  PLANNED_CUT  matches a line in --cuts (deliberate removals)
  DROPPED      no match and not planned
and every new rewrite sentence as
  NEW_FACT     carries a number, citation or named entity absent from the original (fail)
  AUTHOR_CLAIM first-person statement about the writer's experience (the user must confirm it)
  REWORDING    no new facts
Verdict: PASS only with no CONFLICT, no DROPPED, no NEW_FACT and no
missing hard facts outside the planned cuts; AUTHOR_CLAIM items are reported
for confirmation.

Usage:
    python meaning_guard.py original.md rewrite.md [--nli] [--json out.json]
    python meaning_guard.py original.md rewrite.md --restructure [--cuts cuts.txt] [--json out.json]
Exit code 0 = PASS, 2 = REVIEW (inspect the listed items).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hz_common import dump_json, get_nlp, load_sentence_model, model_id, read_text, split_sentences, strip_markup  # noqa: E402

NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"


def windows(sents: list[str]) -> list[str]:
    return sents + [a + " " + b for a, b in zip(sents, sents[1:])]


NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}


def hard_facts(text: str) -> dict[str, set]:
    t = strip_markup(text)
    facts = {
        "numbers": set(re.findall(r"(?<![\w.])\d+(?:[.,]\d+)*%?", t)),
        "citations": set(re.findall(r"\([A-Z][A-Za-z\-]+(?: et al\.)?(?:,| and [A-Z][A-Za-z\-]+,)? \d{4}[a-z]?\)|\[\d+(?:[,\u2013-]\s*\d+)*\]", t)),
        "urls": set(re.findall(r"https?://\S+", text)),
        "quotes": {q.strip() for q in re.findall(r"[\"\u201c]([^\"\u201d]{8,200})[\"\u201d]", t)},
    }
    nlp = get_nlp()
    if nlp is not None:
        ents = {(e.text.strip(), e.label_) for e in nlp(t).ents
                if e.label_ in ("PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "DATE")}
        facts["entities"] = {e[0] for e in ents}
    return facts


def spelled(num: str, text: str) -> bool:
    """True if a digit fact is present in text as a word ("10" -> "ten")."""
    words = [w for w, d in NUMBER_WORDS.items() if d == num]
    return any(re.search(rf"\b{w}\b", text, re.I) for w in words)


def diff_facts(fo: dict, fr: dict, original: str, rewrite: str) -> dict:
    facts = {}
    for k in fo:
        missing = sorted(fo[k] - fr.get(k, set()))
        added = sorted(fr.get(k, set()) - fo[k])
        if k == "numbers":
            missing = [x for x in missing if not spelled(x, rewrite)]
            added = [x for x in added if not spelled(x, original)]
        if missing or added:
            facts[k] = {"missing": missing, "added": added}
    return facts


def guard(original: str, rewrite: str, use_nli: bool = False, cover_thr: float = 0.62, add_thr: float = 0.5) -> dict:
    import numpy as np
    so = split_sentences(strip_markup(original))
    sr = split_sentences(strip_markup(rewrite))
    m = load_sentence_model(model_id("sent_emb"))
    wo, wr = windows(so), windows(sr)
    Eo = m.encode(so, normalize_embeddings=True, show_progress_bar=False)
    Er = m.encode(sr, normalize_embeddings=True, show_progress_bar=False)
    Ewo = m.encode(wo, normalize_embeddings=True, show_progress_bar=False)
    Ewr = m.encode(wr, normalize_embeddings=True, show_progress_bar=False)
    vo, vr = Eo.mean(0), Er.mean(0)
    doc_cos = float(vo @ vr / (np.linalg.norm(vo) * np.linalg.norm(vr)))
    cov = Eo @ Ewr.T
    sup = Er @ Ewo.T
    best_cov = cov.max(1)
    best_sup = sup.max(1)
    lost = [{"sentence": so[i], "best_match_sim": float(best_cov[i]), "best_match": wr[int(cov[i].argmax())]}
            for i in np.where(best_cov < cover_thr)[0]]
    unsupported = [{"sentence": sr[i], "best_support_sim": float(best_sup[i])}
                   for i in np.where(best_sup < add_thr)[0]]
    fo, fr = hard_facts(original), hard_facts(rewrite)
    facts = diff_facts(fo, fr, original, rewrite)
    out = {
        "doc_cosine": doc_cos,
        "coverage": float((best_cov >= cover_thr).mean()),
        "support": float((best_sup >= add_thr).mean()),
        "length_ratio_words": len(rewrite.split()) / max(1, len(original.split())),
        "lost_sentences": lost,
        "unsupported_sentences": unsupported,
        "fact_diffs": facts,
    }
    if use_nli:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(NLI_MODEL)
        pairs = [(wr[int(cov[i].argmax())], so[i]) for i in range(len(so))]
        logits = ce.predict(pairs, apply_softmax=True)
        # label order for cross-encoder/nli-*: contradiction, entailment, neutral
        ent = logits[:, 1]
        con = logits[:, 0]
        out["nli_entailment_mean"] = float(ent.mean())
        out["nli_contradictions"] = [{"original": so[i], "rewrite_window": pairs[i][0], "p_contradiction": float(con[i])}
                                     for i in range(len(so)) if con[i] > 0.5]
    hard_fail = any(v["missing"] or v["added"] for k, v in facts.items() if k in ("numbers", "citations", "urls", "quotes"))
    out["verdict"] = "PASS" if (doc_cos >= 0.85 and out["coverage"] >= 0.9 and not unsupported and not hard_fail
                                and not out.get("nli_contradictions")) else "REVIEW"
    return out


def windows3(sents: list[str]) -> list[str]:
    return windows(sents) + [" ".join(sents[i:i + 3]) for i in range(len(sents) - 2)]


FIRST_PERSON = re.compile(r"\b(I|I'm|I've|I'd|me|my|we|our|us)\b")


def guard_restructure(original: str, rewrite: str, cuts: list[str] | None = None, use_nli: bool = True,
                      keep_thr: float = 0.62, cut_thr: float = 0.7, add_thr: float = 0.5) -> dict:
    import numpy as np
    cuts = [c.strip() for c in (cuts or []) if c.strip()]
    so = split_sentences(strip_markup(original))
    sr = split_sentences(strip_markup(rewrite))
    m = load_sentence_model(model_id("sent_emb"))
    wr = windows3(sr)
    Eo = m.encode(so, normalize_embeddings=True, show_progress_bar=False)
    Er = m.encode(sr, normalize_embeddings=True, show_progress_bar=False)
    Ewr = m.encode(wr, normalize_embeddings=True, show_progress_bar=False)
    Ewo = m.encode(windows3(so), normalize_embeddings=True, show_progress_bar=False)
    Ec = m.encode(cuts, normalize_embeddings=True, show_progress_bar=False) if cuts else None
    cov = Eo @ Ewr.T
    ent = con = None
    if use_nli:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(NLI_MODEL)
        # premise = best three rewrite windows, take the max entailment and the max contradiction
        top = np.argsort(-cov, 1)[:, :3]
        pairs = [(wr[j], so[i]) for i in range(len(so)) for j in top[i]]
        P = ce.predict(pairs, apply_softmax=True).reshape(len(so), top.shape[1], 3)
        ent, con = P[:, :, 1].max(1), P[:, :, 0]
    rows = []
    for i, s_ in enumerate(so):
        best = int(cov[i].argmax())
        sim = float(cov[i, best])
        planned = bool(cuts) and (any(c.lower() in s_.lower() for c in cuts) or float((Eo[i] @ Ec.T).max()) >= cut_thr)
        row = {"sentence": s_, "best_window": wr[best], "sim": round(sim, 3)}
        if use_nli:
            # contradiction only counts against the window that matches best
            row["p_entail"] = round(float(ent[i]), 3)
            row["p_contradiction"] = round(float(con[i, 0]), 3)
        if use_nli and row["p_contradiction"] > 0.5 and not planned:
            status = "CONFLICT"
        elif (use_nli and row["p_entail"] >= 0.5) or sim >= keep_thr:
            status = "KEPT"
        elif planned:
            status = "PLANNED_CUT"
        else:
            status = "DROPPED"
        row["status"] = status
        rows.append(row)
    fo, fr = hard_facts(original), hard_facts(rewrite)
    # facts inside planned-cut sentences, or written in the --cuts lines themselves, count as planned
    cut_text = " ".join([r["sentence"] for r in rows if r["status"] == "PLANNED_CUT"] + cuts)
    fc = hard_facts(cut_text) if cut_text else {k: set() for k in fo}
    facts = diff_facts(fo, fr, original, rewrite)
    for k, v in facts.items():
        v["missing_unplanned"] = [x for x in v["missing"] if x not in fc.get(k, set())]
    sup = (Er @ Ewo.T).max(1)
    additions = []
    for i, s_ in enumerate(sr):
        if float(sup[i]) >= add_thr:
            continue
        f = hard_facts(s_)
        new = {k: sorted(v - fo.get(k, set())) for k, v in f.items() if k in ("numbers", "citations", "entities")}
        new = {k: v for k, v in new.items() if v}
        kind = "NEW_FACT" if new else "AUTHOR_CLAIM" if FIRST_PERSON.search(s_) else "REWORDING"
        additions.append({"sentence": s_, "best_support_sim": round(float(sup[i]), 3), "kind": kind, "new_facts": new})
    counts = {k: sum(r["status"] == k for r in rows) for k in ("KEPT", "CONFLICT", "PLANNED_CUT", "DROPPED")}
    hard_fail = any(v["missing_unplanned"] or v["added"] for k, v in facts.items() if k in ("numbers", "citations", "urls", "quotes"))
    fail = counts["CONFLICT"] or counts["DROPPED"] or hard_fail or any(a_["kind"] == "NEW_FACT" for a_ in additions)
    return {"mode": "restructure", "counts": counts, "sentences": rows,
            "conflicts": [r for r in rows if r["status"] == "CONFLICT"],
            "dropped": [r for r in rows if r["status"] == "DROPPED"],
            "planned_cuts": [r for r in rows if r["status"] == "PLANNED_CUT"],
            "additions": additions, "author_claims": [a_ for a_ in additions if a_["kind"] == "AUTHOR_CLAIM"],
            "fact_diffs": facts, "nli": use_nli,
            "verdict": "REVIEW" if fail else "PASS"}


def report(res: dict) -> str:
    """Plain-text listing of every item that needs a decision (both modes)."""
    L = [f"Meaning guard verdict: {res['verdict']}"]
    if res.get("mode") == "restructure":
        L.append("counts: " + json.dumps(res["counts"]))
        for r in res["conflicts"]:
            L.append(f"CONFLICT p={r['p_contradiction']:.2f}: \"{r['sentence']}\"  vs  \"{r['best_window']}\"")
        for r in res["dropped"]:
            L.append(f"DROPPED (sim {r['sim']:.2f}): \"{r['sentence']}\"")
        for r in res["planned_cuts"]:
            L.append(f"PLANNED_CUT: \"{r['sentence']}\"")
        for a_ in res["additions"]:
            L.append(f"{a_['kind']}: \"{a_['sentence']}\"" + (f" new={a_['new_facts']}" if a_["new_facts"] else ""))
    else:
        for x in res["lost_sentences"]:
            L.append(f"LOST (sim {x['best_match_sim']:.2f}): \"{x['sentence']}\"")
        for x in res["unsupported_sentences"]:
            L.append(f"UNSUPPORTED (sim {x['best_support_sim']:.2f}): \"{x['sentence']}\"")
        for x in res.get("nli_contradictions", []):
            L.append(f"CONTRADICTION p={x['p_contradiction']:.2f}: \"{x['original']}\"  vs  \"{x['rewrite_window']}\"")
    for k, v in res["fact_diffs"].items():
        L.append(f"FACTS {k}: {json.dumps(v)}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("original")
    ap.add_argument("rewrite")
    ap.add_argument("--nli", action="store_true")
    ap.add_argument("--restructure", action="store_true", help="paragraph-level rewrite: claim-level matching + NLI")
    ap.add_argument("--cuts", help="file with one planned cut per line (phrase or sentence)")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.restructure:
        cuts = Path(a.cuts).read_text().splitlines() if a.cuts else None
        res = guard_restructure(read_text(a.original), read_text(a.rewrite), cuts)
    else:
        res = guard(read_text(a.original), read_text(a.rewrite), a.nli)
    print(report(res))
    if a.json:
        dump_json(res, a.json)
    sys.exit(0 if res["verdict"] == "PASS" else 2)


if __name__ == "__main__":
    main()
