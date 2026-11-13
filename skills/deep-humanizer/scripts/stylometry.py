#!/usr/bin/env python
"""Layer 2 - syntactic, register and rhythm features (Biber-style).

Instruction-tuned LLMs write in a dense, noun-heavy, informationally even
register: more nominalisations and participial clauses, fewer contractions,
pronouns, hedges and finite subordinate clauses, and sentence lengths that
vary less than human sentence lengths do. These features are content-light,
so they survive word-level paraphrase better than vocabulary does.

All rates are per 1,000 tokens unless the name says otherwise. The
function-word profile is returned separately so check.py can compute a
Burrows-Delta-style distance to a reference (e.g. the author's own writing).

Usage:
    python stylometry.py draft.md [--json out.json]
"""
from __future__ import annotations

import argparse
import gzip
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hz_common import (burstiness, dump_json, get_nlp, paragraphs, read_text, safe_cv,  # noqa: E402
                       strip_markup)

FUNCTION_WORDS = (
    "the of and to a in that is was it for on with as be by at this have from or an but not are "
    "which they you were his her she he we there been one all would their has if so what can who "
    "will more no when out up about into than them only some could do then its my i any these our "
    "also just like even very much because how may such those both each might most should while "
    "though although however yet still"
).split()

PRIVATE_VERBS = set("think feel believe guess suppose know mean suspect doubt wonder reckon figure imagine assume "
                    "hope realize realise recall remember forget understand notice".split())
HEDGES = {"maybe", "perhaps", "probably", "possibly", "apparently", "seemingly", "roughly", "presumably", "arguably"}
HEDGE_PHRASES = ["kind of", "sort of", "i think", "i guess", "i suspect", "seems to", "tend to", "more or less"]
AMPLIFIERS = {"very", "really", "extremely", "highly", "truly", "deeply", "incredibly", "remarkably", "profoundly",
              "significantly", "particularly", "exceptionally"}
DOWNTONERS = {"somewhat", "slightly", "fairly", "rather", "quite", "pretty", "barely", "hardly", "mildly", "partly"}
NOMINAL_SUFFIXES = ("tion", "tions", "sion", "sions", "ment", "ments", "ness", "nesses", "ity", "ities",
                    "ance", "ances", "ence", "ences", "ization", "isation")
# POS templates that writers most often edited out of LLM prose (Chakrabarty et al. 2025, LAMP; Penn tags)
LLM_POS_TEMPLATES = [("DT", "NN", "IN", "NN", "CC"), ("NN", "IN", "NN", "CC", "NN"), ("DT", "JJ", "NN", "IN", "PRP$"),
                     ("DT", "NN", "IN", "JJ", "NN"), ("IN", "DT", "NN", "IN", "NN", "CC")]

CONTRACTION_PARTS = {"n't", "'re", "'ve", "'ll", "'d", "'m", "’re", "’ve", "’ll", "’d", "’m", "n’t"}


def mtld(tokens: list[str], threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis 2010), bidirectional mean."""
    def one_pass(seq):
        factors, types, count = 0.0, set(), 0
        for w in seq:
            count += 1
            types.add(w)
            if len(types) / count <= threshold:
                factors += 1
                types, count = set(), 0
        if count:
            ttr = len(types) / count
            factors += (1 - ttr) / (1 - threshold) if ttr != 1 else 0
        return len(seq) / factors if factors else float("nan")
    if len(tokens) < 50:
        return float("nan")
    return (one_pass(tokens) + one_pass(tokens[::-1])) / 2


def mattr(tokens: list[str], window: int = 50) -> float:
    if len(tokens) < window:
        return len(set(tokens)) / max(1, len(tokens))
    counts = Counter(tokens[:window])
    vals = [len(counts) / window]
    for i in range(window, len(tokens)):
        counts[tokens[i]] += 1
        old = tokens[i - window]
        counts[old] -= 1
        if counts[old] == 0:
            del counts[old]
        vals.append(len(counts) / window)
    return sum(vals) / len(vals)


def compression_ratio(s: str) -> float:
    b = s.encode("utf-8")
    return len(b) / max(1, len(gzip.compress(b)))


def entropy(counter: Counter) -> float:
    n = sum(counter.values())
    return -sum((c / n) * math.log2(c / n) for c in counter.values()) if n else float("nan")


def register_spans(text: str, top: int = 8) -> dict:
    """Locate the register markers that distinguish instruction-tuned LLM prose (Reinhart et al. 2025)
    so Pass B can edit specific places: participial riders, nominalisations, coordinated pairs, noun
    chains, that-clause subjects; plus long words used only once (elaborate vocabulary that raises
    lexical diversity above the human band). Returns the most marked sentences."""
    nlp = get_nlp()
    if nlp is None:
        return {}
    prose = strip_markup(text)
    docs = [nlp(p) for p in paragraphs(prose)]
    out = []
    for d in docs:
        for s in d.sents:
            m = []
            for t in s:
                if t.tag_ == "VBG" and t.dep_ in ("advcl", "acl", "xcomp") and not any(c.dep_ == "aux" for c in t.children):
                    m.append(("participial", d[t.i:min(t.right_edge.i + 1, t.i + 6)].text))
                elif t.tag_ == "VBN" and t.dep_ in ("advcl", "acl") and not any(c.dep_ in ("aux", "auxpass") for c in t.children):
                    m.append(("past_participial", d[t.i:min(t.right_edge.i + 1, t.i + 6)].text))
                if t.pos_ == "NOUN" and t.lower_.endswith(NOMINAL_SUFFIXES) and len(t.lower_) > 6:
                    m.append(("nominalisation", t.text))
                if t.dep_ == "cc" and t.head.pos_ in ("NOUN", "ADJ", "PROPN"):
                    conj = [c for c in t.head.children if c.dep_ == "conj"]
                    if conj:
                        m.append(("coordinated_pair", f"{t.head.text} {t.text} {conj[0].text}"))
                if t.dep_ == "csubj" and any(c.lower_ == "that" and c.dep_ == "mark" for c in t.children):
                    m.append(("that_clause_subject", d[t.left_edge.i:t.i + 1].text))
            run = []
            for t in list(s) + [None]:
                if t is not None and t.pos_ in ("NOUN", "PROPN"):
                    run.append(t.text)
                else:
                    if len(run) >= 3:
                        m.append(("noun_chain", " ".join(run)))
                    run = []
            if m:
                out.append({"sentence": s.text.strip(), "markers": m, "n": len(m)})
    out.sort(key=lambda r: -r["n"])
    lows = Counter(t.lower_ for d in docs for t in d if t.is_alpha)
    elaborate = sorted({t.lower_ for d in docs for t in d if t.is_alpha and t.pos_ in ("NOUN", "VERB", "ADJ", "ADV")
                        and len(t.text) >= 10 and lows[t.lower_] == 1 and not t.text[0].isupper()})
    totals = Counter(k for r in out for k, _ in r["markers"])
    return {"sentences": out[:top], "totals": dict(totals), "elaborate_once": elaborate[:25]}


def features(text: str) -> dict:
    import numpy as np
    nlp = get_nlp()
    if nlp is None:
        raise SystemExit("stylometry.py needs spaCy + en_core_web_sm")
    prose = strip_markup(text)
    paras = paragraphs(prose)
    docs = [nlp(p) for p in paras]
    toks = [t for d in docs for t in d if not t.is_space]
    words_ = [t for t in toks if t.is_alpha]
    n = max(1, len(words_))
    k = 1000.0 / n
    sents = [s for d in docs for s in d.sents if any(t.is_alpha for t in s)]
    sent_lens = [sum(1 for t in s if t.is_alpha) for s in sents]
    para_lens = [sum(1 for _ in d.sents) for d in docs]

    c = Counter()
    dep_dists, depths = [], []
    for t in toks:
        tag, dep, low = t.tag_, t.dep_, t.lower_
        if t.pos_ == "NOUN" and low.endswith(NOMINAL_SUFFIXES) and len(low) > 6:
            c["nominalization"] += 1
        if tag == "VBG" and dep in ("advcl", "acl", "xcomp") and not any(ch.dep_ == "aux" for ch in t.children):
            c["present_participial_clause"] += 1
        if tag == "VBN" and dep in ("advcl", "acl") and not any(ch.dep_ in ("aux", "auxpass") for ch in t.children):
            c["past_participial_clause"] += 1
        if dep in ("nsubjpass", "csubjpass"):
            if any(ch.dep_ == "agent" for ch in t.head.children):
                c["by_passive"] += 1
            else:
                c["agentless_passive"] += 1
        if low == "that" and dep == "mark" and t.head.dep_ in ("ccomp", "acl"):
            c["that_complement"] += 1
        if dep == "relcl":
            c["relative_clause"] += 1
        if dep == "csubj" and any(ch.lower_ == "that" and ch.dep_ == "mark" for ch in t.children):
            c["that_clause_subject"] += 1
        if dep in ("advcl", "ccomp", "csubj", "relcl", "acl"):
            c["finite_or_nonfinite_subclause"] += 1
        if low in CONTRACTION_PARTS or (low in ("'s", "’s") and t.pos_ == "AUX"):
            c["contraction"] += 1
        if t.pos_ == "PRON" and low in ("i", "me", "my", "mine", "we", "us", "our", "ours"):
            c["first_person"] += 1
        if t.pos_ == "PRON" and low in ("you", "your", "yours"):
            c["second_person"] += 1
        if t.pos_ == "VERB" and t.lemma_.lower() in PRIVATE_VERBS:
            c["private_verb"] += 1
        if low in HEDGES:
            c["hedge"] += 1
        if low in AMPLIFIERS:
            c["amplifier"] += 1
        if low in DOWNTONERS:
            c["downtoner"] += 1
        if dep == "amod":
            c["attributive_adj"] += 1
        if t.pos_ == "ADV":
            c["adverb"] += 1
        if t.pos_ == "NOUN":
            c["noun"] += 1
        if t.pos_ == "VERB":
            c["verb"] += 1
        if t.pos_ == "ADP":
            c["preposition"] += 1
        if dep == "cc" and t.head.pos_ in ("NOUN", "ADJ", "PROPN"):
            c["phrasal_coordination"] += 1
        if dep == "cc" and t.head.pos_ in ("VERB", "AUX"):
            c["clausal_coordination"] += 1
        if t.text == "(":
            c["parenthesis"] += 1
        if not t.is_punct and t.head.i != t.i:
            dep_dists.append(abs(t.i - t.head.i))
    low_text = prose.lower()
    for ph in HEDGE_PHRASES:
        c["hedge"] += low_text.count(ph)

    for s in sents:
        def depth(tok):
            d = 0
            while tok.head.i != tok.i and d < 200:
                tok = tok.head
                d += 1
            return d
        depths.append(max(depth(t) for t in s))

    # noun-compound chains of >= 3 nouns ("model evaluation framework design")
    run = 0
    for t in toks:
        if t.pos_ in ("NOUN", "PROPN"):
            run += 1
        else:
            if run >= 3:
                c["noun_chain3"] += 1
            run = 0
    tags = [t.tag_ for t in toks]
    tagset = {}
    for L in {len(tpl) for tpl in LLM_POS_TEMPLATES}:
        tagset[L] = Counter(tuple(tags[i:i + L]) for i in range(len(tags) - L + 1))
    c["lamp_pos_template"] = sum(tagset[len(tpl)][tpl] for tpl in LLM_POS_TEMPLATES)
    # Shaib-style template repetition inside the document: share of POS 6-grams occurring >= 2 times
    six = Counter(tuple(tags[i:i + 6]) for i in range(len(tags) - 5))
    rep6 = sum(v for v in six.values() if v >= 2) / max(1, sum(six.values()))
    word_lows = [t.lower_ for t in words_]
    fw = Counter(w for w in word_lows if w in FUNCTION_WORDS)
    fw_profile = {w: fw.get(w, 0) * k for w in FUNCTION_WORDS}
    pos_seq = " ".join(t.pos_ for t in toks)
    pos_bigrams = Counter(zip([t.pos_ for t in toks], [t.pos_ for t in toks][1:]))
    openers = [next((t for t in s if not t.is_punct), None) for s in sents]
    opener_pos = Counter(o.pos_ for o in openers if o is not None)
    opener_words = Counter(o.lower_ for o in openers if o is not None)
    sl = np.asarray(sent_lens, dtype=float) if sent_lens else np.asarray([0.0])

    rates = {f"{key}_per_1k": round(v * k, 3) for key, v in c.items()}
    for key in ("nominalization", "present_participial_clause", "past_participial_clause", "agentless_passive",
                "by_passive", "that_complement", "relative_clause", "contraction", "first_person", "second_person",
                "private_verb", "hedge", "amplifier", "downtoner", "attributive_adj", "adverb", "noun", "verb",
                "preposition", "phrasal_coordination", "clausal_coordination", "parenthesis",
                "finite_or_nonfinite_subclause", "that_clause_subject", "noun_chain3", "lamp_pos_template"):
        rates.setdefault(f"{key}_per_1k", 0.0)

    out = {
        "n_tokens": n,
        "n_sentences": len(sents),
        "n_paragraphs": len(paras),
        "sent_len_mean": float(sl.mean()),
        "sent_len_sd": float(sl.std(ddof=1)) if sl.size > 1 else float("nan"),
        "sent_len_cv": safe_cv(sl),
        "sent_len_burstiness": burstiness(sl),
        "frac_short_sent_lt8": float((sl < 8).mean()),
        "frac_long_sent_gt30": float((sl > 30).mean()),
        "sent_len_adjacent_absdiff_mean": float(np.abs(np.diff(sl)).mean()) if sl.size > 1 else float("nan"),
        "para_len_cv": safe_cv(para_lens) if len(para_lens) > 1 else float("nan"),
        "mtld": mtld(word_lows),
        "mattr50": mattr(word_lows),
        "hapax_ratio": sum(1 for v in Counter(word_lows).values() if v == 1) / n,
        "mean_word_len": float(np.mean([len(w) for w in word_lows])) if word_lows else float("nan"),
        "long_word_ratio": sum(1 for w in word_lows if len(w) >= 7) / n,
        "noun_verb_ratio": c["noun"] / max(1, c["verb"]),
        "mean_dep_distance": float(np.mean(dep_dists)) if dep_dists else float("nan"),
        "mean_parse_depth": float(np.mean(depths)) if depths else float("nan"),
        "subclauses_per_sentence": c["finite_or_nonfinite_subclause"] / max(1, len(sents)),
        "pos_bigram_entropy": entropy(pos_bigrams),
        "pos_compression_ratio": compression_ratio(pos_seq),
        "pos6_repeat_share": rep6,
        "text_compression_ratio": compression_ratio(prose),
        "opener_pos_entropy": entropy(opener_pos),
        "opener_word_type_ratio": len(opener_words) / max(1, len(sents)),
        "question_rate": sum(1 for s in sents if s.text.strip().endswith("?")) / max(1, len(sents)),
        **rates,
    }
    return {"features": out, "function_word_profile": fw_profile}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--json")
    a = ap.parse_args()
    dump_json(features(read_text(a.path)), a.json)


if __name__ == "__main__":
    main()
