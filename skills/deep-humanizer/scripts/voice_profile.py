#!/usr/bin/env python
"""Writer voice profiles: measure an author's own habits and compare a draft with them.

The calibrated check round compares a draft with a *generic* human band (the
reference bank). A writer's own band can sit elsewhere: an academic who
writes dense, noun-heavy prose, a blogger who uses dashes, a non-native
writer with plain syntax. A voice profile records that band so the agent
edits toward the writer, not toward an average human.

A profile is a directory, kept outside any public repository:

    <profile>/voice.md       the writer's style card. The agent reads it before editing.
                             The writer edits it; build_profile.py only rewrites the block
                             between the "measured" markers.
    <profile>/profile.json   measured bands of the writer's own samples (build_profile.py)
    <profile>/samples/       the writing samples (copied by build_profile.py)

A profile made by hand from assets/voice_template.md (voice.md only) works too:
the agent reads it, and the check round applies its phrase lists.

What a profile changes: the editing guidance (voice gaps, personal phrase
lists, spelling convention, which generic targets match the writer's own
style). What it never changes: the calibrated verdict, the held-out detector
gate, the meaning gate.

Profile lookup: --profile DIR | NAME | path/to/voice.md, else the environment
variable DEEP_HUMANIZER_PROFILE. A bare NAME is looked up in
~/.deep-humanizer/profiles/NAME (override the root with DEEP_HUMANIZER_HOME).

Standalone quick voice check (surface + syntax only, no large models):
    python voice_profile.py draft.md --profile blog [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hz_common import dump_json, paragraphs, read_text, split_sentences, strip_markup, words  # noqa: E402

PROFILE_ENV = "DEEP_HUMANIZER_PROFILE"
MEASURED_START = "<!-- measured:start -->"
MEASURED_END = "<!-- measured:end -->"
SECTION_AVOID = "phrases i avoid"
SECTION_MINE = "words that are mine"
SAMPLE_SUFFIXES = (".txt", ".md")


def profile_home() -> Path:
    return Path(os.environ.get("DEEP_HUMANIZER_HOME", "~/.deep-humanizer")).expanduser() / "profiles"


# ------------------------------------------------------------------ loading

def resolve_profile(arg: str | None = None) -> tuple[Path | None, str]:
    """Return (profile_dir, source). An explicit argument that cannot be found is an error;
    a bad environment variable only produces a warning."""
    val, source = (arg, "--profile") if arg else (os.environ.get(PROFILE_ENV), PROFILE_ENV)
    if not val:
        return None, ""
    p = Path(val).expanduser()
    cands = [p.parent if p.is_file() else p, profile_home() / val]
    for c in cands:
        if c.is_dir() and ((c / "voice.md").exists() or (c / "profile.json").exists()):
            return c.resolve(), source
    msg = f"voice profile not found: {val!r} (looked for voice.md or profile.json in {', '.join(map(str, cands))})"
    if source == "--profile":
        raise SystemExit(msg)
    print(f"[voice] {msg}; continuing without a profile", file=sys.stderr)
    return None, ""


def section_bullets(md: str, heading_prefix: str) -> list[str]:
    """Bullet items under the first heading that starts with heading_prefix (case-insensitive).
    Placeholder bullets in parentheses and empty bullets are ignored."""
    out, on = [], False
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("#"):
            on = s.lstrip("#").strip().lower().startswith(heading_prefix)
            continue
        if on and re.match(r"^[-*]\s+", s):
            item = re.sub(r"^[-*]\s+", "", s).strip()
            item = re.sub(r"\s+<!--.*?-->$", "", item).strip().strip("`\"'\u201c\u201d")
            if item and not item.startswith("("):
                out.append(item)
    return out


def sample_files(d: Path) -> list[Path]:
    return sorted(f for f in d.iterdir() if f.is_file() and f.suffix.lower() in SAMPLE_SUFFIXES
                  and not f.name.startswith("."))


def load_profile(arg: str | None = None) -> dict | None:
    d, source = resolve_profile(arg)
    if d is None:
        return None
    data = json.loads((d / "profile.json").read_text(encoding="utf-8")) if (d / "profile.json").exists() else {}
    md = (d / "voice.md").read_text(encoding="utf-8") if (d / "voice.md").exists() else ""
    sd = data.get("samples_dir") or "samples"
    sp = Path(sd).expanduser()
    sp = sp if sp.is_absolute() else d / sp
    files = sample_files(sp) if sp.is_dir() else []
    return {"dir": str(d), "source": source, "name": data.get("name") or d.name, "data": data, "voice_md": md,
            "avoid": section_bullets(md, SECTION_AVOID), "mine": section_bullets(md, SECTION_MINE),
            "samples_dir": str(sp) if len(files) >= 3 else None, "n_sample_files": len(files)}


# ------------------------------------------------------------------ measurement helpers

def normalise_paragraphs(text: str) -> tuple[str, bool]:
    """Text pasted from web pages or word processors often separates paragraphs with a single line
    break. Within each blank-line block, treat each line as a paragraph if at least 80% of the lines
    end like sentences, or if the median line is longer than 120 characters (hard-wrapped text is
    wrapped at 72-100). Hard-wrapped text (short lines ending mid-sentence) is left alone."""
    blocks = re.split(r"\n[ \t]*\n", text.strip())
    out, changed = [], False
    for b in blocks:
        lines = [ln.strip() for ln in b.split("\n") if ln.strip()]
        ends = sum(1 for ln in lines if re.search(r"[.!?:\"'\u201d\u2019)]$", ln))
        long_lines = bool(lines) and sorted(len(ln) for ln in lines)[len(lines) // 2] > 120
        if len(lines) >= 2 and (ends >= 0.8 * len(lines) or long_lines):
            out.extend(lines)
            changed = True
        else:
            out.append(b.strip())
    return ("\n\n".join(out), True) if changed else (text, False)


def chunk_text(text: str, target: int = 300, min_words: int = 100) -> list[str]:
    """Paragraph-aligned chunks of about `target` words; very long paragraphs are split by sentence."""
    units: list[str] = []
    for p in paragraphs(strip_markup(text)):
        if len(words(p)) > 1.3 * target:
            cur, n = [], 0
            for s in split_sentences(p):
                cur.append(s)
                n += len(words(s))
                if n >= target:
                    units.append(" ".join(cur))
                    cur, n = [], 0
            if cur:
                units.append(" ".join(cur))
        else:
            units.append(p)
    chunks, cur, n = [], [], 0
    for u in units:
        nu = len(words(u))
        if cur and n + nu > 1.3 * target and n >= min_words:
            chunks.append("\n\n".join(cur))
            cur, n = [], 0
        cur.append(u)
        n += nu
        if n >= target:
            chunks.append("\n\n".join(cur))
            cur, n = [], 0
    if cur:
        if chunks and n < min_words:
            chunks[-1] += "\n\n" + "\n\n".join(cur)
        else:
            chunks.append("\n\n".join(cur))
    return [c for c in chunks if len(words(c)) >= min_words]


ISE_EXCLUDE = set("""otherwise likewise clockwise precise concise rise arise sunrise wise advise exercise promise surprise
comprise enterprise expertise noise premise devise revise compromise franchise merchandise televise advertise supervise
improvise despise disguise chastise excise incise apprise raise praise poise guise cruise bruise demise paradise treatise
reprise mortise porpoise tortoise turquoise valise chemise anise cerise surmise circumcise""".split())
IZE_EXCLUDE = {"size", "prize", "seize", "capsize", "maize", "baize", "downsize", "oversize", "resize"}
_ISE_RE = re.compile(r"^([a-z]{2,})(is|iz)(e|es|ed|ing|er|ers|ation|ations|able)$")
_ANALY_RE = re.compile(r"^(analy|paraly|cataly)([sz])(e|es|ed|ing|er|ers)$")
_OUR_STEMS = ("colo|flavo|hono|humo|labo|neighbo|rumo|behavio|favo|harbo|vapo|savo|endeavo|tumo|odo|armo|parlo|"
              "clamo|rigo|vigo|splendo|valo|fervo|demeano")
_OUR_RE = re.compile(rf"^({_OUR_STEMS})(u?)r(s|ed|ing|ful|ite|ites|able|ably|al|ally|ism|less|hood|hoods|ly)?$")
_RE_RE = re.compile(r"^(cent|theat|fib|lit|calib|lust|somb|spect)(re|er)(s|d|ed)?$")
_LL_RE = re.compile(r"^(travel|model|label|cancel|signal|fuel|level|channel|tunnel|total|dial|counsel|marshal)(l?)"
                    r"(ed|ing|er|ers|or|ors)$")


def spelling(text: str) -> dict:
    """Count British vs American spellings (ise/ize, our/or, re/er, doubled l, analyse/analyze).
    Capitalised tokens are skipped so names do not count."""
    br, am = Counter(), Counter()
    for tok in re.findall(r"[A-Za-z]+", strip_markup(text)):
        if tok[0].isupper() and not tok.isupper():
            continue
        w = tok.lower()
        m = _ANALY_RE.match(w)
        if m:
            (br if m.group(2) == "s" else am)[w] += 1
            continue
        m = _ISE_RE.match(w)
        if m and m.group(1) + "ise" not in ISE_EXCLUDE and m.group(1) + "ize" not in IZE_EXCLUDE:
            (br if m.group(2) == "is" else am)[w] += 1
            continue
        m = _OUR_RE.match(w)
        if m and w != "humoral":
            (br if m.group(2) == "u" else am)[w] += 1
            continue
        m = _RE_RE.match(w)
        if m:
            (br if m.group(2) == "re" else am)[w] += 1
            continue
        m = _LL_RE.match(w)
        if m:
            (br if m.group(2) == "l" else am)[w] += 1
    b, a = sum(br.values()), sum(am.values())
    if b >= 3 and b >= 3 * a:
        conv = "British"
    elif a >= 3 and a >= 3 * b:
        conv = "American"
    elif a >= 2 and b >= 2:
        conv = "mixed"
    else:
        conv = "unknown"
    return {"convention": conv, "british": b, "american": a,
            "british_examples": [w for w, _ in br.most_common(8)], "american_examples": [w for w, _ in am.most_common(8)]}


CONNECTIVES = ["however", "but", "and", "so", "also", "therefore", "thus", "hence", "moreover", "furthermore",
               "additionally", "in addition", "first", "second", "finally", "instead", "still", "yet", "then", "now",
               "of course", "in other words", "for example", "for instance", "as a result", "overall", "in short",
               "indeed", "importantly", "notably", "ultimately", "consequently", "in conclusion", "in summary",
               "interestingly", "in essence", "meanwhile", "similarly", "likewise", "nevertheless", "nonetheless",
               "admittedly", "honestly", "actually", "basically"]
LLM_FAVOURITE_OPENERS = ["moreover", "furthermore", "additionally", "in addition", "notably", "importantly",
                         "ultimately", "consequently", "overall", "in conclusion", "in summary", "in essence",
                         "interestingly", "indeed", "thus", "hence"]


def _opener(sentence: str) -> tuple[str, str | None]:
    s = sentence.strip().lstrip("\"'\u201c\u2018(*_").lower()
    first = (re.findall(r"[a-z]+(?:'[a-z]+)?", s) or [""])[0]
    conn = None
    for c in sorted(CONNECTIVES, key=len, reverse=True):
        if re.match(rf"{re.escape(c)}\b", s):
            conn = c
            break
    return first, conn


def text_habits(text: str) -> dict:
    """Countable habits that the feature layers do not report directly."""
    prose = strip_markup(text)
    sents = split_sentences(prose)
    paras = paragraphs(prose)
    n_w = max(1, len(words(prose)))
    k = 1000.0 / n_w
    toks = re.findall(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)?", prose)
    openers = Counter()
    conns = Counter()
    for s in sents:
        f, c = _opener(s)
        if f:
            openers[f] += 1
        if c:
            conns[c] += 1
    return {
        "n_words": n_w, "n_sentences": len(sents), "n_paragraphs": len(paras),
        "words_per_paragraph": n_w / max(1, len(paras)),
        "sentences_per_paragraph": len(sents) / max(1, len(paras)),
        "i_per_1k": sum(1 for t in toks if t == "I" or t.startswith(("I'", "I\u2019"))) * k,
        "we_per_1k": sum(1 for t in toks if t.lower() in ("we", "we're", "we've", "we'll", "we'd", "our", "us")) * k,
        "dash_glyphs": {"em_dash": prose.count("\u2014"), "spaced_en_dash": len(re.findall(r"\s\u2013\s", prose)),
                        "spaced_hyphen": len(re.findall(r"\s-\s", prose)), "double_hyphen": prose.count("--")},
        "quotes": {"straight": prose.count('"') + len(re.findall(r"(?<![A-Za-z])'|'(?![A-Za-z])", prose)),
                   "curly": len(re.findall("[\u201c\u201d\u2018]", prose))},
        "exclamation_rate": sum(1 for s in sents if s.rstrip().endswith("!")) / max(1, len(sents)),
        "openers": dict(openers.most_common(25)),
        "connectives": dict(conns),
        "spelling": spelling(prose),
    }


def extra_features(text: str) -> dict:
    """Scalar voice features added to the flattened layer features (prefix voice.)."""
    h = text_habits(text)
    return {"voice.words_per_paragraph": h["words_per_paragraph"],
            "voice.sentences_per_paragraph": h["sentences_per_paragraph"],
            "voice.i_per_1k": h["i_per_1k"], "voice.we_per_1k": h["we_per_1k"],
            "voice.exclamation_rate": h["exclamation_rate"]}


# label, unit kind, advice if the draft is higher than the author, advice if lower
VOICE_FEATURES = {
    "sty.sent_len_mean": ("average sentence length", "words",
                          "Split long sentences where you would.", "Join choppy sentences where you would."),
    "sty.sent_len_sd": ("spread of sentence lengths", "words",
                        "Your sentence lengths vary less; even them out.", "Vary sentence length the way you do: some short, some long."),
    "sty.frac_short_sent_lt8": ("share of very short sentences (<8 words)", "share", "Fewer fragments.", "Let a few points land in short sentences."),
    "sty.frac_long_sent_gt30": ("share of long sentences (>30 words)", "share", "Break up the longest sentences.", ""),
    "voice.sentences_per_paragraph": ("sentences per paragraph", "sents",
                                      "Use shorter paragraphs, as you do.", "Your paragraphs are usually longer; merge related ones."),
    "voice.words_per_paragraph": ("words per paragraph", "count", "Shorter paragraphs.", "Longer, fuller paragraphs."),
    "sty.contraction_per_1k": ("contractions", "per1k", "Fewer contractions than this.", "Use contractions where you would (it's, don't)."),
    "voice.i_per_1k": ("'I'", "per1k", "Less 'I' than this.", "You write in the first person singular more often."),
    "voice.we_per_1k": ("'we/our/us'", "per1k", "Less 'we' than this.", "You use 'we' more often."),
    "sty.second_person_per_1k": ("'you'", "per1k", "Address the reader less directly.", "You address the reader more often."),
    "sty.hedge_per_1k": ("hedges (maybe, perhaps, I think)", "per1k", "Hedge less; commit where the evidence is clear.", "You hedge more than this draft does."),
    "sty.amplifier_per_1k": ("amplifiers (very, really, highly)", "per1k", "Cut amplifiers.", ""),
    "sty.question_rate": ("questions", "share", "Fewer rhetorical questions.", "You use questions more often (e.g. to frame a section)."),
    "voice.exclamation_rate": ("exclamations", "share", "Fewer exclamation marks.", ""),
    "sty.nominalization_per_1k": ("nominalisations (-tion, -ment)", "per1k", "Turn nouns back into verbs.", ""),
    "sty.present_participial_clause_per_1k": ("trailing -ing clauses", "per1k", "Turn -ing riders into finite clauses.", ""),
    "sty.phrasal_coordination_per_1k": ("'X and Y' pairs", "per1k", "Break reflexive pairs.", ""),
    "sty.agentless_passive_per_1k": ("passives without an agent", "per1k", "Name who did it.", "You use the passive more (e.g. in methods)."),
    "sty.parenthesis_per_1k": ("parentheses", "per1k", "Fewer asides in parentheses.", "You use parenthetical asides more."),
    "sty.mtld": ("lexical variety (MTLD)", "index", "Reuse your key terms instead of synonyms.", "You vary vocabulary more than this."),
    "sty.mean_word_len": ("average word length", "letters", "Prefer plainer, shorter words.", ""),
    "srf.em_dash": ("dashes", "per1k", "Replace dashes with your usual punctuation.", "You use dashes; keep them at your rate."),
    "srf.semicolon": ("semicolons", "per1k", "Fewer semicolons.", "You use semicolons more."),
    "srf.colon": ("colons", "per1k", "Fewer colons.", "You use colons more (e.g. to introduce a list or point)."),
    "srf.triad": ("three-item lists", "per1k", "Keep three items only when there are three.", ""),
    "srf.ai_vocab": ("AI-list vocabulary", "per1k", "Remove AI-list words that are not yours.", ""),
}
FLOORS = {"per1k": 1.0, "share": 0.03, "words": 1.5, "sents": 1.0, "count": 10.0, "letters": 0.1, "index": 6.0}


def bands(rows: list[dict]) -> dict:
    """Per-feature distribution over the author's chunks."""
    import numpy as np
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and not
                   (isinstance(v, float) and math.isnan(v))})
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows if isinstance(r.get(k), (int, float)) and not
                      (isinstance(r.get(k), float) and math.isnan(r[k]))], dtype=float)
        if v.size == 0:
            continue
        med = float(np.median(v))
        out[k] = {"n": int(v.size), "median": med, "p10": float(np.percentile(v, 10)),
                  "p90": float(np.percentile(v, 90)), "min": float(v.min()), "max": float(v.max()),
                  "mad": float(np.median(np.abs(v - med))), "values": [round(float(x), 4) for x in sorted(v)]}
    return out


def _scale(b: dict, unit: str | None) -> float:
    """Robust spread of the writer's chunks. Curated voice features have a unit floor so a writer who
    never uses a feature is not flagged for one occurrence; other features use their own spread only."""
    floor = FLOORS.get(unit, 0.0) if unit else 0.0
    return max(1.4826 * b["mad"], 0.15 * abs(b["median"]), floor, 1e-9)


def in_writer_range(b: dict, v: float, unit: str | None) -> bool:
    """Inside the writer's band: robust |z| < 2 and, with 4+ chunks, within the values the writer produced."""
    z = (v - b["median"]) / _scale(b, unit)
    return abs(z) < 2 and (b["n"] < 4 or b["min"] <= v <= b["max"])


def function_word_stats(profiles: list[dict], top: int = 30) -> dict | None:
    """Burrows-Delta style reference: mean and sd of the author's top function words over chunks,
    plus leave-one-out distances of each chunk to the rest."""
    import numpy as np
    if len(profiles) < 5:
        return None
    words_ = sorted(profiles[0])
    X = np.array([[p.get(w, 0.0) for w in words_] for p in profiles])
    order = np.argsort(-X.mean(0))[:top]
    W = [words_[i] for i in order]
    X = X[:, order]
    loo = []
    for i in range(len(X)):
        rest = np.delete(X, i, 0)
        mu, sd = rest.mean(0), rest.std(0, ddof=1) + 1e-6
        loo.append(float(np.mean(np.abs((X[i] - mu) / sd))))
    return {"words": W, "mean": X.mean(0).tolist(), "sd": (X.std(0, ddof=1) + 1e-6).tolist(),
            "loo_delta": sorted(loo)}


def function_word_delta(fw_profile: dict, ref: dict) -> float:
    import numpy as np
    x = np.array([fw_profile.get(w, 0.0) for w in ref["words"]])
    return float(np.mean(np.abs((x - np.array(ref["mean"])) / np.array(ref["sd"]))))


# ------------------------------------------------------------------ comparison

def _phrase_hits(text: str, phrases: list[str]) -> list[dict]:
    prose = strip_markup(text)
    sents = split_sentences(prose)
    out = []
    for ph in phrases:
        rx = re.compile(r"(?<![\w])" + r"\s+".join(map(re.escape, ph.split())) + r"(?![\w])", re.I)
        n = len(rx.findall(prose))
        if n:
            ctx = [s for s in sents if rx.search(s)][:2]
            out.append({"phrase": ph, "count": n, "in": ctx})
    return out


def compare(text: str, res: dict, prof: dict) -> dict:
    """Voice comparison of a draft against a profile. `res` is check.extract_all output (any layers)."""
    data = prof.get("data", {})
    B = data.get("bands", {})
    flat = dict(res.get("flat", {}))
    flat.update(extra_features(text))
    n_words = len(words(strip_markup(text)))
    out: dict = {"profile": prof["name"], "profile_dir": prof["dir"], "profile_source": prof.get("source"),
                 "measured": bool(B), "n_words": n_words,
                 "short_text_warning": n_words < 150}
    gaps, within = [], []
    for f, (label, unit, adv_hi, adv_lo) in VOICE_FEATURES.items():
        b = B.get(f)
        if not b or f not in flat or b["n"] < 3:
            continue
        v = float(flat[f])
        z = (v - b["median"]) / _scale(b, unit)
        outside_range = (v < b["min"] or v > b["max"]) if b["n"] >= 4 else True
        row = {"feature": f, "label": label, "value": round(v, 3), "author_median": round(b["median"], 3),
               "author_p10": round(b["p10"], 3), "author_p90": round(b["p90"], 3), "z": round(z, 2)}
        if abs(z) >= 2 and outside_range:
            row["advice"] = (adv_hi if z > 0 else adv_lo) or ("Bring it toward your usual." if z > 0 else "")
            row["direction"] = "higher than yours" if z > 0 else "lower than yours"
            gaps.append(row)
        else:
            within.append(f)
    out["gaps"] = sorted(gaps, key=lambda r: -abs(r["z"]))
    out["n_within"] = len(within)
    # personal lists
    out["avoid_hits"] = _phrase_hits(text, prof.get("avoid", []))
    mine = {m.lower() for m in prof.get("mine", [])}
    hits = res.get("surface", {}).get("ai_vocab_hits", {}) or {}
    out["ai_vocab_yours"] = {k: v for k, v in hits.items() if k.lower() in mine}
    out["ai_vocab_not_yours"] = {k: v for k, v in hits.items() if k.lower() not in mine}
    # habits: spelling, openers, dashes
    h = text_habits(text)
    ah = data.get("habits", {})
    conv = ah.get("spelling", {}).get("convention")
    sp = h["spelling"]
    if conv == "British" and sp["american"]:
        out["spelling_mismatch"] = {"yours": "British", "draft_has": sp["american_examples"]}
    elif conv == "American" and sp["british"]:
        out["spelling_mismatch"] = {"yours": "American", "draft_has": sp["british_examples"]}
    a_conn = ah.get("connectives", {})
    if ah.get("n_sentences", 0) >= 40:
        unused = [c for c in h["connectives"] if c in LLM_FAVOURITE_OPENERS and not a_conn.get(c)]
        if unused:
            out["openers_you_never_use"] = {c: h["connectives"][c] for c in unused}
    ad = ah.get("dash_glyphs", {})
    if ad and h["dash_glyphs"]["em_dash"] and not ad.get("em_dash") and (ad.get("spaced_en_dash") or ad.get("spaced_hyphen")):
        out["dash_glyph_note"] = "Your samples use spaced en dashes or hyphens, not em dashes."
    # function words
    fwr = data.get("function_words")
    fwp = res.get("stylometry", {}).get("function_word_profile")
    if fwr and fwp:
        import numpy as np
        d = function_word_delta(fwp, fwr)
        loo = np.array(fwr["loo_delta"])
        out["function_word_delta"] = {"draft": round(d, 3), "author_median": round(float(np.median(loo)), 3),
                                      "author_max": round(float(loo.max()), 3),
                                      "above_author_range": bool(d > loo.max())}
    au = res.get("embedding", {}).get("author_style")
    if au and "style_sim_to_author" in au:
        out["style_embedding"] = {k: round(v, 3) if isinstance(v, float) else v for k, v in au.items()}
        out["style_embedding"]["below_author_min"] = au["style_sim_to_author"] < au["author_loo_sim_min"]
    return out


def annotate_targets(ev: dict, prof: dict, stats: dict, percentile_fn) -> None:
    """Mark generic targets where the writer's own median also sits on the AI side of the generic
    human band (AI-side percentile >= 75) and the draft is inside the writer's range. These are
    the writer's habits, not edits to make. The verdict is not changed."""
    B = prof.get("data", {}).get("bands", {})
    feats = stats.get("features", {})
    for t in ev.get("targets", []):
        b, st = B.get(t["feature"]), feats.get(t["feature"])
        if not b or not st or b["n"] < 3:
            continue
        pct = percentile_fn(b["median"], st["human_quantiles"])
        a_side = pct if st["direction"] == "higher_is_ai" else 100 - pct
        unit = VOICE_FEATURES[t["feature"]][1] if t["feature"] in VOICE_FEATURES else None
        t["author_median"] = round(b["median"], 3)
        t["author_ai_side_percentile"] = round(a_side, 1)
        t["voice_consistent"] = bool(a_side >= 75 and in_writer_range(b, float(t["value"]), unit))


def markdown(v: dict) -> list[str]:
    def f(x):
        return f"{x:.3g}" if isinstance(x, float) else str(x)
    L = [f"## Voice match to profile '{v['profile']}'", "",
         f"Profile: `{v['profile_dir']}` (via {v.get('profile_source')}). The profile guides the edit; it does not change "
         "the verdict above.", ""]
    if v.get("short_text_warning"):
        L += ["Text is under 150 words: voice comparisons are rough.", ""]
    if not v.get("measured"):
        L += ["No measured bands (voice.md only). Run build_profile.py on your samples for habit comparisons.", ""]
    if v.get("gaps"):
        L += [f"Voice gaps ({len(v['gaps'])}; {v['n_within']} other habits within your range):", ""]
        for g in v["gaps"]:
            L.append(f"- {g['label']}: draft {f(g['value'])}, yours {f(g['author_median'])} "
                     f"(usual {f(g['author_p10'])}-{f(g['author_p90'])}), {g['direction']}. {g.get('advice', '')}")
        L.append("")
    elif v.get("measured"):
        L += [f"All {v['n_within']} measured habits are within your range.", ""]
    if v.get("avoid_hits"):
        L += ["Phrases you avoid (from voice.md), found in the draft:", ""]
        L += [f"- \"{h['phrase']}\" x{h['count']}: " + " | ".join(s[:140] for s in h["in"]) for h in v["avoid_hits"]] + [""]
    if v.get("ai_vocab_not_yours") or v.get("ai_vocab_yours"):
        L.append(f"AI-list words not in your vocabulary: {json.dumps(v.get('ai_vocab_not_yours', {}))}")
        if v.get("ai_vocab_yours"):
            L.append(f"AI-list words that are yours (keep): {json.dumps(v['ai_vocab_yours'])}")
        L.append("")
    if v.get("spelling_mismatch"):
        sm = v["spelling_mismatch"]
        L += [f"Spelling: you write {sm['yours']}; the draft has {', '.join(sm['draft_has'])}.", ""]
    if v.get("openers_you_never_use"):
        L += [f"Sentence openers that never appear in your samples: {json.dumps(v['openers_you_never_use'])}", ""]
    if v.get("dash_glyph_note"):
        L += [v["dash_glyph_note"], ""]
    fd = v.get("function_word_delta")
    if fd:
        L += [f"Function-word distance (heuristic): draft {fd['draft']}, your own chunks median {fd['author_median']}, "
              f"max {fd['author_max']}" + (" -> outside your range" if fd["above_author_range"] else ""), ""]
    se = v.get("style_embedding")
    if se:
        L += [f"Style embedding: similarity {se['style_sim_to_author']} to your samples (your samples to each other: "
              f"mean {se['author_loo_sim_mean']}, min {se['author_loo_sim_min']})"
              + (" -> below your own minimum" if se.get("below_author_min") else ""), ""]
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--profile", help=f"profile directory, name, or voice.md path (default ${PROFILE_ENV})")
    ap.add_argument("--json")
    a = ap.parse_args()
    prof = load_profile(a.profile)
    if prof is None:
        raise SystemExit(f"no profile: pass --profile or set {PROFILE_ENV}")
    import check
    text = read_text(a.path)
    res = check.extract_all(text, ("surface", "stylometry"), reference="none")
    v = compare(text, res, prof)
    if a.json:
        dump_json(v, a.json)
    print("\n".join(markdown(v)))


if __name__ == "__main__":
    main()
