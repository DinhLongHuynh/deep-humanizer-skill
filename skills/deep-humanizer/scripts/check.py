#!/usr/bin/env python
"""Check round: score a text on every layer, locate it relative to the human
reference distribution, and say what to fix next.

Layers
  surface    surface_scan.py   lexical / phrasal / punctuation / formatting tells
  stylometry stylometry.py     Biber-style register, rhythm, lexical diversity
  lm         lm_signals.py     Fast-DetectGPT, Binoculars, surprisal spread, GLTR ranks
  embedding  embed_geometry.py intrinsic dimension, sentence trajectory, reference-bank distance
  detectors  detectors.py      held-out open classifiers (+ commercial APIs with --external)
  structure  structure_scan.py paragraph-level build: fact runs, stance, closers, per-paragraph detectors

Verdict. CLEAN / BORDERLINE / FLAGGED come from the calibrated reference
models. If any held-out detector (open or API) scores the text at >= 0.5 while
the calibrated verdict is CLEAN or BORDERLINE, the verdict becomes
DETECTORS DISAGREE and the calibrated one is kept as verdict_calibrated. A
detector that scores the --controls human texts at >= 0.5 is marked
uninformative for this register and does not trigger the override.

Each scalar feature is placed on the human reference distribution stored in
<reference>/feature_stats.json (built by calibrate.py): its human percentile,
whether it lies on the AI side beyond the human 90th percentile, and how
discriminative the feature was during calibration (AUROC). Per-layer and
combined logistic models give a probability that the text belongs to the AI
reference class. The verdict uses the calibrated cut-offs in the same file.

Usage
  python check.py draft.md                                  # all local layers
  python check.py rewrite.md --baseline draft.md            # + deltas and meaning guard
  python check.py rewrite.md --author my_samples/           # + voice match to the author
  python check.py rewrite.md --final --external gptzero     # + held-out commercial detector
  python check.py draft.md --fast                           # surface + stylometry + LM w/o Binoculars
  python check.py v2.md --baseline v1.md --restructure --cuts cuts.txt   # after a paragraph rewrite
  python check.py draft.md --controls human_controls/       # read detectors against same-register human texts
  python check.py draft.md --profile blog                   # + voice match to the writer's profile (build_profile.py)
Outputs a markdown report (stdout or --md) and full JSON (--json).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embed_geometry  # noqa: E402
from hz_common import ASSETS, dump_json, read_text, split_sentences, strip_markup, to_jsonable, word_count  # noqa: E402

LAYER_PREFIX = {"srf": "surface", "sty": "stylometry", "lm": "lm", "emb": "embedding", "det": "detectors"}

TACTICS = {
    "lm.fastdetectgpt_norm": "P1 - Hotspot sentences use the scoring model's most probable wording. Rewrite them from the writer's angle: specific nouns and verbs from the source, an unexpected-but-correct word choice, a different clause order. Do not just swap synonyms.",
    "lm.binoculars": "P1 - Text is less surprising than an LLM expects text to be. Same remedy as Fast-DetectGPT: replace predictable connective phrasing and generic summary sentences with content the source supports.",
    "lm.gltr_top10": "P2 - Too many tokens are the model's top-10 choice. Break the predictable runs listed under hotspots; prefer the writer's own idiom over the default collocation.",
    "lm.logppl_observer": "P2 - Overall surprisal is low. Add concrete, source-grounded detail; drop throat-clearing and restatement that the model can predict.",
    "lm.surprisal_sd": "P3 - Information density is too even. Alternate dense sentences that carry a specific fact with short plain ones; let one sentence carry an unexpected but true detail.",
    "lm.uid_local": "P3 - Token-to-token surprisal changes too smoothly. Mix registers within the paragraph (a plain aside, a technical term, a short evaluative clause).",
    "lm.sent_surprisal_sd": "P3 - Every sentence is equally predictable. Vary what sentences do: claim, example, qualification, aside, question.",
    "sty.sent_len_cv": "S1 - Sentence lengths are too uniform. Combine some sentences, split others; include at least one very short and one long sentence per paragraph where the register allows.",
    "sty.sent_len_burstiness": "S1 - Sentence lengths are too regular (low burstiness). Same remedy as sentence-length CV.",
    "sty.sent_len_sd": "S1 - Sentence-length spread is narrow. Vary length deliberately.",
    "sty.frac_short_sent_lt8": "S1 - Almost no short sentences. Add a few short, plain sentences where a point lands.",
    "sty.sent_len_adjacent_absdiff_mean": "S1 - Neighbouring sentences have similar lengths. Alternate.",
    "sty.nominalization_per_1k": "S2 - Noun-heavy register. Turn nominalisations back into verbs with agents ('the implementation of X' -> 'we implemented X').",
    "sty.present_participial_clause_per_1k": "S3 - Too many participial riders (', highlighting...', ', ensuring...'). Make them finite clauses with a real subject, or cut them.",
    "sty.past_participial_clause_per_1k": "S3 - Reduce stacked past-participial modifiers; use relative or main clauses.",
    "sty.attributive_adj_per_1k": "S4 - Adjective stacking. Keep the one adjective that carries information.",
    "sty.phrasal_coordination_per_1k": "S4 - Paired nouns/adjectives ('clear and concise', 'X and Y'). Keep the more precise member.",
    "sty.contraction_per_1k": "S5 - No contractions in a register that normally has them. Use them where the writer would (not in formal academic prose).",
    "sty.first_person_per_1k": "S5 - Missing writer presence. Where the genre allows, state who did/thinks what ('we', 'I').",
    "sty.private_verb_per_1k": "S5 - Few stance verbs (think, suspect, doubt). Add genuine stance where the writer holds one.",
    "sty.hedge_per_1k": "S5 - Claims are uniformly confident. Calibrate: hedge where the evidence is weaker, commit where it is strong.",
    "sty.downtoner_per_1k": "S5 - Add graded qualification where it is real (somewhat, fairly, partly).",
    "sty.amplifier_per_1k": "S5 - Too many amplifiers (significantly, particularly, deeply). Cut them or quantify.",
    "sty.mtld": "S6 - Lexical diversity is outside the human band. If high: drop elegant variation and reuse the plain term; if low: name things specifically.",
    "sty.mattr50": "S6 - Local lexical diversity outside the human band (see MTLD).",
    "sty.mean_word_len": "S6 - Words are longer than the human norm. Prefer short, common words where they are exact.",
    "sty.long_word_ratio": "S6 - Too many long words. Plain alternatives.",
    "sty.noun_verb_ratio": "S2 - Too nominal. Put actions in verbs.",
    "sty.pos_compression_ratio": "S7 - Syntactic templates repeat (POS sequence compresses well). Vary sentence frames: inversion, fronted adverbial, question, fragment where allowed.",
    "sty.opener_pos_entropy": "S7 - Sentences open the same way. Vary first constituents.",
    "sty.opener_word_type_ratio": "S7 - Repeated sentence openers.",
    "sty.mean_dep_distance": "S7 - Dependency structure outside the human band; vary embedding depth.",
    "sty.subclauses_per_sentence": "S7 - Subordination rate outside the human band.",
    "sty.para_len_cv": "D1 - Paragraphs are uniform blocks. Let paragraph length follow content: a one-sentence paragraph only if it adds a fact.",
    "sty.text_compression_ratio": "D2 - Text is repetitive at the string level (restatement). Cut restatements and summaries of what was just said.",
    "emb.phd": "E1 - Intrinsic dimension of the token-embedding cloud is outside the human band. Its direction depends on the generator: 2023 models wrote lower-dimensional text, but in the default bank 2025-26 models score HIGHER than humans (more lexical variety). If the AI side is higher: cut elegant variation and decorative vocabulary, reuse key terms. If lower: add the source's concrete sub-topics and vary syntax. Noisy below 150 tokens.",
    "emb.mle_id": "E1 - MLE intrinsic dimension outside the human band (in the default bank AI text is lower in news and fiction): vary content and syntax; see emb.phd.",
    "emb.adj_cos_mean": "E2 - Adjacent sentences are semantically too similar (each restates the theme). Make each sentence add new information; cut restatements.",
    "emb.pairwise_cos_mean": "E2 - The whole text circles one semantic point. Add the concrete sub-topics, examples and caveats the source supports.",
    "emb.adj_cos_sd": "E2 - Semantic steps between sentences are too uniform. Let some transitions be small (elaboration) and some larger (shift to example, consequence, aside).",
    "emb.centroid_dispersion": "E2 - Sentences cluster tightly around the document centroid. Same remedy.",
    "emb.effective_rank_per_sent": "E2 - Sentence embeddings span few directions. Same remedy.",
    "emb.ref.knn_ai_fraction": "E3 - The document's nearest neighbours in the reference bank are mostly AI texts: content-level genericness. Ground the text in specifics from the source and the writer's own examples; remove generic framing paragraphs.",
    "emb.ref.emb_lr_p_ai": "E3 - Embedding-level classifier places the text in the AI region. See E3.",
    "emb.ref.human_density_percentile": "E3 - The text is far from any human document of this register. Check register: is it written for the right reader and genre?",
    "lm.surprisal_skew": "P3 - Surprisal lacks a right tail: almost no rare-but-apt tokens. Use the exact term, name, or verb the source supports where the draft has a generic one.",
    "lm.surprisal_kurt": "P3 - Surprisal distribution is too light-tailed. Same remedy as surprisal skew.",
    "lm.d1_var": "P3 - Surprisal changes too smoothly from token to token. Mix plain and dense phrasing within sentences.",
    "lm.d2_var": "P3 - Surprisal rhythm is too regular (DivEye second-order). Vary sentence jobs and phrasing density.",
    "lm.uid_span_range": "P3 - Every 20-token stretch carries the same information load (GPT-who span extremes). Let some stretches be dense with specifics and others plain.",
    "lm.uid_span_max": "P3 - No dense stretch of specific information. Add the concrete detail the source contains.",
    "lm.late_local_volatility": "P4 - The second half settles into predictable phrasing (late-stage stability). Rewrite the closing paragraphs hardest: drop the summary, end on a specific point.",
    "lm.late_deriv_dispersion": "P4 - Second-half surprisal is too stable. Same remedy as late local volatility.",
    "sty.pos6_repeat_share": "S7 - Repeated 6-tag POS templates (Shaib et al.). Rebuild repeated frames: change clause order, merge or split, swap a nominal phrase for a clause.",
    "sty.lamp_pos_template_per_1k": "S7 - POS templates writers most often edited out of LLM prose ('the N of N and N'). Restructure the phrase; do not just swap words.",
    "sty.that_clause_subject_per_1k": "S3 - That-clauses as subjects ('That X is Y suggests...'). Put the claim in a main clause.",
    "sty.noun_chain3_per_1k": "S2 - Stacked noun compounds. Unpack with a verb or preposition ('the framework we use to evaluate the model').",
    "srf.cliche_trigrams": "L2 - Stock phrase cliches (Antislop). Rewrite the moment, not the words.",
    "lm.surprisal_p95": "P3 - The most surprising tokens are not surprising enough: no apt, specific, less expected words. Use the exact term the source supports in place of the generic one.",
    "lm.gltr_beyond": "P2 - Almost no tokens outside the scoring model's top 1000. Same remedy as surprisal p95; do not stuff rare words.",
    "lm.gltr_top1000": "P2 - Same as gltr_beyond.",
    "lm.mean_log_rank": "P2 - Word choices sit at the top of the model's ranking. Rewrite hotspot sentences with the writer's own wording.",
    "sty.hapax_ratio": "S6 - Too many once-used words (elegant variation, synonym cycling). Reuse the key term; cut decorative variants.",
    "sty.pos_bigram_entropy": "S7 - Narrow range of part-of-speech sequences. Vary sentence frames (see S7).",
    "sty.noun_per_1k": "S2 - Noun-heavy. Put actions in verbs; cut abstract nouns.",
    "sty.verb_per_1k": "S2 - Verb-rate outside the human band (often from participial riders). Check S3 targets first.",
    "sty.preposition_per_1k": "S7 - Prepositional-phrase rate outside the human band; usually follows S2 edits.",
    "srf.surface_score_per_1k": "L1 - Surface tells. Work through the listed hits with references/surface_patterns.md.",
    "srf.ai_vocab": "L1 - Over-represented LLM vocabulary. Replace with the plain word or cut.",
    "srf.em_dash": "L1 - Dash rate above the human band. Choose the real relation: comma, colon, parentheses, full stop.",
}

DEFAULT_REFERENCE = ASSETS / "reference_default"


# ------------------------------------------------------------------ extraction

def flatten(res: dict) -> dict:
    f = {}
    s = res.get("surface")
    if s:
        f["srf.surface_score_per_1k"] = s["surface_score_per_1k"]
        for k, v in s["rates_per_1k"].items():
            f[f"srf.{k}"] = v
        f["srf.cliche_trigrams"] = s["counts"].get("cliche_trigrams", 0) * 1000.0 / max(1, s["n_words"])
    st = res.get("stylometry")
    if st:
        for k, v in st["features"].items():
            if k not in ("n_tokens", "n_sentences", "n_paragraphs"):
                f[f"sty.{k}"] = v
    lm = res.get("lm")
    if lm and "error" not in lm:
        for k, v in lm.items():
            if isinstance(v, (int, float)) and k != "n_tokens":
                f[f"lm.{k}"] = v
    e = res.get("embedding")
    if e:
        for k in ("phd", "mle_id", "adj_cos_mean", "adj_cos_sd", "pairwise_cos_mean", "centroid_dispersion",
                  "effective_rank_per_sent"):
            if k in e:
                f[f"emb.{k}"] = e[k]
        for k in ("knn_ai_fraction", "emb_lr_p_ai", "human_density_percentile"):
            if k in e.get("reference", {}):
                f[f"emb.ref.{k}"] = e["reference"][k]
    d = res.get("detectors", {}).get("open", {})
    for k, v in d.items():
        if "p_ai" in v:
            f[f"det.{k}"] = v["p_ai"]
    return {k: v for k, v in f.items() if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))}


def extract_all(text: str, layers=("surface", "stylometry", "lm", "embedding", "detectors"), reference=None,
                author=None, performer=True, open_detectors=None) -> dict:
    res: dict = {"n_words": word_count(strip_markup(text))}
    if "surface" in layers:
        import surface_scan
        res["surface"] = surface_scan.scan(text, surface_scan.load_vocab(None))
    if "stylometry" in layers:
        import stylometry
        res["stylometry"] = stylometry.features(text)
        res["stylometry"]["spans"] = stylometry.register_spans(text)
    if "lm" in layers:
        import lm_signals
        res["lm"] = lm_signals.score(text, performer="default" if performer else None)
    if "embedding" in layers:
        import embed_geometry
        res["embedding"] = embed_geometry.analyse(text, reference=reference, author_dir=author)
    if "detectors" in layers:
        import detectors
        res["detectors"] = {"open": detectors.run_open(text, open_detectors)}
    res["flat"] = flatten(res)
    return res


# ------------------------------------------------------------------ evaluation

def _percentile(v: float, qs: list[float]) -> float:
    """Percentile of v within a distribution summarised by 21 quantiles (0,5,...,100)."""
    import numpy as np
    grid = np.linspace(0, 100, len(qs))
    qs = np.asarray(qs, dtype=float)
    if v <= qs[0]:
        return 0.0
    if v >= qs[-1]:
        return 100.0
    qs_u, idx = np.unique(qs, return_index=True)
    return float(np.interp(v, qs_u, grid[idx]))


def _lr_prob(model: dict, flat: dict):
    import numpy as np
    feats = model["features"]
    if any(f not in flat for f in feats):
        missing = [f for f in feats if f not in flat]
        if len(missing) > 0.2 * len(feats):
            return None
    x = np.array([flat.get(f, mu) for f, mu in zip(feats, model["mu"])], dtype=float)
    z = (x - np.array(model["mu"])) / np.array(model["sd"])
    logit = float(z @ np.array(model["coef"]) + model["intercept"])
    return 1 / (1 + math.exp(-logit))


def evaluate(flat: dict, ref_dir: Path) -> dict:
    stats_path = ref_dir / "feature_stats.json"
    if not stats_path.exists():
        return {"warning": f"no calibration at {stats_path}; run calibrate.py. Raw values only."}
    S = json.loads(stats_path.read_text())
    feats, rows = S["features"], []
    for name, st in feats.items():
        if name not in flat:
            continue
        v = flat[name]
        pct = _percentile(v, st["human_quantiles"])
        ai_pct = pct if st["direction"] == "higher_is_ai" else 100 - pct
        rows.append({
            "feature": name, "layer": LAYER_PREFIX[name.split(".")[0]], "value": v,
            "human_median": st["human_quantiles"][10], "ai_median": st["ai_quantiles"][10],
            "human_percentile": round(pct, 1), "ai_side_percentile": round(ai_pct, 1),
            "auroc": st["auroc"], "direction": st["direction"],
            "diagnostic": st["auroc"] >= S.get("diagnostic_auroc", 0.7),
            "beyond_human_p90": ai_pct >= 90,
        })
    layers = {}
    for lname in ("surface", "stylometry", "lm", "embedding", "detectors"):
        lr = [r for r in rows if r["layer"] == lname and r["diagnostic"]]
        if not lr:
            continue
        info = {"n_diagnostic": len(lr), "n_beyond_p90": sum(r["beyond_human_p90"] for r in lr),
                "mean_ai_side_percentile": round(sum(r["ai_side_percentile"] for r in lr) / len(lr), 1)}
        m = S.get("layer_models", {}).get(lname)
        if m:
            p = _lr_prob(m, flat)
            if p is not None:
                info["p_ai"] = round(p, 3)
                info["cv_auroc"] = m.get("cv_auroc")
                info["flag"] = p >= m.get("threshold", 0.5)
        else:
            info["flag"] = info["n_beyond_p90"] >= max(1, math.ceil(0.3 * len(lr)))
        layers[lname] = info
    combined = None
    if S.get("combined_model"):
        combined = _lr_prob(S["combined_model"], flat)
    V = S.get("verdict", {})
    det_flags = [layers.get("detectors", {}).get("flag")] if "detectors" in layers else []
    n_flag = sum(1 for k, v in layers.items() if v.get("flag"))
    if combined is None:
        verdict = "UNCALIBRATED"
    elif combined <= V.get("clean_max", 0.2) and n_flag == 0:
        verdict = "CLEAN"
    elif combined >= V.get("flag_min", 0.5) or n_flag >= 2 or any(det_flags):
        verdict = "FLAGGED"
    else:
        verdict = "BORDERLINE"
    targets = sorted([r for r in rows if r["diagnostic"] and r["beyond_human_p90"] and r["layer"] != "detectors"],
                     key=lambda r: (-r["auroc"] * (r["ai_side_percentile"] - 50)))
    # overshoot: a discriminative feature pushed past the human band on the non-AI side
    overshoot = [r for r in rows if r["diagnostic"] and r["layer"] != "detectors" and r["ai_side_percentile"] <= 2.5]
    return {"verdict": verdict, "combined_p_ai": None if combined is None else round(combined, 3),
            "combined_cv_auroc": S.get("combined_model", {}).get("cv_auroc"), "layers": layers,
            "n_layers_flagged": n_flag, "features": rows,
            "targets": [{**t, "tactic": TACTICS.get(t["feature"], "")} for t in targets[:8]],
            "overshoot": [{k: r[k] for k in ("feature", "value", "human_median", "human_percentile")} for r in overshoot],
            "reference_note": S.get("note", "")}


# ------------------------------------------------------------------ held-out gate, controls, edit depth

def score_controls(ctrl_dir: str, names) -> dict:
    """Held-out detector scores on human control texts (pre-2023, same register)."""
    import detectors
    out = {}
    for f in sorted(Path(ctrl_dir).glob("*")):
        if f.suffix.lower() in (".md", ".txt") and f.is_file():
            r = detectors.run_open(read_text(str(f)), names)
            out[f.name] = {k: round(v["p_ai"], 3) for k, v in r.items() if "p_ai" in v}
    return out


def detector_gate(res: dict, ev: dict, controls: dict | None = None, thr: float = 0.5) -> dict:
    held = {}
    for k, v in res.get("detectors", {}).get("open", {}).items():
        if "p_ai" in v:
            held[k] = round(float(v["p_ai"]), 3)
    for k, v in res.get("detectors", {}).get("external", {}).items():
        if v.get("p_ai") is not None:
            held[f"{k} (API)"] = round(float(v["p_ai"]), 3)
    unin = set()
    for scores in (controls or {}).values():
        unin |= {k for k, p in scores.items() if p >= thr}
    high = [k for k, p in held.items() if p >= thr and k not in unin]
    gate = {"held_out": held, "high": high, "uninformative_on_controls": sorted(unin), "threshold": thr,
            "controls": controls or {}}
    if ev.get("verdict") in ("CLEAN", "BORDERLINE") and high:
        ev["verdict_calibrated"] = ev["verdict"]
        ev["verdict"] = "DETECTORS DISAGREE"
    ev["detector_gate"] = gate
    return gate


def edit_depth(btext: str, text: str) -> dict:
    import difflib
    so = split_sentences(strip_markup(btext))
    sr = split_sentences(strip_markup(text))
    kept = set(so)
    changed = sum(1 for x in sr if x not in kept)
    ratio = difflib.SequenceMatcher(None, btext.split(), text.split()).ratio()
    po = [len(p.split()) for p in strip_markup(btext).split("\n\n") if p.strip()]
    pr = [len(p.split()) for p in strip_markup(text).split("\n\n") if p.strip()]
    return {"sentences_changed": changed, "sentences_total": len(sr),
            "share_changed": round(changed / max(1, len(sr)), 2), "word_similarity": round(ratio, 2),
            "paragraphs_before": len(po), "paragraphs_after": len(pr),
            "level": "polish" if changed / max(1, len(sr)) < 0.5 and ratio > 0.8 else "restructure"}


def next_step(ev: dict, depth: dict | None, structure: dict | None) -> list[str]:
    v = ev.get("verdict")
    gate = ev.get("detector_gate", {})
    L = []
    flagged = []
    if structure:
        flagged = [i + 1 for i, r in enumerate(structure["paragraphs"]) if r["flags"]]
    if v == "DETECTORS DISAGREE":
        L.append(f"Do not stop. Calibrated verdict is {ev.get('verdict_calibrated')}, but held-out detector(s) "
                 f"{', '.join(gate['high'])} score the text >= {gate['threshold']}. Sentence-level edits rarely move "
                 "these; the signal is usually in how paragraphs are built.")
        if depth and depth["level"] == "polish":
            L.append(f"The last round was polish ({depth['sentences_changed']}/{depth['sentences_total']} sentences changed, "
                     f"word similarity {depth['word_similarity']}). Do Pass A0 (paragraph restructure) next.")
        elif depth and depth["level"] == "restructure":
            L.append("The last round already restructured the text. If the detectors stay high after one restructure "
                     "round, the remaining signal is probably the content itself: ask the author to write 2-3 sentences "
                     "per section in their own words, then fit and fact-check them. Do not keep iterating.")
        else:
            L.append("Go to Pass A0 (paragraph restructure).")
        if flagged:
            L.append(f"Start with paragraphs {flagged} (see Structure).")
    elif v == "CLEAN" and structure and structure["doc"]["flags"]:
        L.append("Calibrated layers and held-out detectors are clean; document-level structure flags remain (see "
                 "Structure). Fix them only if a reader would notice.")
    if not gate.get("held_out"):
        L.append("No held-out detector was run; the verdict is calibrated-only.")
    if gate.get("uninformative_on_controls"):
        L.append(f"Uninformative here (score >= {gate['threshold']} on human controls): "
                 f"{', '.join(gate['uninformative_on_controls'])}.")
    elif not gate.get("controls"):
        L.append("No human controls were scored (--controls). Some detectors flag dense technical human prose; "
                 "read their scores with that in mind.")
    return L


# ------------------------------------------------------------------ hotspots

def hotspots(res: dict, k: int = 6) -> list[dict]:
    import numpy as np
    score: dict[str, dict] = {}
    lm = res.get("lm", {})
    sent_rows = lm.get("sentences", [])
    # the most machine-like ~30% of sentences in this document (relative, because absolute
    # per-sentence values depend on the scoring model and sentence length)
    cut = max(0.0, float(np.percentile([s["fdg_local"] for s in sent_rows], 70))) if sent_rows else 0.0
    for s in sent_rows:
        score.setdefault(s["text"], {"sentence": s["text"], "reasons": [], "score": 0.0})
        z = s["fdg_local"]
        if z >= cut and z > 0:
            score[s["text"]]["reasons"].append(f"predictable wording (local FDG {z:.2f}, top-10 {s['top10_rate']:.0%})")
            score[s["text"]]["score"] += 5 * z
    srf = res.get("surface", {})
    sents = list(score) or split_sentences(strip_markup(res.get("_text", "")))
    for h in srf.get("hits", []):
        for s in sents:
            if h["match"][:40] in s:
                score.setdefault(s, {"sentence": s, "reasons": [], "score": 0.0})
                score[s]["reasons"].append(f"surface:{h['family']} '{h['match'][:40]}'")
                score[s]["score"] += h["weight"] / 2
                break
    ranked = sorted((v for v in score.values() if v["reasons"]), key=lambda v: -v["score"])
    return ranked[:k]


# ------------------------------------------------------------------ report

def _fmt(v):
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def markdown(res: dict, ev: dict, base: dict | None, guard: dict | None, depth: dict | None = None) -> str:
    L = [f"# Check round report", "",
         f"Words: {res['n_words']}" + ("  (below 150 words most signals are unreliable)" if res["n_words"] < 150 else "")]
    if "verdict" in ev:
        cal = f" (calibrated: {ev['verdict_calibrated']})" if ev.get("verdict_calibrated") else ""
        L += [f"Verdict: **{ev['verdict']}**{cal}  |  combined p(AI-reference) = {_fmt(ev['combined_p_ai'])}"
              f" (cv AUROC {_fmt(ev.get('combined_cv_auroc'))})  |  layers flagged: {ev['n_layers_flagged']}", ""]
        ns = next_step(ev, depth, res.get("structure"))
        if ns:
            L += ["## Next step", ""] + [f"- {x}" for x in ns] + [""]
        if depth:
            L += [f"Edit depth vs baseline: {depth['sentences_changed']}/{depth['sentences_total']} sentences changed "
                  f"({depth['share_changed']:.0%}), word similarity {depth['word_similarity']}, paragraphs "
                  f"{depth['paragraphs_before']} -> {depth['paragraphs_after']}  ->  **{depth['level']}**", ""]
        L += ["| layer | p(AI) | diagnostic features | beyond human p90 | mean AI-side pct | flag |", "|---|---|---|---|---|---|"]
        for k, v in ev["layers"].items():
            L.append(f"| {k} | {_fmt(v.get('p_ai', '-'))} | {v['n_diagnostic']} | {v['n_beyond_p90']} | {v['mean_ai_side_percentile']} | {'YES' if v.get('flag') else 'no'} |")
        L.append("")
    else:
        L += [ev.get("warning", ""), ""]
    det = res.get("detectors", {})
    if det:
        L += ["## Held-out detectors", ""]
        L += ["Not sentence-level rewrite targets: do not tune wording against them. They do decide whether to stop "
              "(see Verdict) and, per paragraph, where to restructure. A move from 'AI' to a 'humanized' class is "
              "not a move to 'human'.", ""]
        for k, v in det.get("open", {}).items():
            line = f"- {k}: " + (f"p(AI) = {v['p_ai']:.3f}" if "p_ai" in v else v.get("error", ""))
            if v.get("p_humanized") is not None:
                line += f"; p(humanized class) = {v['p_humanized']:.3f}"
            L.append(line)
        for k, v in det.get("external", {}).items():
            if v.get("p_ai") is not None:
                extra = {x: v[x] for x in ("model_version", "predicted_class", "ai_subclass", "humanized_windows",
                                           "low_confidence", "warning_length") if v.get(x) is not None}
                L.append(f"- {k} (API): p(AI) = {_fmt(v['p_ai'])}" + (f" {json.dumps(extra)}" if extra else ""))
            else:
                L.append(f"- {k} (API): {json.dumps(v)[:200]}")
        ctrl = ev.get("detector_gate", {}).get("controls")
        if ctrl:
            L += ["", "Human controls (same detectors):", ""] + [f"- {k}: {json.dumps(v)}" for k, v in ctrl.items()]
        L.append("")
    if res.get("structure"):
        import structure_scan
        L += structure_scan.markdown(res["structure"])
    fix = [t for t in ev.get("targets", []) if not t.get("voice_consistent")]
    own = [t for t in ev.get("targets", []) if t.get("voice_consistent")]
    if fix:
        L += ["## What to fix next (most discriminative features on the AI side of the human band)", ""]
        for t in fix:
            L.append(f"- `{t['feature']}` = {_fmt(t['value'])} (human median {_fmt(t['human_median'])}, AI median "
                     f"{_fmt(t['ai_median'])}; AI-side pct {t['ai_side_percentile']}, AUROC {t['auroc']:.2f}). {t['tactic']}")
        L.append("")
    if own:
        L += ["## Generic targets that match the writer's own style (keep; held-out detectors may still react to them)", ""]
        for t in own:
            L.append(f"- `{t['feature']}` = {_fmt(t['value'])}: the writer's own median is {_fmt(t['author_median'])} "
                     f"(AI-side pct {t['author_ai_side_percentile']} on the generic human band). Not an edit target.")
        L.append("")
    if ev.get("voice"):
        import voice_profile
        L += voice_profile.markdown(ev["voice"])
    sp = res.get("stylometry", {}).get("spans") or {}
    if sp.get("sentences"):
        L += ["## Syntax hotspots (Pass B targets: register markers by sentence)", "",
              f"Totals: {json.dumps(sp.get('totals', {}))}", ""]
        for r in sp["sentences"]:
            mk = "; ".join(f"{k}: '{v}'" for k, v in r["markers"][:6])
            L.append(f"- \"{r['sentence'][:220]}\" -- {mk}")
        if sp.get("elaborate_once"):
            L += ["", "Long words used only once (candidates for a plainer word or for reusing the key term): "
                  + ", ".join(sp["elaborate_once"])]
        L.append("")
    if ev.get("overshoot"):
        L += ["## Overshoot (beyond the human band on the non-AI side: pull back toward the human median)", ""]
        for o in ev["overshoot"]:
            L.append(f"- `{o['feature']}` = {_fmt(o['value'])} (human median {_fmt(o['human_median'])}, "
                     f"human percentile {o['human_percentile']})")
        L.append("")
    hs = hotspots(res)
    if hs:
        L += ["## Hotspot sentences", ""]
        for h in hs:
            L.append(f"- \"{h['sentence'][:220]}\" -- " + "; ".join(h["reasons"][:3]))
        L.append("")
    runs = res.get("lm", {}).get("predictable_runs", [])
    if runs:
        L += ["## Most predictable token runs (rank-1 under the observer LM)", ""] + [f"- \"{r}\"" for r in runs[:6]] + [""]
    srf = res.get("surface", {})
    if srf.get("family_counts") or srf.get("ai_vocab_hits"):
        L += ["## Surface tells", "", f"Families: {json.dumps(srf.get('family_counts', {}))}",
              f"AI-vocabulary hits: {json.dumps(dict(list(srf.get('ai_vocab_hits', {}).items())[:15]))}",
              f"Dashes: {srf['counts']['em_or_spaced_dash']}, one-line closers: {srf['counts']['one_line_closers']}", ""]
    au = res.get("embedding", {}).get("author_style")
    if au and not ev.get("voice"):
        L += ["## Voice match to author samples", "", "```", json.dumps(au, indent=1), "```", ""]
    if base is not None:
        L += ["## Change from baseline", "", "| feature | baseline | now | toward human? |", "|---|---|---|---|"]
        S = {r["feature"]: r for r in ev.get("features", [])}
        for name, r in sorted(S.items(), key=lambda kv: -kv[1]["auroc"])[:20]:
            if name in base:
                b, n = base[name], r["value"]
                better = (n < b) if r["direction"] == "higher_is_ai" else (n > b)
                L.append(f"| {name} | {_fmt(b)} | {_fmt(n)} | {'yes' if better else 'no'} |")
        L.append("")
    if guard is not None:
        import meaning_guard
        L += ["## Meaning guard", ""]
        if guard.get("mode") != "restructure":
            L.append(f"doc cosine {guard['doc_cosine']:.3f}, coverage {guard['coverage']:.2f}, support "
                     f"{guard['support']:.2f}, length ratio {guard['length_ratio_words']:.2f}")
        L += ["Every item is listed; none are summarised away.", "", "```", meaning_guard.report(guard), "```", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--baseline", help="original text: adds per-feature deltas and the meaning guard")
    ap.add_argument("--reference", default=str(DEFAULT_REFERENCE), help="calibrated reference bank directory")
    ap.add_argument("--author", help="directory with >= 3 samples of the author's own writing")
    ap.add_argument("--profile", help="writer voice profile: directory, name under ~/.deep-humanizer/profiles, or voice.md "
                                      "path (default: $DEEP_HUMANIZER_PROFILE). Guides edits; never changes the verdict")
    ap.add_argument("--layers", default="surface,stylometry,lm,embedding,detectors")
    ap.add_argument("--fast", action="store_true", help="surface+stylometry+lm without Binoculars")
    ap.add_argument("--open-detectors", default=None, help="comma list (default: tabularisai,desklib,fakespot,tropa_mini)")
    ap.add_argument("--final", action="store_true", help="final confirmation: all open detectors + external")
    ap.add_argument("--external", default="", help="commercial providers, or 'available' for all with keys set (sends text out)")
    ap.add_argument("--controls", help="directory of human control texts (pre-2023, same register) scored by the held-out detectors")
    ap.add_argument("--no-structure", action="store_true", help="skip the paragraph-level structure scan")
    ap.add_argument("--nli", action="store_true", help="meaning guard with NLI (default on with --restructure)")
    ap.add_argument("--restructure", action="store_true", help="meaning guard in restructure mode (claim-level, for paragraph rewrites)")
    ap.add_argument("--cuts", help="planned cuts for the restructure meaning guard, one per line")
    ap.add_argument("--json", help="write full results JSON")
    ap.add_argument("--md", help="write markdown report here (default stdout)")
    a = ap.parse_args()

    text = read_text(a.path)
    layers = ("surface", "stylometry", "lm") if a.fast else tuple(a.layers.split(","))
    od = None
    if a.open_detectors:
        od = a.open_detectors.split(",")
    if a.final:
        import detectors
        od = list(detectors.FINAL_OPEN)
    ref = Path(a.reference) if a.reference else None
    prof = None
    import os
    if a.profile or os.environ.get("DEEP_HUMANIZER_PROFILE"):
        import voice_profile
        prof = voice_profile.load_profile(a.profile)
    author = a.author or (prof.get("samples_dir") if prof else None)
    res = extract_all(text, layers, str(ref) if ref and embed_geometry.has_bank(ref) else "none", author,
                      performer=not a.fast, open_detectors=od)
    res["_text"] = text
    if a.external:
        import detectors
        names = detectors.available_external() if a.external == "available" else a.external.split(",")
        res.setdefault("detectors", {})["external"] = detectors.run_external(text, names)
    ev = evaluate(res["flat"], ref) if ref else {}
    if "verdict" in ev:
        ctrl = score_controls(a.controls, od) if a.controls and "detectors" in layers else None
        detector_gate(res, ev, ctrl)
    if prof is not None:
        import voice_profile
        if "verdict" in ev and (ref / "feature_stats.json").exists():
            voice_profile.annotate_targets(ev, prof, json.loads((ref / "feature_stats.json").read_text()), _percentile)
        ev["voice"] = voice_profile.compare(text, res, prof)
    if not a.no_structure:
        import structure_scan
        gate = ev.get("detector_gate", {})
        full = {k: v["p_ai"] for k, v in res.get("detectors", {}).get("open", {}).items() if "p_ai" in v}
        res["structure"] = structure_scan.scan(text, od, run_detectors="detectors" in layers, full_scores=full,
                                               skip=set(gate.get("uninformative_on_controls", [])))
    base_flat, guard, depth = None, None, None
    if a.baseline:
        btext = read_text(a.baseline)
        base_flat = extract_all(btext, layers, str(ref) if ref and embed_geometry.has_bank(ref) else "none",
                                None, performer=not a.fast, open_detectors=od)["flat"]
        import meaning_guard
        if a.restructure:
            cuts = Path(a.cuts).read_text().splitlines() if a.cuts else None
            guard = meaning_guard.guard_restructure(btext, text, cuts)
        else:
            guard = meaning_guard.guard(btext, text, use_nli=a.nli)
        depth = edit_depth(btext, text)
    md = markdown(res, ev, base_flat, guard, depth)
    res.pop("_text", None)
    if a.json:
        dump_json({"results": res, "evaluation": ev, "baseline_flat": base_flat, "meaning_guard": guard,
                   "edit_depth": depth}, a.json)
    if a.md:
        Path(a.md).write_text(md)
    else:
        print(md)


if __name__ == "__main__":
    main()
