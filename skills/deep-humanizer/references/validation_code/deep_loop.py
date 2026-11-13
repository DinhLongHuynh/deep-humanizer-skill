"""Validation harness: emulate an agent following deep-humanizer/SKILL.md on one document.

Loop (SKILL.md Steps 1-4): diagnose with the local layers (surface, stylometry, LM, embedding;
detectors held out) -> targeted rewrite by an LLM given the skill's pass instructions and the check
report -> meaning gate (fix once if REVIEW) -> re-check -> up to MAX_PASSES -> select the version
with the lowest combined p among versions that pass the meaning gate.
Runs in-process so models load once. Requires `host` (Claude Science kernel) for LLM calls.
"""
import json
import re
import sys
from pathlib import Path

SK = Path("deep-humanizer")
sys.path.insert(0, str(SK / "scripts"))
import check  # noqa: E402
import meaning_guard  # noqa: E402

LAYERS = ("surface", "stylometry", "lm", "embedding")
MAX_PASSES = 3

_skill = (SK / "SKILL.md").read_text()
PASSES = _skill[_skill.index("### Step 2"):_skill.index("### Step 3")]
_tac = (SK / "references" / "tactics.md").read_text()
CODE_KEY = _tac[_tac.index("## Code key"):_tac.index("## Full tactic map")]
_sp = (SK / "references" / "surface_patterns.md").read_text()
SURFACE = _sp[_sp.index("## Staging instead"):_sp.index("## Integrity flags")]

SYSTEM = f"""You are an expert editor applying the deep-humanizer skill. You revise a draft so it reads like a careful human
writer of its register, guided by a measurement report. Follow these instructions exactly.

{PASSES}

Report codes:
{CODE_KEY}

Surface pattern catalogue:
{SURFACE}

Hard rules: keep every claim, number, name, date, citation and quotation of the ORIGINAL; never add facts, names,
numbers, events or details that are not in the original; keep roughly the original length (within 15 percent) and the
paragraph structure appropriate to the register; edit the targeted sentences and leave sentences that already read
well; do not use character tricks or deliberate errors. Return only the revised text, no commentary."""

REG_NOTE = {"academic": "a research-paper abstract (formal academic register: no contractions, no first-person singular, no rhetorical questions)",
            "news": "a news article (journalistic register: attributive verbs, concrete detail from the source, no editorializing)",
            "fiction": "a short story responding to a writing prompt (literary register)"}


def check_text(text, ref, baseline=None):
    res = check.extract_all(text, LAYERS, reference=str(ref), author=None, performer=True, open_detectors=None)
    res["_text"] = text
    ev = check.evaluate(res["flat"], ref)
    guard = meaning_guard.guard(baseline, text) if baseline else None
    md = check.markdown(res, ev, None, guard)
    return res, ev, guard, md


def guard_issues(g):
    out = []
    for s in g.get("lost_sentences", []):
        out.append(f"LOST (restore its content): {s['sentence']}")
    for s in g.get("unsupported_sentences", []):
        out.append(f"UNSUPPORTED (remove or rewrite so it only states what the original says): {s['sentence']}")
    for k, v in g.get("fact_diffs", {}).items():
        if v["missing"]:
            out.append(f"MISSING {k} (restore exactly): {v['missing']}")
        if v["added"] and k in ("numbers", "citations", "urls", "quotes", "entities"):
            out.append(f"ADDED {k} not in the original (remove unless only reformatted): {v['added']}")
    return out


def llm(prompt, model):
    last = None
    for mt in (8000, 16000):
        try:
            r = host.llm({"prompt": prompt, "system": SYSTEM, "model": model, "max_tokens": mt})  # noqa: F821
            if r.get("text", "").strip():
                return r["text"].strip()
        except Exception as exc:  # empty response at max_tokens etc.
            last = exc
    raise RuntimeError(f"rewrite failed twice: {last!r}")


def _rewrite_prompt(doc, cur, md, p):
    reg = REG_NOTE.get(doc.get("register"), "prose")
    return (f"Register: {reg}.\n\nCHECK REPORT for the current version (pass {p - 1}):\n{md}\n\n"
            f"ORIGINAL (source of all facts):\n{doc['text']}\n\nCURRENT VERSION to revise:\n{cur}\n\n"
            "Revise the CURRENT VERSION following the passes (A discourse, B syntax, C word choice, D surface), "
            "concentrating on the listed targets and hotspot sentences. Return only the revised text.")


def _llm_many(prompts, model):
    outs = host.llm([{"prompt": q, "system": SYSTEM, "model": model, "max_tokens": 8000} for q in prompts],  # noqa: F821
                    max_concurrency=8)
    res = []
    for q, o in zip(prompts, outs):
        t = o.get("text") if isinstance(o, dict) else None
        res.append(t.strip() if t and t.strip() else llm(q, model))
    return res


def deep_humanize_batch(docs, ref, model, out_path, log=print):
    state = {}
    for d in docs:
        _, ev, _, md = check_text(d["text"], ref)
        state[d["id"]] = {"doc": d, "cur": d["text"], "md": md, "done": False,
                          "history": [{"pass": 0, "text": d["text"], "combined": ev.get("combined_p_ai"),
                                       "verdict": ev.get("verdict"), "guard": "PASS",
                                       "layers": {k: v.get("p_ai") for k, v in ev.get("layers", {}).items()}}]}
    log(f"pass 0 checked {len(docs)}")
    for p in range(1, MAX_PASSES + 1):
        act = [s for s in state.values() if not s["done"]]
        if not act:
            break
        news = _llm_many([_rewrite_prompt(s["doc"], s["cur"], s["md"], p) for s in act], model)
        guards = [meaning_guard.guard(s["doc"]["text"], n) for s, n in zip(act, news)]
        bad = [i for i, g in enumerate(guards) if g["verdict"] != "PASS"]
        if bad:
            fixes = [(f"ORIGINAL:\n{act[i]['doc']['text']}\n\nREVISION:\n{news[i]}\n\nThe meaning check found these "
                      "problems:\n- " + "\n- ".join(guard_issues(guards[i])) + "\n\nCorrect only these problems in the "
                      "REVISION, keeping its style changes. Return only the corrected revision.") for i in bad]
            fixed = _llm_many(fixes, model)
            for i, t in zip(bad, fixed):
                news[i] = t
                guards[i] = meaning_guard.guard(act[i]["doc"]["text"], t)
        for s, new, g in zip(act, news, guards):
            _, ev, _, md = check_text(new, ref)
            s["history"].append({"pass": p, "text": new, "combined": ev.get("combined_p_ai"), "verdict": ev.get("verdict"),
                                 "guard": g["verdict"], "doc_cos": g["doc_cosine"], "coverage": g["coverage"],
                                 "n_unsupported": len(g["unsupported_sentences"]), "fact_diffs": g["fact_diffs"],
                                 "fixed_after_guard": None,
                                 "layers": {k: v.get("p_ai") for k, v in ev.get("layers", {}).items()}})
            s["cur"], s["md"] = new, md
            s["done"] = ev.get("verdict") == "CLEAN" and g["verdict"] == "PASS"
        log(f"pass {p}: rewrote {len(act)}, meaning fixes {len(bad)}, clean+pass "
            f"{sum(1 for s in state.values() if s['done'])}")
        json.dump({k: {"history": v["history"]} for k, v in state.items()}, open(out_path, "w"), indent=1)
    out = []
    for k, s in state.items():
        h = s["history"]
        cands = [x for x in h[1:] if x["guard"] == "PASS"] or h[1:]
        best = min(cands, key=lambda x: x["combined"] if x["combined"] is not None else 9)
        out.append({"id": k, "best_pass": best["pass"], "best_text": best["text"], "selected_guard": best["guard"],
                    "history": h})
    json.dump(out, open(out_path, "w"), indent=1)
    return out


# ---- stepwise variant: LLM calls executed elsewhere (requests/outputs exchanged as JSON)

def init_state(docs, ref):
    state = {}
    for d in docs:
        _, ev, _, md = check_text(d["text"], ref)
        state[d["id"]] = {"doc": d, "cur": d["text"], "md": md, "done": False,
                          "history": [{"pass": 0, "text": d["text"], "combined": ev.get("combined_p_ai"),
                                       "verdict": ev.get("verdict"), "guard": "PASS",
                                       "layers": {k: v.get("p_ai") for k, v in ev.get("layers", {}).items()}}]}
    return state


def rewrite_requests(state, p, model):
    return [{"key": k, "system": SYSTEM, "prompt": _rewrite_prompt(s["doc"], s["cur"], s["md"], p), "model": model}
            for k, s in state.items() if not s["done"]]


def fix_requests(state, news, model):
    """news: {key: text}. Returns (guards, fix requests for texts failing the meaning gate)."""
    guards, reqs = {}, []
    for k, t in news.items():
        g = meaning_guard.guard(state[k]["doc"]["text"], t)
        guards[k] = g
        if g["verdict"] != "PASS":
            reqs.append({"key": k, "system": SYSTEM, "model": model, "prompt":
                         f"ORIGINAL:\n{state[k]['doc']['text']}\n\nREVISION:\n{t}\n\nThe meaning check found these "
                         "problems:\n- " + "\n- ".join(guard_issues(g)) + "\n\nCorrect only these problems in the "
                         "REVISION, keeping its style changes. Return only the corrected revision."})
    return guards, reqs


def apply_pass(state, p, news, fixed, ref):
    for k, t in fixed.items():
        news[k] = t
    for k, new in news.items():
        g = meaning_guard.guard(state[k]["doc"]["text"], new)
        _, ev, _, md = check_text(new, ref)
        s = state[k]
        s["history"].append({"pass": p, "text": new, "combined": ev.get("combined_p_ai"), "verdict": ev.get("verdict"),
                             "guard": g["verdict"], "doc_cos": g["doc_cosine"], "coverage": g["coverage"],
                             "n_unsupported": len(g["unsupported_sentences"]), "fact_diffs": g["fact_diffs"],
                             "layers": {kk: v.get("p_ai") for kk, v in ev.get("layers", {}).items()}})
        s["cur"], s["md"] = new, md
        s["done"] = ev.get("verdict") == "CLEAN" and g["verdict"] == "PASS"


def finalize(state, out_path):
    out = []
    for k, s in state.items():
        h = s["history"]
        cands = [x for x in h[1:] if x["guard"] == "PASS"] or h[1:]
        best = min(cands, key=lambda x: x["combined"] if x["combined"] is not None else 9)
        out.append({"id": k, "best_pass": best["pass"], "best_text": best["text"], "selected_guard": best["guard"],
                    "history": h})
    json.dump(out, open(out_path, "w"), indent=1)
    return out


def deep_humanize(doc, ref, model, log=print):
    original = doc["text"]
    reg = REG_NOTE.get(doc.get("register"), "prose")
    _, ev0, _, md0 = check_text(original, ref)
    hist = [{"pass": 0, "text": original, "combined": ev0.get("combined_p_ai"), "verdict": ev0.get("verdict"),
             "guard": "PASS", "layers": {k: v.get("p_ai") for k, v in ev0.get("layers", {}).items()}}]
    cur, md = original, md0
    for p in range(1, MAX_PASSES + 1):
        prompt = (f"Register: {reg}.\n\nCHECK REPORT for the current version (pass {p - 1}):\n{md}\n\n"
                  f"ORIGINAL (source of all facts):\n{original}\n\n"
                  f"CURRENT VERSION to revise:\n{cur}\n\n"
                  "Revise the CURRENT VERSION following the passes (A discourse, B syntax, C word choice, D surface), "
                  "concentrating on the listed targets and hotspot sentences. Return only the revised text.")
        new = llm(prompt, model)
        g = meaning_guard.guard(original, new)
        if g["verdict"] != "PASS":
            fix = (f"ORIGINAL:\n{original}\n\nREVISION:\n{new}\n\nThe meaning check found these problems:\n- "
                   + "\n- ".join(guard_issues(g)) + "\n\nCorrect only these problems in the REVISION, keeping its "
                   "style changes. Return only the corrected revision.")
            new = llm(fix, model)
            g = meaning_guard.guard(original, new)
        _, ev, _, md = check_text(new, ref)
        hist.append({"pass": p, "text": new, "combined": ev.get("combined_p_ai"), "verdict": ev.get("verdict"),
                     "guard": g["verdict"], "doc_cos": g["doc_cosine"], "coverage": g["coverage"],
                     "n_unsupported": len(g["unsupported_sentences"]), "fact_diffs": g["fact_diffs"],
                     "layers": {k: v.get("p_ai") for k, v in ev.get("layers", {}).items()}})
        log(f"  pass {p}: combined {ev.get('combined_p_ai')} {ev.get('verdict')} guard {g['verdict']}")
        cur = new
        if ev.get("verdict") == "CLEAN" and g["verdict"] == "PASS":
            break
    cands = [h for h in hist[1:] if h["guard"] == "PASS"] or hist[1:]
    best = min(cands, key=lambda h: h["combined"] if h["combined"] is not None else 9)
    return {"id": doc["id"], "best_pass": best["pass"], "best_text": best["text"], "history": hist,
            "selected_guard": best["guard"]}
