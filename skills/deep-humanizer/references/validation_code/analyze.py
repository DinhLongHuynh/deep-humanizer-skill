"""Score all validation conditions with the check round (train-split bank), held-out detectors and the
meaning guard; write a tidy table. Exec'd inside the humanizer-nlp kernel after deep_loop has run.
Conditions: human (held-out human texts), ai (raw generations), shallow (single-prompt LLM 'make it human'
rewrite), deep_v1 (skill without syntax hotspots), deep (final skill)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "deep-humanizer/scripts")
import check  # noqa: E402
import detectors  # noqa: E402
import meaning_guard  # noqa: E402
import embed_geometry  # noqa: E402

REF = Path("validation/bank_train")
HELD_OUT = ["tabularisai", "desklib", "fakespot", "tropa_mini"]
KEY_FEATS = ["srf.surface_score_per_1k", "srf.ai_vocab", "srf.em_dash", "sty.present_participial_clause_per_1k",
             "sty.nominalization_per_1k", "sty.phrasal_coordination_per_1k", "sty.agentless_passive_per_1k",
             "sty.sent_len_cv", "sty.mtld", "sty.contraction_per_1k", "lm.fastdetectgpt_norm", "lm.binoculars",
             "lm.surprisal_sd", "lm.uid_span_range", "lm.late_local_volatility", "emb.phd", "emb.mle_id",
             "emb.pairwise_cos_mean", "emb.ref.emb_lr_p_ai"]


def score(text, original=None):
    res = check.extract_all(text, ("surface", "stylometry", "lm", "embedding"), reference=str(REF), performer=True)
    ev = check.evaluate(res["flat"], REF)
    row = {"combined": ev.get("combined_p_ai"), "verdict": ev.get("verdict"), "n_words": res["n_words"]}
    for k, v in ev.get("layers", {}).items():
        row[f"layer.{k}"] = v.get("p_ai")
    for f in KEY_FEATS:
        row[f] = res["flat"].get(f)
    for k, v in detectors.run_open(text, HELD_OUT).items():
        row[f"det.{k}"] = v.get("p_ai")
        if v.get("p_humanized") is not None:
            row["det.tropa_p_humanized"] = v["p_humanized"]
    if original is not None:
        g = meaning_guard.guard(original, text)
        row.update({"guard": g["verdict"], "doc_cos": g["doc_cosine"], "coverage": g["coverage"],
                    "n_unsupported": len(g["unsupported_sentences"]),
                    "n_fact_diffs": sum(len(v["missing"]) + len(v["added"]) for v in g["fact_diffs"].values()),
                    "len_ratio": g["length_ratio_words"]})
    row["_emb"] = embed_geometry.doc_embedding(text)[0].tolist()
    return row


def run(test, shallow, deep_versions, log=print, cache_path="validation/scores_cache.json"):
    """deep_versions: {"deep_v1": deep_out_v1, "deep": deep_out_v2}. Scores are cached by condition/id."""
    import os
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    def sc(key, text, original=None):
        if key not in cache:
            cache[key] = score(text, original)
            json.dump(cache, open(cache_path, "w"))
        return cache[key]
    ai_by_group = {r["group"]: r for r in test if r["label"] == "ai"}
    rows = []
    for r in test:
        cond = "human" if r["label"] == "human" else "ai"
        rows.append({"id": r["id"], "group": r["group"], "register": r["register"], "generator": r.get("generator"),
                     "condition": cond, **sc(f"{cond}/{r['id']}", r["text"])})
    log("scored human + ai")
    for s in shallow:
        rows.append({"id": s["id"], "group": s["group"], "register": s["register"], "generator": s.get("generator"),
                     "condition": "shallow", **sc(f"shallow/{s['id']}", s["text"], ai_by_group[s["group"]]["text"])})
    log("scored shallow")
    gid = {r["id"]: r for r in test}
    for cond, outs in deep_versions.items():
        for d in outs:
            src = gid[d["id"]]
            rows.append({"id": d["id"], "group": src["group"], "register": src["register"],
                         "generator": src.get("generator"), "condition": cond, "best_pass": d["best_pass"],
                         **sc(f"{cond}/{d['id']}", d["best_text"], src["text"])})
        log(f"scored {cond}")
    return pd.DataFrame(rows)


def summarise(df):
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, "deep-humanizer/scripts")
    import batch_compare
    H = df[df.condition == "human"]
    out = {}
    for c in [x for x in ["ai", "shallow", "deep_v1", "deep"] if x in set(df.condition)]:
        C = df[df.condition == c]
        o = {"n": len(C), "combined_mean": C.combined.mean(), "combined_median": C.combined.median(),
             "verdict": C.verdict.value_counts().to_dict()}
        for col in [x for x in df.columns if x.startswith(("layer.", "det."))]:
            y = np.r_[np.zeros(len(H)), np.ones(len(C))]
            v = np.r_[H[col].to_numpy(float), C[col].to_numpy(float)]
            ok = ~np.isnan(v)
            o[f"auroc_vs_human.{col}"] = float(roc_auc_score(y[ok], v[ok])) if ok.sum() > 10 else None
            o[f"mean.{col}"] = float(np.nanmean(C[col]))
            if col.startswith("det."):
                o[f"flag_rate_0.5.{col}"] = float((C[col] >= 0.5).mean())
        y = np.r_[np.zeros(len(H)), np.ones(len(C))]
        o["auroc_vs_human.combined"] = float(roc_auc_score(y, np.r_[H.combined, C.combined]))
        Eh, Ec = np.stack(H._emb.to_list()), np.stack(C._emb.to_list())
        o["c2st_auroc_vs_human"] = batch_compare.c2st(Ec, Eh)
        o["mmd2_vs_human"], o["mmd_p_vs_human"] = batch_compare.mmd2(Ec, Eh, n_perm=300)
        if "guard" in C:
            o["guard_pass_rate"] = float((C.guard == "PASS").mean())
            o["doc_cos_mean"] = float(C.doc_cos.mean())
            o["fact_diff_docs"] = int((C.n_fact_diffs > 0).sum())
            o["len_ratio_mean"] = float(C.len_ratio.mean())
        out[c] = o
    out["human"] = {"n": len(H), "combined_mean": H.combined.mean(), "verdict": H.verdict.value_counts().to_dict(),
                    **{f"mean.{c}": float(np.nanmean(H[c])) for c in df.columns if c.startswith(("layer.", "det."))}}
    # feature movement: share of the human-AI median gap closed (0 = raw AI, 1 = human median)
    mv = {}
    for f in KEY_FEATS:
        hm, am = H[f].median(), df[df.condition == "ai"][f].median()
        if am == hm or np.isnan(am) or np.isnan(hm):
            continue
        mv[f] = {c: float((df[df.condition == c][f].median() - am) / (hm - am)) for c in ["shallow", "deep_v1", "deep"] if c in set(df.condition)}
        mv[f]["human_median"], mv[f]["ai_median"] = float(hm), float(am)
    out["gap_closed"] = mv
    return out
