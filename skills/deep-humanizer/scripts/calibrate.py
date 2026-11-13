#!/usr/bin/env python
"""Build a reference bank and calibrate the check round on labelled texts.

Input: JSONL, one document per line:
    {"id": "...", "text": "...", "label": "human" | "ai", "group": "topic-17", "register": "news"}
`group` ties matched human/AI documents on the same topic together so the
cross-validation never trains and tests on the same topic. Use texts of the
same register and length range as the texts you will check, and human texts
written before late 2022 (or otherwise known to be human).

Output directory (default assets/reference_default):
    reference.npz        document embeddings + labels + embedding-LR weights (k-NN / LR / density)
    feature_stats.json   per-feature human & AI quantiles, AUROC, direction; per-layer and combined
                         logistic models with grouped-CV AUROC; verdict cut-offs
    features.csv         the full feature table (out-of-fold values for reference-bank features)
    bank_meta.json       models used, counts, registers

The default bank shipped with the skill was built this way from the
validation corpus described in references/validation.md. For serious use,
build a bank for your own register: e.g. 100+ pre-2023 human texts from the
target venue plus LLM texts written to the same prompts, or the author's own
writing as the human class.

Usage:
    python calibrate.py corpus.jsonl --out ../assets/reference_default [--cache cache_dir] [--no-detectors]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hz_common import ASSETS, dump_json, model_id, to_jsonable  # noqa: E402

LAYERS = {"srf": "surface", "sty": "stylometry", "lm": "lm", "emb": "embedding", "det": "detectors"}
EXCLUDE = {"lm.fastdetectgpt_raw", "lm.logppl_performer", "lm.xppl", "lm.sent_surprisal_range",
           "srf.bold", "srf.pattern_hits"}


def load_jsonl(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    for r in rows:
        r["y"] = 1 if str(r["label"]).lower() in ("ai", "1", "machine", "llm") else 0
        r.setdefault("group", r.get("id"))
    return rows


PREFIX = {"surface": "srf.", "stylometry": "sty.", "lm": "lm.", "embedding": "emb.", "detectors": "det."}


def extract_corpus(rows, cache: Path, layers, open_detectors, refresh=()):
    import numpy as np
    import check
    import embed_geometry
    cache.mkdir(parents=True, exist_ok=True)
    flats, embs = [], []
    for i, r in enumerate(rows):
        cp = cache / f"{r['id']}.json"
        if cp.exists():
            d = json.loads(cp.read_text())
            if refresh:
                part = check.extract_all(r["text"], tuple(refresh), reference="none", author=None, performer=True,
                                         open_detectors=open_detectors)["flat"]
                pre = tuple(PREFIX[l] for l in refresh)
                d["flat"] = {k: v for k, v in d["flat"].items() if not k.startswith(pre)}
                d["flat"].update(part)
                cp.write_text(json.dumps(to_jsonable(d)))
        else:
            res = check.extract_all(r["text"], layers, reference="none", author=None, performer=True,
                                    open_detectors=open_detectors)
            vec, _, _ = embed_geometry.doc_embedding(r["text"])
            d = {"flat": res["flat"], "doc_emb": vec.tolist()}
            cp.write_text(json.dumps(to_jsonable(d)))
        flats.append(d["flat"])
        embs.append(d["doc_emb"])
        if (i + 1) % 10 == 0:
            print(f"[calibrate] {i + 1}/{len(rows)}", file=sys.stderr, flush=True)
    return flats, np.asarray(embs, dtype=np.float32)


def oof_reference_features(E, y, groups, k=15):
    """Out-of-fold k-NN AI fraction, embedding-LR probability and human-density percentile."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    n = len(y)
    knn = np.full(n, np.nan)
    lr = np.full(n, np.nan)
    dens = np.full(n, np.nan)
    gkf = GroupKFold(n_splits=min(5, len(set(groups))))
    for tr, te in gkf.split(E, y, groups):
        Etr, ytr = E[tr], y[tr]
        sims = E[te] @ Etr.T
        kk = min(k, len(tr) - 1)
        nn = np.argsort(-sims, axis=1)[:, :kk]
        knn[te] = ytr[nn].mean(1)
        mu, sd = Etr.mean(0), Etr.std(0) + 1e-6
        m = LogisticRegression(C=0.1, max_iter=2000).fit((Etr - mu) / sd, ytr)
        lr[te] = m.predict_proba((E[te] - mu) / sd)[:, 1]
        H = Etr[ytr == 0]
        if H.shape[0] > kk + 1:
            HH = H @ H.T
            np.fill_diagonal(HH, -np.inf)
            d_h = 1 - np.sort(HH, axis=1)[:, ::-1][:, kk - 1]
            d_te = 1 - np.sort(E[te] @ H.T, axis=1)[:, ::-1][:, kk - 1]
            dens[te] = [(d_h >= d).mean() * 100 for d in d_te]
    return knn, lr, dens


def fit_lr(X, y, groups, C=0.3):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    med = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), med, X)
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=min(5, len(set(groups)))).split(X, y, groups):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m = LogisticRegression(C=C, max_iter=5000).fit((X[tr] - mu) / sd, y[tr])
        oof[te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = LogisticRegression(C=C, max_iter=5000).fit((X - mu) / sd, y)
    return {"mu": mu.tolist(), "sd": sd.tolist(), "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]),
            "cv_auroc": float(roc_auc_score(y, oof))}, oof


def main():
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.linear_model import LogisticRegression

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus")
    ap.add_argument("--out", default=str(ASSETS / "reference_default"))
    ap.add_argument("--cache", default="calib_cache")
    ap.add_argument("--no-detectors", action="store_true")
    ap.add_argument("--open-detectors", default=None)
    ap.add_argument("--diagnostic-auroc", type=float, default=0.7)
    ap.add_argument("--refresh", default="", help="recompute these layers for cached docs (e.g. surface,stylometry)")
    a = ap.parse_args()

    rows = load_jsonl(a.corpus)
    layers = ("surface", "stylometry", "lm", "embedding") + (() if a.no_detectors else ("detectors",))
    od = a.open_detectors.split(",") if a.open_detectors else None
    refresh = [x for x in a.refresh.split(",") if x]
    flats, E = extract_corpus(rows, Path(a.cache), layers, od, refresh)
    y = np.array([r["y"] for r in rows])
    groups = np.array([str(r["group"]) for r in rows])
    df = pd.DataFrame(flats)
    knn, lrp, dens = oof_reference_features(E, y, groups)
    df["emb.ref.knn_ai_fraction"], df["emb.ref.emb_lr_p_ai"], df["emb.ref.human_density_percentile"] = knn, lrp, dens
    df = df[[c for c in df.columns if c not in EXCLUDE]]

    feats = {}
    for c in df.columns:
        v = df[c].to_numpy(dtype=float)
        ok = ~np.isnan(v)
        if ok.sum() < 10 or len(set(y[ok])) < 2 or np.nanstd(v) == 0:
            continue
        au = roc_auc_score(y[ok], v[ok])
        direction = "higher_is_ai" if au >= 0.5 else "lower_is_ai"
        au_eff = max(au, 1 - au)
        s = v[ok] if direction == "higher_is_ai" else -v[ok]
        fpr, tpr, thr = roc_curve(y[ok], s)
        j = int(np.argmax(tpr - fpr))
        q = np.linspace(0, 100, 21)
        feats[c] = {
            "layer": LAYERS[c.split(".")[0]], "auroc": round(float(au_eff), 4), "direction": direction,
            "youden_threshold": float(thr[j] if direction == "higher_is_ai" else -thr[j]),
            "human_quantiles": np.percentile(v[ok & (y == 0)], q).tolist(),
            "ai_quantiles": np.percentile(v[ok & (y == 1)], q).tolist(),
        }

    layer_models, oofs = {}, {}
    for lname in ("surface", "stylometry", "lm", "embedding"):
        cols = [c for c in feats if feats[c]["layer"] == lname]
        if len(cols) < 2:
            continue
        m, oof = fit_lr(df[cols].to_numpy(dtype=float), y, groups)
        m["features"] = cols
        fpr, tpr, thr = roc_curve(y, oof)
        m["threshold"] = 0.5
        layer_models[lname] = m
        oofs[lname] = oof
    comb_cols = [c for c in feats if feats[c]["layer"] != "detectors"]
    comb, comb_oof = fit_lr(df[comb_cols].to_numpy(dtype=float), y, groups, C=0.1)
    comb["features"] = comb_cols
    hp = comb_oof[y == 0]
    ap_ = comb_oof[y == 1]
    verdict = {
        "clean_max": float(np.percentile(hp, 75)),
        "flag_min": 0.5,
        "human_oof_p_ai_quantiles": np.percentile(hp, [5, 25, 50, 75, 95]).tolist(),
        "ai_oof_p_ai_quantiles": np.percentile(ap_, [5, 25, 50, 75, 95]).tolist(),
        "rule": "CLEAN if combined p <= clean_max (the human 75th percentile) and no layer flagged; FLAGGED if "
                "combined p >= flag_min, >= 2 layers flagged, or the detector layer is flagged; else BORDERLINE.",
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {"features": feats, "layer_models": layer_models, "combined_model": comb, "verdict": verdict,
             "diagnostic_auroc": a.diagnostic_auroc, "n_human": int((y == 0).sum()), "n_ai": int((y == 1).sum()),
             "note": f"Calibrated on {a.corpus} ({int((y == 0).sum())} human / {int((y == 1).sum())} AI). "
                     "Values outside this register/length range are less meaningful."}
    dump_json(stats, str(out / "feature_stats.json"))
    mu, sd = E.mean(0), E.std(0) + 1e-6
    elr = LogisticRegression(C=0.1, max_iter=2000).fit((E - mu) / sd, y)
    np.savez_compressed(out / "reference.npz", doc_emb=E, labels=y, lr_coef=elr.coef_[0].astype(np.float32),
                        lr_intercept=elr.intercept_.astype(np.float32), lr_mu=mu.astype(np.float32),
                        lr_sd=sd.astype(np.float32))
    z = np.load(out / "reference.npz")
    (out / "reference.json").write_text(json.dumps({k: np.round(z[k].astype(float), 6).tolist() for k in z.files}))
    regs = sorted({r.get("register", "unknown") for r in rows})
    dump_json({"sent_emb": model_id("sent_emb"), "observer": model_id("observer"), "performer": model_id("performer"),
               "encoder": model_id("encoder"), "n_human": int((y == 0).sum()), "n_ai": int((y == 1).sum()),
               "registers": regs, "corpus": str(a.corpus)}, str(out / "bank_meta.json"))
    df.insert(0, "label", y)
    df.insert(0, "id", [r["id"] for r in rows])
    df.insert(2, "register", [r.get("register", "") for r in rows])
    df.to_csv(out / "features.csv", index=False)
    top = sorted(feats.items(), key=lambda kv: -kv[1]["auroc"])[:15]
    print(json.dumps({"combined_cv_auroc": comb["cv_auroc"],
                      "layer_cv_auroc": {k: v["cv_auroc"] for k, v in layer_models.items()},
                      "top_features": [(k, v["auroc"], v["direction"]) for k, v in top], "verdict": verdict}, indent=1))


if __name__ == "__main__":
    main()
