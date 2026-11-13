#!/usr/bin/env python
"""Distribution-level check for a BATCH of documents.

A single rewritten document can sit inside the human band on every signal
and still belong to a batch that is easy to separate from human writing:
detection recovered with more documents per source even where single
documents evaded every detector (Rivera Soto et al., ICML 2026, arXiv:2505.14608). When the user produces many
texts (a blog series, a set of reviews, all sections of a thesis), check
them as a distribution:

  * MMD^2 (RBF kernel, median-heuristic bandwidth) between the batch's
    document embeddings and the human reference, with a permutation p-value;
    the same against the AI reference for scale.
  * Classifier two-sample test (C2ST): cross-validated AUROC of a logistic
    regression separating batch from human reference. 0.5 = indistinguishable.
  * Per-feature KS distance on cheap scalar features (surface + stylometry,
    optionally LM) between the batch and the reference's human rows, with a
    Bonferroni count, and a classifier two-sample test on those features.
Verdict: SEPARABLE if any C2ST AUROC >= 0.75, MMD p < 0.01 or >= 3 features
separate after Bonferroni; WEAKLY SEPARABLE at C2ST >= 0.62, p < 0.05 or 1-2
features; else INDISTINGUISHABLE. Embedding tests are topic-sensitive, the
feature tests are not; read both.

Usage:
    python batch_compare.py rewrites/ [--reference ../assets/reference_default] [--against my_writing/]
                            [--lm] [--json out.json]
`rewrites/` is a directory of .txt/.md files or a .jsonl with a "text" field.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hz_common import ASSETS, dump_json  # noqa: E402


def load_texts(path: str) -> list[str]:
    p = Path(path)
    if p.is_dir():
        return [f.read_text() for f in sorted(p.iterdir()) if f.suffix in (".txt", ".md")]
    return [json.loads(l)["text"] for l in p.read_text().splitlines() if l.strip()]


def mmd2(X, Y, n_perm: int = 500, seed: int = 0):
    import numpy as np
    from scipy.spatial.distance import cdist
    Z = np.vstack([X, Y])
    D = cdist(Z, Z, "sqeuclidean")
    bw = np.median(D[np.triu_indices_from(D, 1)])
    K = np.exp(-D / bw)
    n = len(X)

    def stat(idx):
        a, b = idx[:n], idx[n:]
        Kxx, Kyy, Kxy = K[np.ix_(a, a)], K[np.ix_(b, b)], K[np.ix_(a, b)]
        m, k = len(a), len(b)
        return ((Kxx.sum() - np.trace(Kxx)) / (m * (m - 1)) + (Kyy.sum() - np.trace(Kyy)) / (k * (k - 1))
                - 2 * Kxy.mean())
    idx = np.arange(len(Z))
    obs = stat(idx)
    rng = np.random.default_rng(seed)
    null = [stat(rng.permutation(idx)) for _ in range(n_perm)]
    return float(obs), float((np.sum(np.array(null) >= obs) + 1) / (n_perm + 1))


def c2st(X, Y, seed: int = 0):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    Z = np.vstack([X, Y])
    y = np.r_[np.ones(len(X)), np.zeros(len(Y))]
    k = int(min(5, len(X), len(Y)))
    if k < 2:
        return float("nan")
    p = cross_val_predict(LogisticRegression(C=0.1, max_iter=3000), Z, y,
                          cv=StratifiedKFold(k, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def main():
    import numpy as np
    import pandas as pd
    from scipy.stats import ks_2samp
    import check
    import embed_geometry

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("batch")
    ap.add_argument("--reference", default=str(ASSETS / "reference_default"))
    ap.add_argument("--against", help="optional second human batch (e.g. the author's own texts) instead of the bank")
    ap.add_argument("--lm", action="store_true", help="include LM features in the KS comparison (slow)")
    ap.add_argument("--json")
    a = ap.parse_args()

    texts = load_texts(a.batch)
    if len(texts) < 5:
        raise SystemExit("need >= 5 documents for a distribution test")
    Eb = np.stack([embed_geometry.doc_embedding(t)[0] for t in texts])
    z, _ = embed_geometry.load_bank(a.reference)
    if a.against:
        H = np.stack([embed_geometry.doc_embedding(t)[0] for t in load_texts(a.against)])
        A = None
    else:
        H, A = z["doc_emb"][z["labels"] == 0], z["doc_emb"][z["labels"] == 1]
    out = {"n_batch": len(texts), "n_human_ref": int(len(H))}
    out["mmd2_vs_human"], out["mmd_p_vs_human"] = mmd2(Eb, H)
    out["c2st_auroc_vs_human"] = c2st(Eb, H)
    if A is not None and len(A):
        out["mmd2_vs_ai"], out["mmd_p_vs_ai"] = mmd2(Eb, A)
        out["c2st_auroc_vs_ai"] = c2st(Eb, A)
        out["mmd2_human_vs_ai_scale"] = mmd2(H, A, n_perm=1)[0]

    layers = ("surface", "stylometry") + (("lm",) if a.lm else ())
    fb = pd.DataFrame([check.extract_all(t, layers, reference="none")["flat"] for t in texts])
    feats_csv = Path(a.reference) / "features.csv"
    ks = {}
    if feats_csv.exists() and not a.against:
        ref = pd.read_csv(feats_csv)
        ref_h = ref[ref.label == 0]
        stats = json.loads((Path(a.reference) / "feature_stats.json").read_text())["features"]
        for c in fb.columns:
            if c in ref_h and c in stats and stats[c]["auroc"] >= 0.65:
                r = ks_2samp(fb[c].dropna(), ref_h[c].dropna())
                ks[c] = {"ks": round(float(r.statistic), 3), "p": float(r.pvalue),
                         "batch_median": float(fb[c].median()), "human_median": float(ref_h[c].median()),
                         "auroc_in_calibration": stats[c]["auroc"]}
    out["feature_ks"] = dict(sorted(ks.items(), key=lambda kv: -kv[1]["ks"]))
    n_bonf = sum(1 for v in ks.values() if v["p"] < 0.05 / max(1, len(ks)))
    out["n_features_separating_bonferroni"] = n_bonf
    if ks:  # classifier two-sample test on the discriminative scalar features
        cols = list(ks)
        Xb = fb[cols].to_numpy(float)
        Xh = ref_h[cols].to_numpy(float)
        med = np.nanmedian(np.vstack([Xb, Xh]), axis=0)
        Xb, Xh = np.where(np.isnan(Xb), med, Xb), np.where(np.isnan(Xh), med, Xh)
        mu, sd = Xh.mean(0), Xh.std(0) + 1e-9
        out["c2st_features_auroc_vs_human"] = c2st((Xb - mu) / sd, (Xh - mu) / sd)
    cf = out.get("c2st_features_auroc_vs_human", 0.5)
    verdict = "INDISTINGUISHABLE"
    if out["c2st_auroc_vs_human"] >= 0.75 or cf >= 0.75 or out["mmd_p_vs_human"] < 0.01 or n_bonf >= 3:
        verdict = "SEPARABLE"
    elif out["c2st_auroc_vs_human"] >= 0.62 or cf >= 0.62 or out["mmd_p_vs_human"] < 0.05 or n_bonf >= 1:
        verdict = "WEAKLY SEPARABLE"
    out["verdict"] = verdict
    out["note"] = ("Embedding tests compare against the bank's human documents; if the batch's register/topic "
                   "differs from the bank, separability may reflect topic, not authorship. Use --against with "
                   "register-matched human texts for a fair test.")
    dump_json(out, a.json)


if __name__ == "__main__":
    main()
