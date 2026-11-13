#!/usr/bin/env python
"""Layer 4 - embedding-space geometry and distribution distance.

Three families of measurements:

1. Intrinsic dimension of the text's contextual token-embedding cloud
   (Tulchinskii et al., NeurIPS 2023). Each text is treated as a point cloud
   of RoBERTa token vectors; the persistent-homology dimension (PHD) of that
   cloud is consistently lower for LLM text than for human text in English
   (about 1.5 lower on average in the original study, with human text around
   9-10 for roberta-base). An MLE (Levina-Bickel) estimate is given as a
   cross-check. Needs >= ~120 tokens; noisy on short texts.

2. Sentence-embedding trajectory. Human prose wanders: adjacent-sentence
   similarity is less uniform and the document occupies a less regular
   region of embedding space. Reported: adjacent cosine mean / sd, mean
   pairwise cosine, dispersion around the centroid, effective rank.

3. Distance to reference distributions (needs a bank from
   build_reference.py): k-NN AI fraction, logistic-regression probability on
   document embeddings, and the document's human-density percentile (is it
   as close to its nearest human neighbours as human documents are to
   theirs?). With --author, a content-independent style embedding
   (Wegmann et al. 2022) is compared with the author's own samples.

Usage:
    python embed_geometry.py draft.md [--reference ../assets/reference_default] [--author samples_dir] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hz_common import (ASSETS, dump_json, load_encoder, load_sentence_model, model_id,  # noqa: E402
                       read_text, split_sentences, strip_markup)


# ------------------------------------------------------------ intrinsic dim

def token_embeddings(text: str, encoder: str | None = None, max_len: int = 512):
    """Last-layer contextual token vectors, one array per <=512-token chunk,
    special tokens removed (as in Tulchinskii et al.)."""
    import torch
    encoder = encoder or model_id("encoder")
    tok, model = load_encoder(encoder)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    chunks = []
    body = max_len - 2
    for s in range(0, len(ids), body):
        piece = ids[s:s + body]
        if len(piece) < 60:
            if chunks:
                break
        x = torch.tensor([[tok.cls_token_id] + piece + [tok.sep_token_id]], device=next(model.parameters()).device)
        with torch.no_grad():
            h = model(input_ids=x).last_hidden_state[0, 1:-1].float().cpu().numpy()
        chunks.append(h)
    return chunks


def _mst_length(X, alpha=1.0):
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(X))
    return float((minimum_spanning_tree(D).data ** alpha).sum())


def phd(X, alpha: float = 1.0, n_reruns: int = 3, n_points: int = 7, n_points_min: int = 3, seed: int = 0) -> float:
    """Persistent-homology dimension via MST-length scaling (Tulchinskii et al. 2023 reference
    implementation logic). E(n) ~ n^((d-alpha)/d) => d = alpha / (1 - slope)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    n_tot = X.shape[0]
    mx = n_tot
    mn = 40
    if mx - mn < 14:
        return float("nan")
    step = (mx - mn) // 7
    test_n = list(range(mn, mx - step, step))
    slopes = []
    for _ in range(n_reruns):
        lengths = []
        for n in test_n:
            restarts = n_points_min if n_tot <= 2 * n else n_points
            vals = [_mst_length(X[rng.choice(n_tot, size=n, replace=False)], alpha) for _ in range(restarts)]
            lengths.append(np.median(vals))
        x, y = np.log(test_n), np.log(lengths)
        N = len(x)
        m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
        slopes.append(m)
    m = float(np.mean(slopes))
    return alpha / (1 - m) if m < 1 else float("nan")


def mle_id(X, k: int = 20) -> float:
    """Levina-Bickel MLE with MacKay-Ghahramani averaging of inverse estimates."""
    import numpy as np
    from scipy.spatial import cKDTree
    if X.shape[0] <= k + 1:
        return float("nan")
    d, _ = cKDTree(X).query(X, k=k + 1)
    d = d[:, 1:]
    d = np.maximum(d, 1e-12)
    inv = np.log(d[:, -1:] / d[:, :-1]).sum(1) / (k - 1)
    return float(1.0 / inv.mean())


# ---------------------------------------------------------- sentence space

def sentence_embeddings(sents: list[str], model: str | None = None):
    m = load_sentence_model(model or model_id("sent_emb"))
    return m.encode(sents, normalize_embeddings=True, batch_size=32, show_progress_bar=False)


def doc_embedding(text: str, model: str | None = None):
    import numpy as np
    sents = split_sentences(strip_markup(text)) or [text]
    E = sentence_embeddings(sents, model)
    v = E.mean(0)
    return v / (np.linalg.norm(v) + 1e-12), E, sents


def trajectory_stats(E) -> dict:
    import numpy as np
    out = {}
    if E.shape[0] >= 3:
        adj = (E[:-1] * E[1:]).sum(1)
        S = E @ E.T
        iu = np.triu_indices(E.shape[0], 1)
        c = E.mean(0)
        sv = np.linalg.svd(E - c, compute_uv=False)
        p = sv ** 2 / (sv ** 2).sum()
        out = {
            "adj_cos_mean": float(adj.mean()),
            "adj_cos_sd": float(adj.std(ddof=1)),
            "pairwise_cos_mean": float(S[iu].mean()),
            "centroid_dispersion": float(np.linalg.norm(E - c, axis=1).mean()),
            "effective_rank": float(np.exp(-(p * np.log(p + 1e-12)).sum())),
            "effective_rank_per_sent": float(np.exp(-(p * np.log(p + 1e-12)).sum()) / E.shape[0]),
        }
    return out


# -------------------------------------------------------- reference banks

def has_bank(path) -> bool:
    p = Path(path)
    return (p / "reference.npz").exists() or (p / "reference.json").exists()


def load_bank(path: str | Path):
    import numpy as np
    p = Path(path)
    if (p / "reference.npz").exists():
        z = np.load(p / "reference.npz", allow_pickle=False)
    else:  # text-only installs ship the bank as JSON (same arrays)
        js = json.loads((p / "reference.json").read_text())
        z = {k: np.asarray(v, dtype=np.int64 if k == "labels" else np.float32) for k, v in js.items()}
    meta = json.loads((p / "bank_meta.json").read_text()) if (p / "bank_meta.json").exists() else {}
    return z, meta


def reference_compare(vec, bank_dir: str | Path, k: int = 15) -> dict:
    import numpy as np
    z, meta = load_bank(bank_dir)
    X, y = z["doc_emb"], z["labels"]
    sims = X @ vec
    order = np.argsort(-sims)[:k]
    out = {"bank": str(bank_dir), "knn_k": k, "knn_ai_fraction": float(y[order].mean()),
           "nearest_label": int(y[order[0]]), "nearest_sim": float(sims[order[0]])}
    if "lr_coef" in (z.files if hasattr(z, "files") else z):
        mu, sd = z["lr_mu"], z["lr_sd"]
        logit = float(((vec - mu) / sd) @ z["lr_coef"] + z["lr_intercept"][0])
        out["emb_lr_p_ai"] = float(1 / (1 + np.exp(-logit)))
    H = X[y == 0]
    if H.shape[0] > k + 1:
        # human-density percentile: distance to k-th nearest human vs. the same for human docs (leave-one-out)
        kth = lambda s: np.sort(s)[::-1][min(k, len(s)) - 1]
        d_doc = 1 - kth(H @ vec)
        HH = H @ H.T
        np.fill_diagonal(HH, -np.inf)
        d_h = 1 - np.sort(HH, axis=1)[:, ::-1][:, k - 1]
        out["human_knn_distance"] = float(d_doc)
        out["human_density_percentile"] = float((d_h >= d_doc).mean() * 100)  # 100 = denser than all humans
        if (y == 1).sum() > k + 1:
            A = X[y == 1]
            out["ai_knn_distance"] = float(1 - kth(A @ vec))
    if meta:
        out["bank_meta"] = {k_: meta[k_] for k_ in ("sent_emb", "n_human", "n_ai", "registers") if k_ in meta}
    return out


def style_match(text: str, author_texts: list[str]) -> dict:
    """Voice match: style-embedding cosine of the draft to the author's samples, located within the
    distribution of the author's own sample-to-sample similarities."""
    import numpy as np
    sm = load_sentence_model(model_id("style_emb"))

    def emb(t):
        s = split_sentences(strip_markup(t)) or [t]
        v = sm.encode(s, normalize_embeddings=True, show_progress_bar=False).mean(0)
        return v / (np.linalg.norm(v) + 1e-12)
    A = np.stack([emb(t) for t in author_texts])
    v = emb(text)
    c = A.mean(0)
    c /= np.linalg.norm(c)
    loo = []
    for i in range(len(A)):
        ci = np.delete(A, i, 0).mean(0)
        ci /= np.linalg.norm(ci)
        loo.append(float(A[i] @ ci))
    sim = float(v @ c)
    return {"style_model": model_id("style_emb"), "style_sim_to_author": sim,
            "author_loo_sim_mean": float(np.mean(loo)), "author_loo_sim_min": float(np.min(loo)),
            "style_percentile_within_author": float((np.array(loo) <= sim).mean() * 100)}


# ---------------------------------------------------------------- driver

def analyse(text: str, reference: str | None = None, author_dir: str | None = None, do_id: bool = True) -> dict:
    import numpy as np
    prose = strip_markup(text).strip()
    out: dict = {}
    if do_id:
        chunks = token_embeddings(prose)
        phds = [phd(c) for c in chunks if c.shape[0] >= 60]
        mles = [mle_id(c) for c in chunks if c.shape[0] >= 60]
        out["n_encoder_tokens"] = int(sum(c.shape[0] for c in chunks))
        out["phd"] = float(np.nanmean(phds)) if phds else float("nan")
        out["mle_id"] = float(np.nanmean(mles)) if mles else float("nan")
        out["id_reliable"] = out["n_encoder_tokens"] >= 150
    vec, E, sents = doc_embedding(prose)
    out.update(trajectory_stats(E))
    out["n_sentences"] = len(sents)
    ref = reference
    if ref is None and has_bank(ASSETS / "reference_default"):
        ref = str(ASSETS / "reference_default")
    if ref and ref != "none":
        out["reference"] = reference_compare(vec, ref)
    if author_dir:
        files = sorted(p for p in Path(author_dir).glob("*") if p.suffix in (".txt", ".md"))
        if len(files) >= 3:
            out["author_style"] = style_match(prose, [p.read_text() for p in files])
        else:
            out["author_style"] = {"error": "need >= 3 author sample files (.txt/.md)"}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--reference", help="reference bank directory (default assets/reference_default)")
    ap.add_argument("--author", help="directory of the author's own writing samples (.txt/.md)")
    ap.add_argument("--no-id", action="store_true", help="skip intrinsic-dimension estimation")
    ap.add_argument("--json")
    a = ap.parse_args()
    dump_json(analyse(read_text(a.path), a.reference, a.author, not a.no_id), a.json)


if __name__ == "__main__":
    main()
