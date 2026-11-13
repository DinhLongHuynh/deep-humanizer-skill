#!/usr/bin/env python
"""Layer 3 - language-model probability signals.

LLM decoding favours high-probability continuations, so machine text sits in
a region where (a) each token is likelier than a token *sampled* from the
model would be (Fast-DetectGPT's conditional probability curvature), (b) the
text is unsurprising relative to how surprising the model expects it to be
(Binoculars: perplexity / cross-perplexity), and (c) surprisal is spread
evenly across tokens and sentences (uniform information density; GLTR
top-k rank histogram). Human writers produce more low-probability but apt
choices, and more uneven information flow.

Signals returned (observer = base LM, performer = its instruct sibling):
    fastdetectgpt_raw          analytic sampling discrepancy, summed (Bao et al. 2024 official form)
    fastdetectgpt_norm         same divided by sqrt(n_tokens): length-invariant, use for thresholds
    binoculars                 log-PPL_performer / X-PPL(observer->performer)   (Hans et al. 2024 code form)
    logppl_observer            mean token surprisal (nats)
    surprisal_sd / _cv         token-level spread
    uid_local                  mean squared difference of consecutive surprisals
    sent_surprisal_sd          spread of per-sentence mean surprisal (information burstiness)
    gltr_top10 / top100 / ...  fraction of tokens by observer rank
    mean_log_rank, mean_entropy

Higher fastdetectgpt_* and gltr_top10, lower binoculars / logppl / spreads
=> more machine-like. Absolute values depend on the scoring models, so
compare against assets/thresholds.json (calibrated for the default Qwen2.5
pair) or recalibrate with calibrate.py when you change models.

It also returns per-sentence scores and the most predictable token runs,
which become the rewrite targets ("hotspots") in the next iteration.

Usage:
    python lm_signals.py draft.md [--json out.json] [--no-performer] [--max-chunk 512]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hz_common import dump_json, load_causal_lm, model_id, read_text, split_sentences, strip_markup  # noqa: E402


def _token_stats(text: str, observer: str, performer: str | None, max_chunk: int):
    import torch
    tok, obs = load_causal_lm(observer)
    perf = load_causal_lm(performer)[1] if performer else None
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    offs = enc["offset_mapping"]
    rows = []  # per predicted token
    dev = next(obs.parameters()).device
    for start in range(0, len(ids), max_chunk):
        chunk = ids[start:start + max_chunk]
        if len(chunk) < 2:
            continue
        x = torch.tensor([chunk], device=dev)
        with torch.no_grad():
            lo = obs(x).logits[0, :-1].float()
            lp_o = torch.log_softmax(lo, -1)
            labels = x[0, 1:]
            p_o = lp_o.exp()
            ll = lp_o.gather(-1, labels[:, None])[:, 0]
            mean_ref = (p_o * lp_o).sum(-1)
            var_ref = (p_o * lp_o.square()).sum(-1) - mean_ref.square()
            ent = -mean_ref
            rank = (lp_o > ll[:, None]).sum(-1) + 1
            if perf is not None:
                lpf = torch.log_softmax(perf(x).logits[0, :-1].float(), -1)
                ce_perf = -lpf.gather(-1, labels[:, None])[:, 0]          # performer surprisal of actual token
                xent = -(p_o * lpf).sum(-1)                                # cross-entropy observer->performer
            else:
                ce_perf = xent = torch.full_like(ll, float("nan"))
        for j in range(len(chunk) - 1):
            gi = start + j + 1
            rows.append({
                "i": gi, "char": offs[gi], "ll": ll[j].item(), "mean_ref": mean_ref[j].item(),
                "var_ref": var_ref[j].item(), "entropy": ent[j].item(), "rank": int(rank[j].item()),
                "ce_perf": ce_perf[j].item(), "xent": xent[j].item(),
            })
    return tok, ids, rows


def surprisal_dynamics(s) -> dict:
    """Shape of the surprisal series, not just its level.
    DivEye (Basani & Chen, TMLR 2026): moments of S, of its first and second difference.
    GPT-who (Venkatraman et al., NAACL Findings 2024): UID span extremes over 20-token windows.
    Late-stage volatility (arXiv 2601.04833): dispersion in the second half of the text.
    Human text: wider spread, heavier right tail, more volatile; AI text: smoother."""
    import numpy as np
    from scipy.stats import kurtosis, skew
    d1 = np.diff(s)
    d2 = np.diff(s, 2)
    out = {
        "surprisal_skew": float(skew(s)),
        "surprisal_kurt": float(kurtosis(s)),
        "d1_var": float(d1.var()),
        "d2_var": float(d2.var()),
        "d2_lag1_autocorr": float(np.corrcoef(d2[:-1], d2[1:])[0, 1]) if len(d2) > 3 else float("nan"),
    }
    N = 20
    if len(s) >= 2 * N:
        spans = np.convolve(s, np.ones(N) / N, mode="valid")
        out["uid_span_max"] = float(spans.max())
        out["uid_span_min"] = float(spans.min())
        out["uid_span_range"] = float(spans.max() - spans.min())
    half = s[len(s) // 2:]
    if len(half) > N + 2:
        out["late_deriv_dispersion"] = float(np.abs(np.diff(half)).std())
        win = np.lib.stride_tricks.sliding_window_view(half, N)
        out["late_local_volatility"] = float(win.std(axis=1).mean())
    return out


def score(text: str, observer: str | None = None, performer: str | None = "default", max_chunk: int = 512,
          n_hotspots: int = 5) -> dict:
    import numpy as np
    observer = observer or model_id("observer")
    if performer == "default":
        performer = model_id("performer")
    prose = strip_markup(text).strip()
    tok, ids, rows = _token_stats(prose, observer, performer, max_chunk)
    if len(rows) < 20:
        return {"error": "text too short for LM signals (need >= ~20 tokens; reliable above ~150)", "n_tokens": len(rows)}
    ll = np.array([r["ll"] for r in rows])
    mr = np.array([r["mean_ref"] for r in rows])
    vr = np.array([r["var_ref"] for r in rows])
    ent = np.array([r["entropy"] for r in rows])
    rk = np.array([r["rank"] for r in rows])
    surpr = -ll
    n = len(rows)
    fdg_raw = float((ll.sum() - mr.sum()) / np.sqrt(vr.sum()))
    out = {
        "observer": observer,
        "performer": performer,
        "n_tokens": n,
        "fastdetectgpt_raw": fdg_raw,
        "fastdetectgpt_norm": fdg_raw / np.sqrt(n),
        "logppl_observer": float(surpr.mean()),
        "surprisal_sd": float(surpr.std(ddof=1)),
        "surprisal_cv": float(surpr.std(ddof=1) / surpr.mean()),
        "surprisal_p95": float(np.percentile(surpr, 95)),
        "uid_local": float(np.mean(np.diff(surpr) ** 2)),
        "mean_entropy": float(ent.mean()),
        "mean_log_rank": float(np.log(rk).mean()),
        "gltr_top1": float((rk <= 1).mean()),
        "gltr_top10": float((rk <= 10).mean()),
        "gltr_top100": float(((rk > 10) & (rk <= 100)).mean()),
        "gltr_top1000": float(((rk > 100) & (rk <= 1000)).mean()),
        "gltr_beyond": float((rk > 1000).mean()),
    }
    out.update(surprisal_dynamics(surpr))
    if performer:
        cep = np.array([r["ce_perf"] for r in rows])
        xen = np.array([r["xent"] for r in rows])
        out["logppl_performer"] = float(cep.mean())
        out["xppl"] = float(xen.mean())
        out["binoculars"] = float(cep.mean() / xen.mean())

    # ---- per-sentence scores, mapped by character offsets
    sents = split_sentences(prose)
    spans, cursor = [], 0
    for s in sents:
        k = prose.find(s[:30], cursor)
        k = cursor if k < 0 else k
        spans.append((k, k + len(s)))
        cursor = k + len(s)
    per_sent = []
    for si, (a, b) in enumerate(spans):
        idx = [j for j, r in enumerate(rows) if a <= r["char"][0] < b]
        if len(idx) < 4:
            continue
        sl, sm, sv = ll[idx], mr[idx], vr[idx]
        per_sent.append({
            "sent_index": si,
            "text": sents[si],
            "n_tokens": len(idx),
            "mean_surprisal": float(-sl.mean()),
            "fdg_local": float((sl.sum() - sm.sum()) / np.sqrt(sv.sum()) / np.sqrt(len(idx))),
            "top10_rate": float((rk[idx] <= 10).mean()),
        })
    if per_sent:
        ms = np.array([p["mean_surprisal"] for p in per_sent])
        out["sent_surprisal_sd"] = float(ms.std(ddof=1)) if len(ms) > 1 else float("nan")
        out["sent_surprisal_range"] = float(ms.max() - ms.min())
        out["sentences"] = per_sent
        out["hotspots"] = sorted(per_sent, key=lambda p: -p["fdg_local"])[:n_hotspots]

    # ---- longest runs of rank-1 tokens: maximally predictable phrasing
    runs, cur = [], []
    for j, r in enumerate(rows):
        if r["rank"] == 1:
            cur.append(j)
        else:
            if len(cur) >= 5:
                runs.append(cur)
            cur = []
    if len(cur) >= 5:
        runs.append(cur)
    runs = sorted(runs, key=len, reverse=True)[:8]
    out["predictable_runs"] = [
        prose[rows[r[0]]["char"][0]: rows[r[-1]]["char"][1]].strip() for r in runs
    ]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--json")
    ap.add_argument("--observer", default=None)
    ap.add_argument("--performer", default="default")
    ap.add_argument("--no-performer", action="store_true", help="skip Binoculars (halves runtime)")
    ap.add_argument("--max-chunk", type=int, default=512)
    a = ap.parse_args()
    perf = None if a.no_performer else a.performer
    dump_json(score(read_text(a.path), a.observer, perf, a.max_chunk), a.json)


if __name__ == "__main__":
    main()
