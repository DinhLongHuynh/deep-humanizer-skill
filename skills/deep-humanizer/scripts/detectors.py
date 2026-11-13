#!/usr/bin/env python
"""Held-out detectors: open-source classifiers (local) and commercial APIs.

These are the *confirmation* signals of the check round. Do not steer the
rewrite with the detector you use to confirm it (Goodhart): rewrite against
the local diagnostic layers (check.py), confirm with these, and report them.

Open classifiers run locally (CPU is fine). Label directions differ between
models, so each entry below states its mapping explicitly (from the model
cards, see references/detectors.md). Text is NFKC-normalised and stripped of
zero-width characters before scoring, as the model cards recommend.

Commercial APIs are called only for providers whose key is in the
environment. Every call sends the text to that company: confirm the user
agrees before using --external. Adapters request no-storage options where
the API offers them, and record the provider's model version so before/after
scores come from the same model (vendors update models without notice).
The adapters follow the official API documentation as of November 2026 but
were not exercised with live keys; on a first run, inspect the "raw" field.

Usage:
    python detectors.py draft.md                          # default open set
    python detectors.py draft.md --open all               # + humanized-class probe and legacy detectors
    python detectors.py draft.md --external gptzero,sapling
    python detectors.py --list
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hz_common import ASSETS, _CACHE, device, dump_json, read_text, strip_markup  # noqa: E402

# head: "seqcls" = AutoModelForSequenceClassification; "mean_pool" = backbone + mean pooling + linear head
OPEN_DETECTORS = {
    "tabularisai": {"hf_id": "tabularisai/ai-text-detection", "head": "seqcls", "ai_index": 1, "max_tokens": 512,
                    "note": "ModernBERT-base 150M, MIT; trained on RAID train + 6 public sets (RAID 0.990, in-distribution)"},
    "desklib": {"hf_id": "desklib/ai-text-detector-v1.01", "head": "mean_pool", "n_out": 1, "max_tokens": 768,
                "note": "DeBERTa-v3-large 434M, MIT; sigmoid(logit) = P(AI); RAID 0.912"},
    "fakespot": {"hf_id": "fakespot-ai/roberta-base-ai-text-detection-v1", "head": "seqcls", "ai_index": 1,
                 "max_tokens": 512, "note": "RoBERTa-base 125M, Apache-2.0; not RAID-trained (out-of-distribution check)"},
    "tropa_mini": {"hf_id": "wasitaigeneratedcom/tropa-mini-ai-text-detector", "head": "mean_pool", "n_out": 4,
                   "labels": ["human", "ai", "ai_edited", "humanized"], "max_tokens": 768,
                   "note": "DeBERTa-v3-large 4-class; P(AI involvement) = 1 - p[human]; reports p[humanized] "
                           "separately (vendor-reported accuracy only)"},
    "openai_gpt2": {"hf_id": "openai-community/roberta-base-openai-detector", "head": "seqcls", "ai_index": 0,
                    "max_tokens": 512, "note": "2019 GPT-2 output detector; label 0 = Fake; near chance on modern LLMs"},
    "hc3_chatgpt": {"hf_id": "Hello-SimpleAI/chatgpt-detector-roberta", "head": "seqcls", "ai_index": 1,
                    "max_tokens": 512, "note": "HC3 2023 ChatGPT detector; ships .bin weights only"},
}
DEFAULT_OPEN = ["tabularisai", "desklib", "fakespot", "tropa_mini"]  # tropa-mini in every round: it was the one
# open detector that agreed with ZeroGPT when the other three scored a text at ~0 (Nov 2026)
FINAL_OPEN = ["tabularisai", "desklib", "fakespot", "tropa_mini"]

INVISIBLE = re.compile("[\u200b\u200c\u200d\u2060\ufeff\u00ad]")


def normalise(text: str) -> str:
    return INVISIBLE.sub("", unicodedata.normalize("NFKC", strip_markup(text))).strip()


# ------------------------------------------------------------------ open models

def _load_seqcls(hf_id: str):
    key = ("clf", hf_id)
    if key not in _CACHE:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(hf_id)
        mdl = AutoModelForSequenceClassification.from_pretrained(hf_id)
        mdl.to(device()).eval()
        _CACHE[key] = (tok, mdl)
    return _CACHE[key]


def _load_mean_pool(hf_id: str, n_out: int):
    """Custom heads (desklib, tropa-mini): mean-pooled encoder + linear layer. Rebuilt from the
    safetensors weights so no remote code has to be executed."""
    key = ("mp", hf_id)
    if key not in _CACHE:
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        cfg = AutoConfig.from_pretrained(hf_id)
        tok = AutoTokenizer.from_pretrained(hf_id)
        backbone = AutoModel.from_config(cfg)
        sd = load_file(hf_hub_download(hf_id, "model.safetensors"))
        bb = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        missing, _ = backbone.load_state_dict(bb, strict=False)
        if len(missing) > 5:
            raise RuntimeError(f"{hf_id}: backbone weights did not map cleanly ({len(missing)} missing)")
        w = sd["classifier.weight"]
        if w.shape[0] != n_out:
            raise RuntimeError(f"{hf_id}: expected {n_out} outputs, checkpoint has {w.shape[0]}")
        head = torch.nn.Linear(cfg.hidden_size, n_out)
        head.load_state_dict({"weight": w, "bias": sd["classifier.bias"]})
        dev = device()
        backbone.to(dev).eval()
        head.to(dev).eval()
        _CACHE[key] = (tok, backbone, head)
    return _CACHE[key]


def _chunks(tok, text: str, max_tokens: int):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    step = max_tokens - 2
    return [tok.decode(ids[i:i + step]) for i in range(0, max(1, len(ids)), step)] or [text]


def _score_one(spec: dict, text: str) -> dict:
    import numpy as np
    import torch
    rows, weights = [], []
    if spec["head"] == "seqcls":
        tok, mdl = _load_seqcls(spec["hf_id"])
        for ch in _chunks(tok, text, spec["max_tokens"]):
            enc = tok(ch, truncation=True, max_length=spec["max_tokens"], return_tensors="pt").to(mdl.device)
            with torch.no_grad():
                rows.append(torch.softmax(mdl(**enc).logits[0].float(), -1).cpu().numpy())
            weights.append(int(enc["attention_mask"].sum()))
        P = np.average(np.stack(rows), axis=0, weights=weights)
        return {"p_ai": float(P[spec["ai_index"]]), "chunk_p_ai": [float(r[spec["ai_index"]]) for r in rows]}
    tok, bb, head = _load_mean_pool(spec["hf_id"], spec["n_out"])
    dev = next(bb.parameters()).device
    for ch in _chunks(tok, text, spec["max_tokens"]):
        enc = tok(ch, truncation=True, max_length=spec["max_tokens"], return_tensors="pt").to(dev)
        with torch.no_grad():
            h = bb(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            logits = head((h * m).sum(1) / m.sum(1).clamp(min=1e-9))[0].float()
            p = torch.sigmoid(logits) if spec["n_out"] == 1 else torch.softmax(logits, -1)
        rows.append(p.cpu().numpy())
        weights.append(int(enc["attention_mask"].sum()))
    P = np.average(np.stack(rows), axis=0, weights=weights)
    if spec["n_out"] == 1:
        return {"p_ai": float(P[0]), "chunk_p_ai": [float(r[0]) for r in rows]}
    probs = {lab: float(v) for lab, v in zip(spec["labels"], P)}
    return {"p_ai": 1.0 - probs["human"], "class_probs": probs, "p_humanized": probs.get("humanized")}


def run_open(text: str, names=None) -> dict:
    names = names or DEFAULT_OPEN
    text = normalise(text)
    out = {}
    for name in names:
        spec = OPEN_DETECTORS[name]
        try:
            out[name] = {"hf_id": spec["hf_id"], **_score_one(spec, text)}
        except Exception as exc:
            out[name] = {"hf_id": spec["hf_id"], "error": repr(exc)[:300]}
    return out


# ------------------------------------------------------------- commercial APIs

def _key(var: str) -> str:
    v = os.environ.get(var)
    if not v:
        raise KeyError(f"set {var}")
    return v


def _post(url, headers=None, body=None, timeout=90):
    import requests
    r = requests.post(url, headers={"Content-Type": "application/json", **(headers or {})}, json=body, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def _gptzero(text):
    body = {"document": text}
    if os.environ.get("GPTZERO_MODEL_VERSION"):
        body["modelVersion"] = os.environ["GPTZERO_MODEL_VERSION"]
    js = _post("https://api.gptzero.me/v2/predict/text", {"x-api-key": _key("GPTZERO_API_KEY")}, body)
    d = js["documents"][0]
    cp = d.get("class_probabilities", {})
    sub = (d.get("subclass") or {}).get("ai", {}).get("class_probabilities")
    return {"p_ai": cp.get("ai"), "p_ai_or_mixed": (cp.get("ai") or 0) + (cp.get("mixed") or 0),
            "predicted_class": d.get("predicted_class"), "confidence": d.get("confidence_category"),
            "ai_subclass": sub, "model_version": js.get("version"),
            "segments": [s.get("generated_prob") for s in d.get("sentences", [])]}


def _pangram(text):
    base = "https://text.external-api.pangram.com"
    h = {"x-api-key": _key("PANGRAM_API_KEY")}
    task = _post(base + "/task", h, {"text": text, "model": os.environ.get("PANGRAM_MODEL", "pangram-4"),
                                     "public_dashboard_link": False})
    import requests
    tid = task["task_id"]
    for _ in range(60):
        time.sleep(2)
        js = requests.get(f"{base}/task/{tid}", headers=h, timeout=60).json()
        if js.get("stage") == "STAGE_SUCCESS":
            win = js.get("windows", [])
            return {"p_ai": js.get("fraction_ai"), "fraction_ai_assisted": js.get("fraction_ai_assisted"),
                    "prediction": js.get("prediction_short") or js.get("prediction"), "model_version": js.get("version"),
                    "humanized_windows": sum(1 for w in win if w.get("is_humanized")),
                    "segments": [w.get("ai_assistance_score") for w in win]}
        if js.get("stage") == "STAGE_FAILED":
            raise RuntimeError(f"Pangram task failed: {json.dumps(js)[:200]}")
    raise TimeoutError("Pangram task did not finish in 120 s")


def _originality(text):
    js = _post("https://api.originality.ai/api/v3/scan", {"X-OAI-API-KEY": _key("ORIGINALITY_API_KEY")},
               {"title": "check", "content": text, "check_ai": True, "check_plagiarism": False, "check_facts": False,
                "check_readability": False, "check_grammar": False, "storeScan": False,
                "aiModelVersion": os.environ.get("ORIGINALITY_MODEL", "model-5.0")})
    ai = js.get("results", {}).get("ai", {})
    sents = [s.get("ai_score") for s in ai.get("sentences", []) if s.get("ai_score") is not None]
    return {"p_ai": sum(sents) / len(sents) if sents else None, "p_ai_is_proxy": "mean sentence ai_score",
            "prediction_label": ai.get("prediction_label"), "ai_assisted_percent": ai.get("ai_assisted_percent"),
            "segments": sents, "warning": "Originality auto-tops-up credits by default; disable before loops"}


def _sapling(text):
    js = _post("https://api.sapling.ai/api/v1/aidetect", {"Authorization": f"Bearer {_key('SAPLING_API_KEY')}"},
               {"text": text, "sent_scores": True, "zero_retention": True,
                "version": os.environ.get("SAPLING_MODEL_VERSION", "20261104")})
    return {"p_ai": js.get("score"), "ai_fraction": js.get("ai_fraction"), "low_confidence": js.get("low_confidence"),
            "segments": [s.get("score") for s in js.get("sentence_scores", [])]}


def _winston(text):
    js = _post("https://api.gowinston.ai/v2/ai-content-detection", {"Authorization": f"Bearer {_key('WINSTON_API_KEY')}"},
               {"text": text, "version": os.environ.get("WINSTON_MODEL_VERSION", "5.2"), "sentences": True,
                "language": "auto"})
    s = js.get("score")
    return {"p_ai": None if s is None else 1 - float(s) / 100, "human_score_0_100": s,
            "attack_detected": js.get("attack_detected"), "model_version": js.get("version")}


def _find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def _hive(text):
    js = _post("https://api.thehive.ai/api/v2/task/sync", {"Authorization": f"token {_key('HIVE_API_KEY')}"},
               {"text_data": text})
    agg = _find_key(js, "aggregate_score")
    p = None
    if isinstance(agg, list):
        p = next((c.get("score") for c in agg if c.get("class") == "ai_generated"), None)
    elif isinstance(agg, (int, float)):
        p = float(agg)
    return {"p_ai": p, "raw_aggregate": agg}


def _copyleaks(text):
    cache = Path.home() / ".cache" / "deep_humanizer" / "copyleaks_token.json"
    tok = None
    if cache.exists():
        c = json.loads(cache.read_text())
        if c.get("t", 0) > time.time() - 40 * 3600:
            tok = c["token"]
    if tok is None:  # the login endpoint blocks for 5 min on 429, so the token is cached
        js = _post("https://id.copyleaks.com/v3/account/login/api", body={"email": _key("COPYLEAKS_EMAIL"),
                                                                          "key": _key("COPYLEAKS_API_KEY")})
        tok = js["access_token"]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"token": tok, "t": time.time()}))
        os.chmod(cache, 0o600)
    js = _post(f"https://api.copyleaks.com/v2/writer-detector/{uuid.uuid4().hex[:32]}/check",
               {"Authorization": f"Bearer {tok}"}, {"text": text, "sandbox": False, "explain": False, "sensitivity": 2})
    return {"p_ai": js.get("summary", {}).get("ai"), "model_version": js.get("modelVersion"),
            "note": "share of text classified AI, not a probability"}


def _zerogpt(text):
    js = _post("https://api.zerogpt.com/api/detect/detectText", {"ApiKey": _key("ZEROGPT_API_KEY")},
               {"input_text": text})
    h = js.get("data", {}).get("isHuman")
    return {"p_ai": None if h is None else 1 - float(h) / 100,
            "warning": "endpoint taken from the RAID harness; official docs unverified"}


EXTERNAL = {  # name -> (adapter, env vars, minimum characters)
    "gptzero": (_gptzero, ["GPTZERO_API_KEY"], 0),
    "pangram": (_pangram, ["PANGRAM_API_KEY"], 0),
    "originality": (_originality, ["ORIGINALITY_API_KEY"], 0),
    "sapling": (_sapling, ["SAPLING_API_KEY"], 300),
    "winston": (_winston, ["WINSTON_API_KEY"], 600),
    "hive": (_hive, ["HIVE_API_KEY"], 0),
    "copyleaks": (_copyleaks, ["COPYLEAKS_EMAIL", "COPYLEAKS_API_KEY"], 255),
    "zerogpt": (_zerogpt, ["ZEROGPT_API_KEY"], 0),
}


def available_external() -> list[str]:
    return [n for n, (_, envs, _) in EXTERNAL.items() if all(os.environ.get(e) for e in envs)]


def run_external(text: str, names) -> dict:
    text = normalise(text)
    out = {}
    for name in names:
        name = name.strip().lower()
        if name not in EXTERNAL:
            out[name] = {"error": f"unknown provider; choose from {sorted(EXTERNAL)}"}
            continue
        fn, envs, min_chars = EXTERNAL[name]
        if not all(os.environ.get(e) for e in envs):
            out[name] = {"skipped": f"no key: set {', '.join(envs)}"}
            continue
        try:
            res = fn(text)
            if len(text) < min_chars:
                res["warning_length"] = f"text shorter than the provider's {min_chars}-character minimum"
            out[name] = res
        except Exception as exc:
            out[name] = {"error": repr(exc)[:300]}
    return out


def load_specs() -> list[dict]:
    p = ASSETS / "detector_api_specs.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    return d.get("specs", d) if isinstance(d, dict) else d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?")
    ap.add_argument("--open", default=",".join(DEFAULT_OPEN), help="comma list, 'all', 'final' or 'none'")
    ap.add_argument("--external", default="", help="comma list of commercial providers (needs keys; sends text out)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.list:
        for n, s in OPEN_DETECTORS.items():
            print(f"open  {n:12s} {s['hf_id']:50s} {s['note']}")
        for n, (_, envs, _) in EXTERNAL.items():
            print(f"api   {n:12s} env={','.join(envs):35s} {'KEY SET' if n in available_external() else ''}")
        print("not wrappable (no public API): Turnitin, QuillBot, Scribbr; Grammarly needs an OAuth upload flow")
        return
    text = read_text(a.path)
    names = {"all": list(OPEN_DETECTORS), "final": FINAL_OPEN, "none": []}.get(a.open, a.open.split(","))
    res = {"open": run_open(text, names) if names else {}}
    if a.external:
        res["external"] = run_external(text, a.external.split(","))
    dump_json(res, a.json)


if __name__ == "__main__":
    main()
