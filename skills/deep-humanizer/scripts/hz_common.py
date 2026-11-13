"""Shared helpers for the deep-humanizer toolkit.

Everything model-related is lazy: nothing heavy is imported until a function
needs it, and every loaded model is cached for the life of the process.

Model defaults can be overridden with environment variables so the same
scripts run on a laptop CPU or a GPU box:

    HZ_LM_OBSERVER    base causal LM (surprisal, Fast-DetectGPT, Binoculars observer)
    HZ_LM_PERFORMER   instruction-tuned sibling sharing the tokenizer (Binoculars performer)
    HZ_ENCODER        masked-LM encoder for token-embedding geometry (PHD / MLE intrinsic dim)
    HZ_SENT_EMB       sentence-embedding model (semantic trajectory, reference distance, meaning guard)
    HZ_STYLE_EMB      content-independent style embedding (voice match against author samples)
    HZ_DEVICE         cpu | cuda | mps   (default: best available)
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
ASSETS = SKILL_DIR / "assets"

DEFAULT_MODELS = {
    "observer": "Qwen/Qwen2.5-1.5B",
    "performer": "Qwen/Qwen2.5-1.5B-Instruct",
    "encoder": "FacebookAI/roberta-base",
    "sent_emb": "sentence-transformers/all-mpnet-base-v2",
    "style_emb": "AnnaWegmann/Style-Embedding",
}
ENV_KEYS = {
    "observer": "HZ_LM_OBSERVER",
    "performer": "HZ_LM_PERFORMER",
    "encoder": "HZ_ENCODER",
    "sent_emb": "HZ_SENT_EMB",
    "style_emb": "HZ_STYLE_EMB",
}

_CACHE: dict = {}


def model_id(role: str) -> str:
    return os.environ.get(ENV_KEYS[role], DEFAULT_MODELS[role])


# --------------------------------------------------------------------------- I/O

def read_text(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def strip_markup(text: str) -> str:
    """Remove code blocks, inline code, URLs and markdown decoration so the
    signals measure prose only."""
    t = re.sub(r"```.*?```", " ", text, flags=re.S)
    t = re.sub(r"`[^`]*`", " ", t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)      # headings
    t = re.sub(r"^\s*[-*+]\s+", "", t, flags=re.M)            # bullets
    t = re.sub(r"^\s*\d+[.)]\s+", "", t, flags=re.M)          # numbered items
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)                    # bold
    t = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"\1", t)  # italics
    return t


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def to_jsonable(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def dump_json(obj, path: str | None):
    s = json.dumps(to_jsonable(obj), indent=2, ensure_ascii=False)
    if path:
        Path(path).write_text(s, encoding="utf-8")
    else:
        print(s)


# ------------------------------------------------------------------- tokenizing

_WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*|\d+(?:[.,]\d+)*")


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def get_nlp():
    """spaCy English pipeline (en_core_web_sm). Returns None if unavailable."""
    if "nlp" in _CACHE:
        return _CACHE["nlp"]
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        nlp.max_length = 2_000_000
    except Exception as exc:  # pragma: no cover
        print(f"[hz] spaCy model unavailable ({exc}); falling back to regex splitting", file=sys.stderr)
        nlp = None
    _CACHE["nlp"] = nlp
    return nlp


def split_sentences(text: str) -> list[str]:
    nlp = get_nlp()
    out: list[str] = []
    for para in paragraphs(text):
        if nlp is not None:
            out.extend(s.text.strip() for s in nlp(para).sents if s.text.strip())
        else:
            out.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", para) if s.strip())
    return out


# ----------------------------------------------------------------------- models

def device():
    import torch
    want = os.environ.get("HZ_DEVICE")
    if want:
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_causal_lm(name: str):
    key = ("clm", name)
    if key not in _CACHE:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        dev = device()
        dtype = torch.float32 if dev.type == "cpu" else torch.float16
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        model.to(dev).eval()
        _CACHE[key] = (tok, model)
    return _CACHE[key]


def load_encoder(name: str):
    key = ("enc", name)
    if key not in _CACHE:
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name)
        model.to(device()).eval()
        _CACHE[key] = (tok, model)
    return _CACHE[key]


def load_sentence_model(name: str):
    key = ("st", name)
    if key not in _CACHE:
        from sentence_transformers import SentenceTransformer
        _CACHE[key] = SentenceTransformer(name, device=str(device()))
    return _CACHE[key]


# ------------------------------------------------------------------ statistics

def safe_cv(values) -> float:
    import numpy as np
    v = np.asarray(values, dtype=float)
    if v.size < 2 or v.mean() == 0:
        return float("nan")
    return float(v.std(ddof=1) / v.mean())


def burstiness(values) -> float:
    """Goh & Barabasi burstiness B = (sd - mean) / (sd + mean); -1 periodic, 0 Poisson, >0 bursty."""
    import numpy as np
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return float("nan")
    m, s = v.mean(), v.std(ddof=1)
    return float((s - m) / (s + m)) if (s + m) > 0 else float("nan")


def word_count(text: str) -> int:
    return len(words(text))
