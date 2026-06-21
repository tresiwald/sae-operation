"""
Shared helpers for the operational-fingerprint experiments (exp1-exp3).

All experiments reuse the SAEs and activation statistics produced by the main
pipeline (results/checkpoints/), so none of them retrain anything.
"""

import os, re, sys, json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # put code/ on path
from pipeline.config import (
    MODEL_NAME, FP_THRESHOLD, OPS_EVAL,
    get_device, ckpt_acts, ckpt_sae, ckpt_data, ckpt_results, OUT_DIR,
)
from pipeline.stage3_sae import TopKSAE


# ── model / checkpoint loading ────────────────────────────────────────────────
def resolve_dtype(device):
    env = os.getenv("SAE_DTYPE", "bfloat16" if device.type == "cuda" else "float32")
    return {"float16": torch.float16, "bfloat16": torch.bfloat16,
            "float32": torch.float32}[env]


def load_model():
    device = get_device()
    dtype  = resolve_dtype(device)
    print(f"  loading {MODEL_NAME}  device={device}  dtype={dtype}")
    tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype)
    model.eval().to(device)
    return model, tok, device


def load_acts_checkpoint():
    """{norm: {layer: tensor}, mu, sig, layers}"""
    return torch.load(ckpt_acts(), weights_only=True)


def load_sae(layer):
    ck      = torch.load(ckpt_sae(layer), weights_only=False)
    d_in    = ck["state_dict"]["W_enc"].shape[0]
    sae     = TopKSAE(d_in, ck["d_sae"], ck["k"])
    sae.load_state_dict(ck["state_dict"])
    sae.eval()
    return sae


def load_data():
    with open(ckpt_data(), "rb") as f:
        return pickle_load(f)


def pickle_load(f):
    import pickle
    return pickle.load(f)


def best_layer_from_results(default=None):
    """Read best_layer from results.json, falling back to `default`."""
    p = ckpt_results()
    if p.exists():
        return json.loads(p.read_text()).get("best_layer", default)
    return default


# ── activation collection (single-record, NaN-safe) ──────────────────────────
@torch.no_grad()
def collect_last_token(prompts, model, tok, layers, device, desc="acts"):
    """{layer: float32 CPU tensor [n, d_model]} of last-token residual inputs."""
    bufs = {l: [] for l in layers}

    def mk(li):
        def _h(mod, args):
            h = args[0] if isinstance(args[0], torch.Tensor) else args[0][0]
            bufs[li].append(h[:, -1, :].detach().float().cpu())
        return _h

    handles = [model.model.layers[l].register_forward_pre_hook(mk(l)) for l in layers]
    for p in tqdm(prompts, desc=desc, leave=False):
        inp = tok(p, return_tensors="pt").to(device)
        model(**inp)
    for h in handles:
        h.remove()
    return {l: torch.cat(bufs[l], 0) for l in layers}


def normalize(raw, mu, sig):
    raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return (raw - mu.to(raw)) / sig.to(raw)


@torch.no_grad()
def encode(sae, acts):
    """SAE-encode in chunks → numpy [n, d_sae]."""
    return np.concatenate(
        [sae.encode(acts[i:i + 512]).numpy() for i in range(0, len(acts), 512)],
        axis=0,
    )


# ── fingerprints ──────────────────────────────────────────────────────────────
def cohens_d_fingerprint(feats_op, feats_ctrl):
    """Cohen's d per feature; fingerprint = features with d > FP_THRESHOLD."""
    bm, bs = feats_ctrl.mean(0), feats_ctrl.std(0) + 1e-8
    d = (feats_op.mean(0) - bm) / np.sqrt((bs ** 2 + feats_op.std(0) ** 2 + 1e-8) / 2)
    return d, d > FP_THRESHOLD


def build_training_fingerprints(layer, sae, acts_ck, train_recs):
    """
    Reconstruct the per-op compute fingerprints from the training activations,
    exactly as stage4 does (row order of norm[layer] aligns with train_recs).

    Returns: fps {op: bool mask}, centroids {op: mean feature vec}, feats matrix.
    """
    norm  = acts_ck["norm"][layer]
    feats = np.nan_to_num(encode(sae, norm), nan=0.0)

    def slice_op(op):
        mask = np.array([r["op"] == op for r in train_recs])
        return feats[mask]

    ctrl = slice_op("ctrl")
    fps, centroids = {}, {}
    for op in OPS_EVAL:
        fo = slice_op(op)
        if len(fo) < 5 or len(ctrl) < 5:
            continue
        _, m = cohens_d_fingerprint(fo, ctrl)
        fps[op]       = m
        centroids[op] = fo.mean(0)
    return fps, centroids, feats


def fingerprint_strength(feats, fp_mask):
    """Per-sample mean activation over the fingerprint feature set → array[n]."""
    if fp_mask.sum() == 0:
        return np.zeros(len(feats))
    return feats[:, fp_mask].mean(axis=1)


def cosine_to_centroid(feats, centroid):
    """Per-sample cosine similarity to a centroid → array[n]."""
    c  = centroid / (np.linalg.norm(centroid) + 1e-8)
    fn = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    return fn @ c


# ── answer extraction (shared with stage5 logic) ──────────────────────────────
@torch.no_grad()
def generate_answers(prompts, model, tok, device, max_new_tokens=8):
    """Greedy-decode and return the first integer in each completion (or None)."""
    answers = []
    for p in tqdm(prompts, desc="generate", leave=False):
        inp = tok(p, return_tensors="pt").to(device)
        out = model.generate(inp["input_ids"], max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
        dec = tok.decode(out[0, inp["input_ids"].shape[1]:]).strip()
        m   = re.match(r"\d+", dec)
        answers.append(int(m.group()) if m else None)
    return answers
