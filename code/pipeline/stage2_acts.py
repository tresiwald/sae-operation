"""
Stage 2 — Collect residual-stream activations at all sweep layers.

Reads  : results/data.pkl
Outputs: results/acts_checkpoint.pt
           keys: norm[layer], mu[layer], sig[layer], layers

Device strategy
  MPS  → batch_size=1  (padding triggers NaN in MPS attention)
  CUDA → batched with left-padding (SAE_ACT_BATCH, default 32)
  CPU  → batch_size=1
"""

import os, pickle, sys
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import (
    MODEL_NAME, LAYERS, get_device, ckpt_data, ckpt_acts
)

ACT_BATCH = int(os.getenv("SAE_ACT_BATCH", "32"))   # used on CUDA only


def resolve_layers(model, layers):
    n = model.config.num_hidden_layers
    if layers is None:
        return list(range(n))
    bad = [l for l in layers if not (0 <= l < n)]
    if bad:
        raise ValueError(f"Layers {bad} out of range 0..{n-1}")
    return layers


# ── single-record collection (MPS / CPU) ─────────────────────────────────────
def collect_single(records, model, tokenizer, layers, device):
    bufs: dict[int, list[torch.Tensor]] = {l: [] for l in layers}

    def make_hook(li):
        def _h(module, args):
            h = args[0] if isinstance(args[0], torch.Tensor) else args[0][0]
            bufs[li].append(h[:, -1, :].detach().float().cpu())
        return _h

    handles = [model.model.layers[l].register_forward_pre_hook(make_hook(l))
               for l in layers]
    with torch.no_grad():
        for i, rec in enumerate(tqdm(records, desc="acts")):
            inp = tokenizer(rec["prompt"], return_tensors="pt").to(device)
            model(**inp)
            if device.type == "mps" and i % 500 == 499:
                torch.mps.empty_cache()
    for h in handles:
        h.remove()
    return {l: torch.cat(bufs[l], dim=0) for l in layers}


# ── batched collection (CUDA, left-padded) ────────────────────────────────────
def collect_batched(records, model, tokenizer, layers, device, batch_size):
    """
    Left-pad each batch so all sequences end at the same position.
    The hook captures [:, -1, :] which is always the last real token.
    """
    bufs: dict[int, list[torch.Tensor]] = {l: [] for l in layers}

    def make_hook(li):
        def _h(module, args):
            h = args[0] if isinstance(args[0], torch.Tensor) else args[0][0]
            bufs[li].append(h[:, -1, :].detach().float().cpu())
        return _h

    handles = [model.model.layers[l].register_forward_pre_hook(make_hook(l))
               for l in layers]

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [r["prompt"] for r in records]
    with torch.no_grad():
        for start in tqdm(range(0, len(prompts), batch_size), desc="acts (batched)"):
            batch = prompts[start:start + batch_size]
            enc   = tokenizer(batch, return_tensors="pt", padding=True,
                              truncation=True, max_length=512).to(device)
            model(**enc)
    for h in handles:
        h.remove()
    return {l: torch.cat(bufs[l], dim=0) for l in layers}


def main():
    out = ckpt_acts()
    if out.exists():
        print(f"Checkpoint already exists: {out}  — delete to re-collect.")
        return

    data_path = ckpt_data()
    if not data_path.exists():
        raise FileNotFoundError(f"Run stage1 first: {data_path} not found")

    with open(data_path, "rb") as f:
        data = pickle.load(f)
    train_corpus = data["train_corpus"]
    print(f"Stage 2 — collecting activations for {len(train_corpus)} records …")

    DEVICE = get_device()
    dtype  = torch.float16 if DEVICE.type == "cuda" else torch.float32
    print(f"  model={MODEL_NAME}  device={DEVICE}  dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype)
    model.eval().to(DEVICE)

    layers = resolve_layers(model, LAYERS)
    print(f"  layers={layers}")

    if DEVICE.type == "cuda":
        print(f"  batch_size={ACT_BATCH} (CUDA batched)")
        raw_by_layer = collect_batched(train_corpus, model, tokenizer,
                                       layers, DEVICE, ACT_BATCH)
    else:
        print("  batch_size=1 (MPS/CPU — padding causes NaN)")
        raw_by_layer = collect_single(train_corpus, model, tokenizer, layers, DEVICE)

    for h_ref in []:   # handles already removed inside collect_*
        pass
    del model
    if DEVICE.type == "mps":
        torch.mps.empty_cache()

    # normalise per layer
    norm, mu_d, sig_d = {}, {}, {}
    for l in layers:
        raw = raw_by_layer[l]  # already float32 from hook
        nan_frac = raw.isnan().float().mean().item()
        if nan_frac > 0:
            print(f"  WARNING L{l:2d}: {nan_frac:.1%} NaN values in activations — "
                  f"likely float16 instability under left-padding. Zeroing NaNs.")
        raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        mu  = raw.mean(dim=0, keepdim=True)
        sig = raw.std(dim=0,  keepdim=True)
        if sig.max().item() < 1e-6:
            print(f"  ERROR  L{l:2d}: all activations are zero/constant after NaN removal — "
                  f"this layer's SAE will be uninformative. "
                  f"Try SAE_DEVICE=cpu or reduce batch size (SAE_ACT_BATCH).")
        sig = sig.clamp(min=1e-6)
        norm[l]  = (raw - mu) / sig
        mu_d[l]  = mu
        sig_d[l] = sig
        print(f"  L{l:2d}: {raw.shape}  mean={norm[l].mean():.4f}  std={norm[l].std():.4f}")

    torch.save(dict(norm=norm, mu=mu_d, sig=sig_d, layers=layers), out)
    print(f"  Saved → {out}  ({out.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
