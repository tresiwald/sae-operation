"""
Shared configuration for all pipeline stages.
Override any value via environment variables (all prefixed SAE_):
  SAE_MODEL_NAME, SAE_LAYERS, SAE_N_PER_CELL, SAE_EPOCHS, …
"""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / os.getenv("SAE_OUT_DIR", "results")
OUT_DIR.mkdir(exist_ok=True)

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("SAE_MODEL_NAME", "google/gemma-3-1b-pt")

# Layers to sweep.  Env var: comma-separated ints, e.g. "4,8,13,17,22,25"
# None → sweep all layers (resolved after model is loaded).
_layers_env = os.getenv("SAE_LAYERS", "4,8,13,17,22,25")
LAYERS: list[int] | None = [int(x) for x in _layers_env.split(",") if x.strip()] \
                            if _layers_env.lower() not in ("none", "all", "") else None

# ── Data ──────────────────────────────────────────────────────────────────────
N_PER_CELL   = int(os.getenv("SAE_N_PER_CELL",   "150"))
N_CTRL       = int(os.getenv("SAE_N_CTRL",       "1500"))
HOLDOUT_FRAC = float(os.getenv("SAE_HOLDOUT_FRAC", "0.20"))
OPS_EVAL     = ["add", "sub", "mul", "div"]
FORMATS      = ["symbolic", "mixed", "verbal"]

# ── SAE ───────────────────────────────────────────────────────────────────────
SAE_K        = int(os.getenv("SAE_K",        "32"))
SAE_RATIO    = int(os.getenv("SAE_RATIO",    "8"))     # d_sae = d_model × SAE_RATIO
SAE_LR       = float(os.getenv("SAE_LR",    "3e-4"))
SAE_EPOCHS   = int(os.getenv("SAE_EPOCHS",  "30"))
SAE_BATCH    = int(os.getenv("SAE_BATCH",   "512"))
WARMUP_STEPS = int(os.getenv("SAE_WARMUP",  "200"))
AUX_W        = float(os.getenv("SAE_AUX_W", str(1/32)))
DEAD_THR     = float(os.getenv("SAE_DEAD_THR", "1e-4"))

# ── Analysis ──────────────────────────────────────────────────────────────────
FP_THRESHOLD  = float(os.getenv("SAE_FP_THRESHOLD", "0.5"))
CHEAT_SAMPLE  = int(os.getenv("SAE_CHEAT_SAMPLE",  "300"))

# ── Accuracy (optional) ───────────────────────────────────────────────────────
MEASURE_ACCURACY = os.getenv("SAE_MEASURE_ACCURACY", "0") == "1"
ACC_SAMPLE       = int(os.getenv("SAE_ACC_SAMPLE", "200"))

# ── Checkpoint filenames ───────────────────────────────────────────────────────
def ckpt_data()       -> Path: return OUT_DIR / "data.pkl"
def ckpt_acts()       -> Path: return OUT_DIR / "acts_checkpoint.pt"
def ckpt_sae(layer)   -> Path: return OUT_DIR / f"sae_L{layer}.pt"
def ckpt_results()    -> Path: return OUT_DIR / "results.json"
def ckpt_accuracy()   -> Path: return OUT_DIR / "accuracy.json"

# ── Device helper ─────────────────────────────────────────────────────────────
def get_device():
    import torch
    override = os.getenv("SAE_DEVICE")
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
