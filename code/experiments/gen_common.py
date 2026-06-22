"""
Shared helpers for the GENERATION-time SAE extension.

Unlike the main pipeline (which captures the residual only at the prompt-final
`=` token), these experiments capture the residual at every ANSWER-token
position while the model produces the answer — the place where digit-by-digit
computation actually happens.

Baseline design: each compute problem gets a magnitude-matched COPY twin (same
target number, no arithmetic). Contrasting compute vs copy at matched answer
positions isolates COMPUTATION from MAGNITUDE — the confound exp5 exposed.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import CKPT_DIR
from pipeline.stage3_sae import TopKSAE


def gen_acts_path():        return CKPT_DIR / "gen_acts.pt"
def gen_sae_path(layer):    return CKPT_DIR / f"gen_sae_L{layer}.pt"


def load_gen_sae(layer):
    ck   = torch.load(gen_sae_path(layer), weights_only=False)
    d_in = ck["state_dict"]["W_enc"].shape[0]
    sae  = TopKSAE(d_in, ck["d_sae"], ck["k"])
    sae.load_state_dict(ck["state_dict"])
    sae.eval()
    return sae


# property names exposed to the interpretation step
GEN_PROPS = ["result_mag", "a_mag", "b_mag",        # magnitude (old story)
             "place_from_right", "target_digit", "is_leading"]  # generation-only


def properties_gen(m):
    """Interpretable properties of one answer-token position. Magnitude props are
    always defined; per-digit props are NaN unless the answer tokenised cleanly
    into single digits (so place/value are meaningful)."""
    a, b, c = m.get("a"), m.get("b"), m.get("expected")
    if a is None or b is None or c is None:
        return None
    pfr = m.get("place_from_right", -1)
    tgt = m.get("target_digit", -1)
    return {
        "result_mag":       np.log10(abs(c) + 1),
        "a_mag":            np.log10(abs(a) + 1),
        "b_mag":            np.log10(abs(b) + 1),
        "place_from_right": float(pfr) if pfr >= 0 else np.nan,
        "target_digit":     float(tgt) if tgt >= 0 else np.nan,
        "is_leading":       float(m.get("is_leading", 0)),
    }
