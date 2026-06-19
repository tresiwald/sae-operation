"""
Stage 5 — Model accuracy on H1 holdout (optional, slow).

Reads  : results/data.pkl
Outputs: results/accuracy.json
         results/accuracy.png

Set SAE_ACC_SAMPLE to cap records per op (default 200).
"""

import json, pickle, re, sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import (
    MODEL_NAME, OPS_EVAL, FORMATS, ACC_SAMPLE,
    get_device, ckpt_data, ckpt_accuracy, OUT_DIR
)
from data_gen import BINS

sns.set_theme(style="whitegrid", font_scale=0.9)
COLORS = {"add": "steelblue", "sub": "seagreen", "mul": "darkorange", "div": "mediumpurple"}


def main():
    with open(ckpt_data(), "rb") as f:
        data = pickle.load(f)
    hold_per_op = data["hold_per_op"]

    import random
    rng = random.Random(0)
    sample = []
    for op in OPS_EVAL:
        grp = [r for r in hold_per_op if r["op"] == op and r["variant"] == "compute"]
        rng.shuffle(grp)
        sample.extend(grp[:ACC_SAMPLE])

    print(f"Stage 5 — accuracy on {len(sample)} sampled H1 records …")
    DEVICE    = get_device()
    dtype     = torch.float16 if DEVICE.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype)
    model.eval().to(DEVICE)

    with torch.no_grad():
        for rec in tqdm(sample, desc="acc"):
            if rec.get("expected") is None:
                rec["correct"] = None
                continue
            inp = tokenizer(rec["prompt"], return_tensors="pt").to(DEVICE)
            out = model.generate(inp["input_ids"], max_new_tokens=6, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
            decoded = tokenizer.decode(out[0, inp["input_ids"].shape[1]:]).strip()
            m = re.match(r"\d+", decoded)
            rec["correct"] = (int(m.group()) == rec["expected"]) if m else False

    del model

    acc_table = {}
    print(f"\n  {'op':4s}  {'fmt':8s}  {'bin':3s}  {'n':>4s}  acc")
    for op in OPS_EVAL:
        for fmt in FORMATS:
            for bn in BINS:
                grp = [r for r in sample
                       if r["op"]==op and r["fmt"]==fmt and r["bin"]==bn]
                if not grp:
                    continue
                a = float(np.mean([r["correct"] for r in grp]))
                acc_table[f"{op}_{fmt}_{bn}"] = dict(acc=a, n=len(grp))
                print(f"  {op:4s}  {fmt:8s}  {bn:3s}  {len(grp):>4d}  {a:.1%}")

    with open(ckpt_accuracy(), "w") as f:
        json.dump(acc_table, f, indent=2)
    print(f"Saved → {ckpt_accuracy()}")

    # Heatmap per format
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5), sharey=True)
    for ax, fmt in zip(axes, FORMATS):
        mat = np.full((len(OPS_EVAL), len(BINS)), np.nan)
        for i, op in enumerate(OPS_EVAL):
            for j, bn in enumerate(BINS):
                v = acc_table.get(f"{op}_{fmt}_{bn}", {}).get("acc")
                if v is not None:
                    mat[i, j] = v
        sns.heatmap(mat, annot=True, fmt=".0%", ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
                    xticklabels=list(BINS), yticklabels=OPS_EVAL,
                    cbar=(fmt == FORMATS[-1]), linewidths=0.5)
        ax.set_title(fmt, fontweight="bold"); ax.set_xlabel("Operand bin")
    fig.suptitle(f"Accuracy — {MODEL_NAME}", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot  → {OUT_DIR}/accuracy.png")


if __name__ == "__main__":
    main()
