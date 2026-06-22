"""
Stage 1 — Data generation.

Outputs:
  results/data.pkl   dict with keys: train_corpus, hold_per_op, hold_cheat,
                     hold_multi_compute, hold_multi_cheat
"""

import os, pickle, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_gen import (make_dataset, make_loguniform_dataset,
                      make_multi_op_holdout, split_dataset, make_ctrl_data)
from pipeline.config import (
    N_PER_CELL, N_CTRL, HOLDOUT_FRAC, OPS_EVAL, ckpt_data
)

# SAE_DATA_MODE=loguniform → magnitude-stratified sampling (decorrelates
# magnitude from operation); anything else → the original per-bin sampler.
DATA_MODE  = os.getenv("SAE_DATA_MODE", "standard")
N_PER_OP   = int(os.getenv("SAE_N_PER_OP", "1500"))

def main():
    print(f"Stage 1 — generating dataset … (mode={DATA_MODE})")
    if DATA_MODE == "loguniform":
        all_data = make_loguniform_dataset(n_per_op=N_PER_OP, ops=OPS_EVAL)
    else:
        all_data = make_dataset(n_per_cell=N_PER_CELL, ops=OPS_EVAL)
    train, hold_per_op, hold_cheat = split_dataset(all_data, holdout_frac=HOLDOUT_FRAC)
    ctrl_data      = make_ctrl_data(N_CTRL)
    train_corpus   = [r for r in train if r["variant"] in ("compute", "copy")] + ctrl_data
    hold_multi = make_multi_op_holdout(n_per_cell=100, ops=OPS_EVAL)
    hold_multi_compute = [r for r in hold_multi if r["variant"] == "compute"]
    hold_multi_cheat   = [r for r in hold_multi if r["variant"] == "cheat"]

    print(f"  Train corpus        : {len(train_corpus)}")
    print(f"  H1 per-op compute   : {len(hold_per_op)}")
    print(f"  H2 per-op cheat     : {len(hold_cheat)}")
    print(f"  H3 multi-op compute : {len(hold_multi_compute)}")
    print(f"  H4 multi-op cheat   : {len(hold_multi_cheat)}")

    out = ckpt_data()
    with open(out, "wb") as f:
        pickle.dump(dict(
            train_corpus=train_corpus,
            hold_per_op=hold_per_op,
            hold_cheat=hold_cheat,
            hold_multi_compute=hold_multi_compute,
            hold_multi_cheat=hold_multi_cheat,
        ), f)
    print(f"  Saved → {out}")

if __name__ == "__main__":
    main()
