"""
Train a SAE on the GENERATION-time activations (answer-token positions).

Reuses the pipeline's TopK SAE trainer. The SAE now learns features over residual
states the model occupies WHILE producing each answer digit — not the static
prompt-final snapshot.

Reads  : results/checkpoints/gen_acts.pt
Outputs: results/checkpoints/gen_sae_L{layer}.pt
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.stage3_sae import train_one_layer
from experiments.gen_common import gen_acts_path, gen_sae_path


def main():
    ck = torch.load(gen_acts_path(), weights_only=False)
    layer = ck["layer"]
    print(f"Gen-train — SAE on {ck['acts'].shape[0]} generation positions "
          f"(layer {layer})")

    result = train_one_layer(layer, ck["acts"])
    out = gen_sae_path(layer)
    torch.save(result, out)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
