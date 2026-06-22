"""
Stage 3 — Train one TopK SAE for a single layer.

Called with --layer L (or SLURM_ARRAY_TASK_ID maps to the layer index).

Reads  : results/acts_checkpoint.pt
Outputs: results/sae_L{layer}.pt
           keys: state_dict, history, var_expl, n_live, layer, d_sae, k

Submit as a SLURM array job (one task per layer).  The array index is used
to select the layer from the list in the checkpoint.
"""

import argparse, sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import (
    SAE_K, SAE_RATIO, SAE_LR, SAE_EPOCHS, SAE_BATCH,
    WARMUP_STEPS, AUX_W, DEAD_THR, ckpt_acts, ckpt_sae, get_device
)


# ── TopK SAE ──────────────────────────────────────────────────────────────────
class TopKSAE(nn.Module):
    def __init__(self, d_in, d_sae, k):
        super().__init__()
        self.k = k
        self.pre_bias = nn.Parameter(torch.zeros(d_in))
        self.W_enc    = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc    = nn.Parameter(torch.zeros(d_sae))
        self.W_dec    = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec    = nn.Parameter(torch.zeros(d_in))
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.data = nn.functional.normalize(self.W_dec.data, dim=1)

    def encode(self, x):
        pre  = (x - self.pre_bias) @ self.W_enc + self.b_enc
        vals, idx = pre.topk(self.k, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, idx, vals.clamp(min=0))
        return acts

    def decode(self, acts):  return acts @ self.W_dec + self.b_dec
    def forward(self, x):   acts = self.encode(x); return acts, self.decode(acts)

    @torch.no_grad()
    def normalise_decoder(self):
        self.W_dec.data = nn.functional.normalize(self.W_dec.data, dim=1)


def train_one_layer(layer: int, train_acts: torch.Tensor, device=None) -> dict:
    D_MODEL = train_acts.shape[1]
    D_SAE   = D_MODEL * SAE_RATIO
    device  = device or get_device()

    sae        = TopKSAE(D_MODEL, D_SAE, SAE_K).to(device)
    optimizer  = torch.optim.Adam(sae.parameters(), lr=SAE_LR)
    loader     = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_acts),   # kept on CPU; moved per-batch
        batch_size=SAE_BATCH, shuffle=True,
    )
    feat_usage = torch.zeros(D_SAE, device=device)
    history    = []
    step       = 0

    print(f"  Training SAE  layer={layer}  d_sae={D_SAE}  k={SAE_K}  "
          f"epochs={SAE_EPOCHS}  device={device}")
    for epoch in range(SAE_EPOCHS):
        e_mse = 0.0
        for (x,) in loader:
            x = x.to(device)
            lr = SAE_LR * min(1.0, step / max(1, WARMUP_STEPS))
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            step += 1

            acts, x_hat = sae(x)
            with torch.no_grad():
                feat_usage = 0.99 * feat_usage + 0.01 * (acts > 0).float().mean(0)

            mse  = (x - x_hat).pow(2).mean()
            dead = (feat_usage < DEAD_THR).float()
            aux  = ((x - x_hat).detach() - (acts * dead) @ sae.W_dec).pow(2).mean() \
                   if dead.sum() > 0 else torch.tensor(0.0)

            optimizer.zero_grad()
            (mse + AUX_W * aux).backward()
            nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimizer.step()
            sae.normalise_decoder()
            e_mse += mse.item()

        n_dead   = (feat_usage < DEAD_THR).sum().item()
        mean_mse = e_mse / len(loader)
        if not torch.isfinite(torch.tensor(mean_mse)):
            print(f"  NaN at epoch {epoch+1} — stopping"); break
        history.append(dict(epoch=epoch+1, mse=mean_mse, dead=n_dead))
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  ep {epoch+1:2d}/{SAE_EPOCHS}  mse={mean_mse:.5f}  dead={n_dead}/{D_SAE}")

    sae.eval()
    with torch.no_grad():
        s = train_acts[:1000].to(device)
        var_expl = float(1 - (s - sae(s)[1]).pow(2).sum() / s.pow(2).sum())
    n_live = int((feat_usage > DEAD_THR).sum())
    print(f"  Done: var_expl={var_expl:.2%}  live={n_live}/{D_SAE}")

    # move params back to CPU so the checkpoint loads on any device
    cpu_state = {k: v.detach().cpu() for k, v in sae.state_dict().items()}
    return dict(
        state_dict=cpu_state,
        history=history,
        var_expl=var_expl,
        n_live=n_live,
        layer=layer,
        d_sae=D_SAE,
        k=SAE_K,
    )


def main():
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer index to train. If omitted, uses SLURM_ARRAY_TASK_ID "
                             "as an index into the layers list.")
    args = parser.parse_args()

    acts_path = ckpt_acts()
    if not acts_path.exists():
        raise FileNotFoundError(f"Run stage2 first: {acts_path} not found")

    print(f"Loading activations from {acts_path} …")
    ckpt = torch.load(acts_path, weights_only=True)
    available_layers: list[int] = ckpt["layers"]

    # Resolve which layer to train
    if args.layer is not None:
        layer = args.layer
    else:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        layer   = available_layers[task_id]

    if layer not in available_layers:
        raise ValueError(f"Layer {layer} not in checkpoint: {available_layers}")

    out = ckpt_sae(layer)
    if out.exists():
        print(f"Checkpoint exists: {out}  — skipping.")
        return

    print(f"Stage 3 — training SAE for layer {layer}")
    train_acts = ckpt["norm"][layer]
    result     = train_one_layer(layer, train_acts)

    torch.save(result, out)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
