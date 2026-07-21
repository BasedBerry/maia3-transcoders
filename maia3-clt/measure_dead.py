"""Measure the TRUE per-feature firing-frequency distribution of a trained CLT.

Our training-time `dead` metric is binary at a 500-step (~4M-token) window, which (a) is
~2.4x stricter than the field standard (1 firing per 10^7 tokens) and (b) hides whether the
collapse is a hard bimodal (fire-constantly vs never) or a smooth tail of rare-but-alive
features. This script runs a converged CLT over a large token budget, records each feature's
activation frequency, and reports dead-feature counts at several thresholds plus the
log-frequency histogram — so we know if "93% dead" is real or a measurement artifact.

Usage (on the cluster, needs the base checkpoint + a GPU):
  python measure_dead.py --model 5m --clt-ckpt results/clt-5m-v2/clt_final.pt \
      --data '/grace/u/geilender/lichess_db_standard_rated_2022-07.pgn' --tokens 10000000
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from clt import CLTConfig, CrossLayerTranscoder
from capture import ActivationCapturer, build_cfg
from buffer import ActivationBuffer, BufferConfig
from data import DataConfig, PositionStream


def load_clt(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = CLTConfig(**ck["clt_config"])
    clt = CrossLayerTranscoder(cfg).to(device)
    # Drop recomputed buffers (theta shape changed across versions; all recalibrated below).
    sd = {k: v for k, v in ck["state_dict"].items()
          if k not in ("theta", "theta_initialized", "steps_since_fired")}
    missing, unexpected = clt.load_state_dict(sd, strict=False)
    print(f"loaded {ckpt_path} (step {ck.get('step','?')}); "
          f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return clt, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="5m")
    p.add_argument("--clt-ckpt", required=True)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--tokens", type=float, default=10_000_000, help="square-tokens to measure over")
    p.add_argument("--calib-batches", type=int, default=10)
    p.add_argument("--train-batch", type=int, default=8192)
    p.add_argument("--capture-batch", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=300_000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    dev = args.device
    mcfg = build_cfg(args.model, device=dev)

    # base model (real weights)
    from maia3.uci import load_model
    from maia3.model_registry import resolve_checkpoint_path
    mcfg.checkpoint_path = resolve_checkpoint_path(mcfg.model_spec)
    mcfg.trust_checkpoint = False
    base = load_model(mcfg)
    for pr in base.parameters():
        pr.requires_grad_(False)
    cap = ActivationCapturer(base, mcfg)

    paths = []
    for g in args.data:
        paths.extend(sorted(glob.glob(g)))
    stream = PositionStream(paths, DataConfig(history=mcfg.history))
    buf = ActivationBuffer(stream, cap, BufferConfig(
        buffer_size=args.buffer_size, capture_batch=args.capture_batch,
        train_batch=args.train_batch, pool_device=dev, compute_device=dev))

    clt, ccfg = load_clt(args.clt_ckpt, dev)

    # BatchTopK needs its eval threshold calibrated; JumpReLU is deterministic (uses log_theta).
    if ccfg.activation == "batchtopk":
        clt.train()
        theta = clt.calibrate_theta([buf.next()[0] for _ in range(args.calib_batches)])
        print(f"calibrated theta (mean {theta:.3f})", flush=True)
    clt.eval()

    L, Fdim = ccfg.n_layers, ccfg.n_features
    counts = torch.zeros(L, Fdim, device=dev)
    seen = 0
    with torch.no_grad():
        while seen < args.tokens:
            x, _ = buf.next()
            _, info = clt.encode(x)
            counts += info["mask"].sum(dim=0)          # (L, F) active-token counts
            seen += x.shape[0]
    freq = (counts / seen).cpu()                       # per-token firing frequency

    print(f"\n=== measured over {seen/1e6:.1f}M square-tokens ===")
    total = L * Fdim
    print("dead-feature count at various frequency thresholds:")
    for thr in [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
        n = int((freq <= thr).sum()) if thr > 0 else int((freq == 0).sum())
        label = "never fired" if thr == 0 else f"freq <= {thr:.0e}"
        print(f"  {label:>16}: {n:>6}/{total} ({100*n/total:5.1f}%)")

    # log10-frequency histogram of features that fired at least once
    alive = freq[freq > 0]
    print(f"\nfired at least once: {alive.numel()}/{total} ({100*alive.numel()/total:.1f}%)")
    if alive.numel():
        logf = alive.log10()
        bins = [-8, -7, -6, -5, -4, -3, -2, -1, 0]
        h = torch.histc(logf.clamp(bins[0], bins[-1]), bins=len(bins) - 1, min=bins[0], max=bins[-1])
        print("log10(freq) histogram of ever-firing features:")
        for i in range(len(bins) - 1):
            print(f"  [1e{bins[i]:>3}, 1e{bins[i+1]:>3}): {int(h[i]):>6}")

    # per-layer never-fired
    never = (freq == 0)
    print("\nper-layer never-fired:")
    for l in range(L):
        n = int(never[l].sum())
        print(f"  layer {l}: {n:>5}/{Fdim} ({100*n/Fdim:5.1f}%)")


if __name__ == "__main__":
    main()
