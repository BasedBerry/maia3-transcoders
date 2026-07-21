"""Train a cross-layer transcoder on Maia 3.

Single-GPU (5M):
    CUDA_VISIBLE_DEVICES=2 python train.py --model 5m --data /path/*.pgn.zst \
        --out results/clt-5m --expansion 16 --k 32 --steps 50000

Multi-GPU DDP (23M on 2 GPUs):
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py --model 23m \
        --data /path/*.pgn.zst --out results/clt-23m --expansion 16 --k 32 --steps 50000

Both can run at once (disjoint CUDA_VISIBLE_DEVICES). The base model is replicated per rank;
only the CLT gradients are all-reduced. Games are sharded across ranks so no position is
seen twice per step.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from clt import CLTConfig, CrossLayerTranscoder
from capture import ActivationCapturer, build_cfg
from buffer import ActivationBuffer, BufferConfig
from data import DataConfig, PositionStream

from maia3.models import MAIA3Model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="5m", help="Maia alias: 5m / 23m / ...")
    p.add_argument("--checkpoint-path", default=None, help="Local .pt (else download from HF)")
    p.add_argument("--trust-checkpoint", action="store_true")
    p.add_argument("--data", nargs="+", default=None, help="PGN globs (.pgn/.pgn.zst)")
    p.add_argument("--out", default="results/clt", help="Output dir for checkpoints/logs")

    # CLT hyperparams
    p.add_argument("--activation", default="jumprelu", choices=["batchtopk", "jumprelu"])
    p.add_argument("--expansion", type=int, default=16, help="F = expansion * d_model")
    p.add_argument("--n-features", type=int, default=None, help="Override F directly")
    p.add_argument("--k", type=int, default=32, help="BatchTopK avg total L0 per token")
    p.add_argument("--per-layer-topk", action="store_true",
                   help="Give each layer its own k/L budget (prevents cross-layer starvation)")
    p.add_argument("--k-aux", type=int, default=512)
    p.add_argument("--aux-alpha", type=float, default=1.0 / 32)
    p.add_argument("--dead-steps", type=int, default=1000)

    # JumpReLU (activation=jumprelu): Anthropic Circuit Tracing recipe -- SOFT threshold
    # a=relu(z-tau) + a gentle mean-reduced tanh sparsity penalty (decoder-norm weighted),
    # FIXED lambda with warmup, FREE decoder norm. No top-k, no STE, no hard L0 target.
    p.add_argument("--tau-init", type=float, default=0.0, help="soft-threshold init (relu(z-tau))")
    p.add_argument("--tanh-c", type=float, default=1.0, help="tanh penalty scale c")
    p.add_argument("--sparsity-lambda", type=float, default=1e-3, help="tanh penalty weight (tune up for lower L0)")
    p.add_argument("--lambda-warmup", type=int, default=2000, help="Linear warmup steps for lambda (0 -> lambda)")
    # Lambda ANNEAL: after recon converges, decay lambda toward a floor to remove the tanh
    # consolidation pressure that otherwise kills features late in training.
    p.add_argument("--lambda-anneal-start", type=int, default=5000, help="Step to begin decaying lambda")
    p.add_argument("--lambda-anneal-len", type=int, default=5000, help="Steps over which to decay lambda")
    p.add_argument("--lambda-anneal-frac", type=float, default=0.1, help="Decay lambda to this fraction (floor)")

    # optim
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # buffer / data
    p.add_argument("--buffer-size", type=int, default=500_000)
    p.add_argument("--capture-batch", type=int, default=256)
    p.add_argument("--train-batch", type=int, default=8192)
    p.add_argument("--pool-device", default="cpu")
    p.add_argument("--elo-low", type=int, default=800)
    p.add_argument("--elo-high", type=int, default=2800)
    p.add_argument("--max-pos-per-game", type=int, default=20)

    # logging / ckpt
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--eval-interval", type=int, default=1000)
    p.add_argument("--ckpt-interval", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)

    # testing
    p.add_argument("--smoke-random-init", action="store_true",
                   help="Skip checkpoint load; random base weights (pipeline test only)")
    return p.parse_args()


def setup_ddp():
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local)
        return rank, world, local, True
    return 0, 1, 0, False


def load_base_model(args, cfg):
    if args.smoke_random_init:
        return MAIA3Model(cfg).to(cfg.device).eval()
    from maia3.uci import load_model
    from maia3.model_registry import resolve_checkpoint_path
    cfg.checkpoint_path = args.checkpoint_path or resolve_checkpoint_path(cfg.model_spec)
    cfg.trust_checkpoint = args.trust_checkpoint
    return load_model(cfg)


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    prog = (step - args.warmup) / max(1, args.steps - args.warmup)
    return 0.5 * args.lr * (1 + math.cos(math.pi * min(1.0, prog)))


def main():
    args = parse_args()
    rank, world, local, is_ddp = setup_ddp()
    is_main = rank == 0
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed + rank)

    # --- base model (frozen, per-rank) ---
    mcfg = build_cfg(args.model, device=device)
    base = load_base_model(args, mcfg)
    for p_ in base.parameters():
        p_.requires_grad_(False)
    cap = ActivationCapturer(base, mcfg)

    # --- data / buffer (per-rank shard) --- built before the CLT so we can mean-init b_pre
    paths = []
    for g in (args.data or []):
        paths.extend(sorted(glob.glob(g)))
    if not paths and not args.smoke_random_init:
        raise SystemExit("No PGN files matched --data")
    dcfg = DataConfig(history=mcfg.history, elo_low=args.elo_low, elo_high=args.elo_high,
                      max_positions_per_game=args.max_pos_per_game, seed=args.seed)
    stream = PositionStream(paths, dcfg, rank=rank, world_size=world)
    # "cuda"/"same" -> this rank's own device (avoids pinning all ranks to cuda:0 under DDP)
    pool_dev = device if args.pool_device in ("cuda", "same") else args.pool_device
    bcfg = BufferConfig(buffer_size=args.buffer_size, capture_batch=args.capture_batch,
                        train_batch=args.train_batch, pool_device=pool_dev,
                        compute_device=device)
    buf = ActivationBuffer(stream, cap, bcfg)

    # --- CLT ---
    F = args.n_features or args.expansion * mcfg.dim_vit
    ccfg = CLTConfig(n_layers=mcfg.num_blocks, d_model=mcfg.dim_vit, n_features=F,
                     activation=args.activation,
                     k=args.k, per_layer_topk=args.per_layer_topk,
                     k_aux=args.k_aux, aux_alpha=args.aux_alpha,
                     dead_steps_threshold=args.dead_steps,
                     tau_init=args.tau_init, tanh_c=args.tanh_c)
    clt = CrossLayerTranscoder(ccfg).to(device)
    # Mean-init the pre-encoder bias from the data (before DDP wrap so rank 0's value
    # broadcasts to all ranks). Standard SAE practice; centers features at init.
    with torch.no_grad():
        x0, _ = buf.next()
        clt.init_pre_bias(x0.mean(dim=0))                    # (L, d)
    model = DDP(clt, device_ids=[local]) if is_ddp else clt
    raw = model.module if is_ddp else model

    opt = torch.optim.AdamW(clt.parameters(), lr=args.lr, betas=(0.9, 0.999))

    is_jr = args.activation == "jumprelu"

    def lambda_at(step):
        # warmup 0->lambda, hold, then anneal lambda -> lambda*frac to drop the consolidation
        # pressure once recon has converged (prevents the late-training feature death).
        lm = args.sparsity_lambda
        if step < args.lambda_warmup:
            return lm * step / max(1, args.lambda_warmup)
        if step < args.lambda_anneal_start:
            return lm
        frac = min(1.0, (step - args.lambda_anneal_start) / max(1, args.lambda_anneal_len))
        return lm * (1.0 - frac * (1.0 - args.lambda_anneal_frac))

    out = Path(args.out)
    if is_main:
        out.mkdir(parents=True, exist_ok=True)
        extra = (f"lambda={args.sparsity_lambda}->{args.sparsity_lambda*args.lambda_anneal_frac:g} "
                 f"(anneal {args.lambda_anneal_start}-{args.lambda_anneal_start+args.lambda_anneal_len}) "
                 f"tanh_c={args.tanh_c}" if is_jr else f"k={args.k}")
        print(f"[cfg] model={args.model} d={mcfg.dim_vit} L={mcfg.num_blocks} F={F} "
              f"act={args.activation} {extra} ddp={is_ddp} world={world} device={device}", flush=True)

    t0 = time.time()
    for step in range(args.steps):
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step, args)

        x, y = buf.next()
        model.train()
        out_dict = model(x, y)
        lam = lambda_at(step)
        if is_jr:
            loss = out_dict["recon_loss"] + lam * out_dict["tanh_penalty"]
        else:
            loss = out_dict["recon_loss"] + args.aux_alpha * out_dict["aux_loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(clt.parameters(), args.grad_clip)
        opt.step()
        if not is_jr:
            raw.normalize_decoder()   # BatchTopK: unit-norm decoder. JumpReLU: norm free (penalized).

        if is_main and step % args.log_interval == 0:
            fvu = out_dict["fvu_per_layer"].mean().item()
            toks = (step + 1) * args.train_batch * world
            sparsity = (f"lam {lam:.2e} pen {out_dict['tanh_penalty'].item():.1f}" if is_jr
                        else f"aux {out_dict['aux_loss'].item():.4f}")
            print(f"step {step:>6} recon {out_dict['recon_loss'].item():.4f} {sparsity} "
                  f"fvu {fvu:.4f} L0 {out_dict['l0'].item():.1f} "
                  f"dead {int(out_dict['n_dead'].item())} lr {opt.param_groups[0]['lr']:.2e} "
                  f"tok {toks/1e6:.1f}M ep {buf.epochs} {toks/(time.time()-t0)/1e3:.0f}k tok/s",
                  flush=True)

        if is_main and step > 0 and step % args.eval_interval == 0:
            _eval(raw, buf, step)

        if is_main and step > 0 and step % args.ckpt_interval == 0:
            _save(raw, ccfg, args, out, step)

    if is_main:
        _save(raw, ccfg, args, out, args.steps, final=True)
    if is_ddp:
        dist.destroy_process_group()


@torch.no_grad()
def _eval(raw: CrossLayerTranscoder, buf: ActivationBuffer, step: int):
    # BatchTopK needs its eval threshold calibrated; JumpReLU is deterministic (train==eval).
    theta = 0.0
    if raw.cfg.activation == "batchtopk":
        theta = raw.calibrate_theta([buf.next()[0] for _ in range(5)])
    raw.eval()
    x, y = buf.next()
    o = raw(x, y)
    raw.train()
    print(f"  [eval @ {step}] theta {theta:.3f} recon {o['recon_loss'].item():.4f} "
          f"L0 {o['l0'].item():.1f} fvu/layer "
          f"{[round(v,3) for v in o['fvu_per_layer'].tolist()]}", flush=True)


def _save(raw, ccfg: CLTConfig, args, out: Path, step: int, final=False):
    name = "clt_final.pt" if final else f"clt_step{step}.pt"
    torch.save({"state_dict": raw.state_dict(), "clt_config": vars(ccfg),
                "args": vars(args), "step": step}, out / name)
    print(f"  [ckpt] saved {out/name}", flush=True)


if __name__ == "__main__":
    main()
