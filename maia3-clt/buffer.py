"""On-the-fly activation shuffle buffer for CLT training.

We never stage activations to disk (8 layers x in/out per square-token is petabytes at
budget). Instead we keep a large shuffled pool in memory, refilled by running the frozen
base model on fresh positions, and serve shuffled minibatches. Shuffling across many games
and squares breaks the strong within-board correlation.

Pool layout: two tensors (buffer_size, L, d) for MLP in/out, fp16 to save RAM. Minibatches
are cast to fp32 and moved to the compute device.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch


@dataclass
class BufferConfig:
    buffer_size: int = 500_000     # samples (square-tokens) held in the pool
    capture_batch: int = 256       # positions per base-model forward (yields *64 samples)
    train_batch: int = 8192        # square-tokens per CLT optimizer step
    refill_fraction: float = 0.5   # refill when this fraction of the pool is consumed
    pool_device: str = "cpu"       # where the big pool lives ('cpu' safe; 'cuda' if VRAM allows)
    compute_device: str = "cuda"   # where minibatches are trained
    store_dtype: torch.dtype = torch.float16


class ActivationBuffer:
    def __init__(self, stream, capturer, cfg: BufferConfig):
        self.stream = stream
        self.capturer = capturer
        self.cfg = cfg
        self.L = capturer.n_layers
        self.d = capturer.d_model
        self._it = iter(stream)
        self.epochs = 0
        self.pool_x = torch.empty(0, self.L, self.d, dtype=cfg.store_dtype, device=cfg.pool_device)
        self.pool_y = torch.empty(0, self.L, self.d, dtype=cfg.store_dtype, device=cfg.pool_device)
        self.ptr = 0
        self._refill()

    def _next_positions(self, n_positions):
        """Collate up to n_positions from the stream, restarting it across epochs."""
        toks, selfs, oppos = [], [], []
        while len(toks) < n_positions:
            try:
                t, se, oe = next(self._it)
            except StopIteration:
                self.epochs += 1
                self._it = iter(self.stream)
                continue
            toks.append(t); selfs.append(se); oppos.append(oe)
        dev = self.capturer.cfg.device
        tokens = torch.stack(toks).to(dev)
        self_elos = torch.tensor(selfs, dtype=torch.long, device=dev)
        oppo_elos = torch.tensor(oppos, dtype=torch.long, device=dev)
        return tokens, self_elos, oppo_elos

    def _refill(self):
        cfg = self.cfg
        # keep the unconsumed tail, drop the already-served head
        keep_x = self.pool_x[self.ptr:]
        keep_y = self.pool_y[self.ptr:]
        new_x = [keep_x]; new_y = [keep_y]
        have = keep_x.shape[0]
        while have < cfg.buffer_size:
            tokens, self_elos, oppo_elos = self._next_positions(cfg.capture_batch)
            x, y = self.capturer.capture(tokens, self_elos, oppo_elos)  # (B*64, L, d) fp32
            new_x.append(x.to(cfg.pool_device, cfg.store_dtype))
            new_y.append(y.to(cfg.pool_device, cfg.store_dtype))
            have += x.shape[0]
        pool_x = torch.cat(new_x, dim=0)
        pool_y = torch.cat(new_y, dim=0)
        # shuffle jointly
        perm = torch.randperm(pool_x.shape[0], device=cfg.pool_device)
        self.pool_x = pool_x[perm]
        self.pool_y = pool_y[perm]
        self.ptr = 0

    def next(self):
        """Return (x, y) minibatch: (train_batch, L, d) fp32 on compute_device."""
        cfg = self.cfg
        if self.ptr + cfg.train_batch > self.pool_x.shape[0] or \
           self.ptr > cfg.refill_fraction * self.pool_x.shape[0]:
            self._refill()
        sl = slice(self.ptr, self.ptr + cfg.train_batch)
        x = self.pool_x[sl].to(cfg.compute_device, torch.float32)
        y = self.pool_y[sl].to(cfg.compute_device, torch.float32)
        self.ptr += cfg.train_batch
        return x, y


if __name__ == "__main__":
    # End-to-end (CPU): synthetic PGN -> random 5M model -> capture -> buffer -> minibatch.
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    from maia3.models import MAIA3Model
    from capture import ActivationCapturer, build_cfg
    from data import DataConfig, PositionStream, make_synthetic_pgn

    torch.manual_seed(0)
    tmp = Path(tempfile.mkdtemp()) / "synthetic.pgn"
    make_synthetic_pgn(tmp, n_games=60)

    mcfg = build_cfg("5m", device="cpu")
    model = MAIA3Model(mcfg).eval()
    cap = ActivationCapturer(model, mcfg)
    stream = PositionStream([tmp], DataConfig(max_positions_per_game=12))

    bcfg = BufferConfig(buffer_size=20_000, capture_batch=64, train_batch=4096,
                        pool_device="cpu", compute_device="cpu")
    buf = ActivationBuffer(stream, cap, bcfg)
    x, y = buf.next()
    print(f"minibatch x{tuple(x.shape)} y{tuple(y.shape)} dtype={x.dtype} epochs={buf.epochs}")
    assert x.shape == (bcfg.train_batch, cap.n_layers, cap.d_model)
    x2, y2 = buf.next()
    print(f"second batch ok, pool={buf.pool_x.shape[0]} ptr={buf.ptr}")
    print("OK")
