"""Capture Maia 3 MLP activations for cross-layer transcoder training.

For every EncoderOnlyBlock we grab:
  * MLP input  x_L : the input to `linear1` (the post-norm1 residual)  -> forward_pre_hook
  * MLP output y_L : the output of `linear2` (the MLP's residual contribution) -> forward_hook

Both are exact: `dropout` between GELU and linear2 is identity in eval, so y_L is precisely
what the CLT must reconstruct, and x_L is precisely what its encoders read.

The base model is tiny, so we generate activations on the fly (no disk staging). One position
yields 64 square-token samples; each sample carries all L layers' (x, y), which is the unit a
cross-layer transcoder needs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from maia3.models import MAIA3Model
from maia3.model_registry import resolve_model_spec, BASE_SIZE_CONFIG


def build_cfg(model_alias: str, device: str = "cuda") -> SimpleNamespace:
    """Build a MAIA3Model config namespace from a built-in alias (e.g. '5m', '23m')."""
    spec = resolve_model_spec(model_alias)
    cfg = SimpleNamespace(**{**BASE_SIZE_CONFIG, **spec.config})
    cfg.device = device
    cfg.model_spec = spec
    return cfg


class ActivationCapturer:
    """Wraps a frozen MAIA3Model and captures per-layer MLP in/out activations."""

    def __init__(self, model: MAIA3Model, cfg):
        self.model = model.eval()
        self.cfg = cfg
        self.n_layers = cfg.num_blocks
        self.d_model = cfg.dim_vit
        self._in: dict[int, torch.Tensor] = {}
        self._out: dict[int, torch.Tensor] = {}
        self._handles = []
        self._register()

    def _register(self):
        for li, blk in enumerate(self.model.transformer.layers):
            def pre_hook(mod, args, _li=li):
                self._in[_li] = args[0]
            def post_hook(mod, args, output, _li=li):
                self._out[_li] = output
            self._handles.append(blk.linear1.register_forward_pre_hook(pre_hook))
            self._handles.append(blk.linear2.register_forward_hook(post_hook))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @torch.no_grad()
    def capture(self, tokens, self_elos, oppo_elos):
        """Run the model and return (mlp_in, mlp_out), each (B*64, L, d) float32.

        tokens: (B, 64, feat) ; self_elos/oppo_elos: (B,) long."""
        self._in.clear(); self._out.clear()
        self.model(tokens, self_elos, oppo_elos)
        B = tokens.shape[0]
        # stack layers -> (B, 64, L, d) -> (B*64, L, d)
        x = torch.stack([self._in[li] for li in range(self.n_layers)], dim=2)
        y = torch.stack([self._out[li] for li in range(self.n_layers)], dim=2)
        x = x.reshape(B * 64, self.n_layers, self.d_model).float()
        y = y.reshape(B * 64, self.n_layers, self.d_model).float()
        return x, y


if __name__ == "__main__":
    # Smoke test: build a random-weight 5M model, capture activations, check shapes and
    # that captured y actually equals the model's real linear2 output.
    torch.manual_seed(0)
    cfg = build_cfg("5m", device="cpu")
    model = MAIA3Model(cfg).to("cpu").eval()
    cap = ActivationCapturer(model, cfg)

    B = 4
    feat = 12 * cfg.history + 1  # matches get_historical_tokens (include_time_info=False)
    tokens = torch.randn(B, 64, feat)
    self_elos = torch.randint(800, 2800, (B,))
    oppo_elos = torch.randint(800, 2800, (B,))

    x, y = cap.capture(tokens, self_elos, oppo_elos)
    print(f"mlp_in  {tuple(x.shape)}  mlp_out {tuple(y.shape)}")
    assert x.shape == (B * 64, cfg.num_blocks, cfg.dim_vit)
    assert y.shape == (B * 64, cfg.num_blocks, cfg.dim_vit)

    # Verify layer-0 captured output == recomputing linear2(gelu(linear1(x0))) directly.
    blk0 = model.transformer.layers[0]
    x0 = cap._in[0]
    y0_ref = blk0.linear2(blk0.dropout(blk0.activation(blk0.linear1(x0))))
    y0_cap = cap._out[0]
    err = (y0_ref - y0_cap).abs().max().item()
    print(f"layer-0 capture max-abs error vs recompute: {err:.2e}")
    assert err < 1e-5, "captured MLP output does not match recomputation"
    print(f"in scale (rms): {x.pow(2).mean().sqrt():.3f}  out scale (rms): {y.pow(2).mean().sqrt():.3f}")
    print("OK")
