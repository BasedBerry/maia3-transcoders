"""Cross-layer transcoder (CLT) for Maia 3.

A CLT replaces the per-layer MLP sublayers with a sparse feature dictionary. For an
L-layer model it has:

  * L encoders   W_enc[l] : reads the MLP *input* x_l (post-norm1 residual) -> F features
  * L(L+1)/2 decoders W_dec[l -> L'] for l <= L' : features at layer l write into the
    MLP *output* of every layer L' >= l.

Reconstruction of layer L''s MLP output:
    y_hat[L'] = b_out[L'] + sum_{l <= L'} a_l @ W_dec[l -> L']^T

Sparsity is enforced with **BatchTopK** (Bussmann et al.): across the whole batch we keep
the top (k * N) positive preactivations, pooled over *all layers and features*, so `k` is
the average total L0 per token summed across layers. At eval we switch to a JumpReLU with a
global threshold theta estimated as an EMA of the per-batch cut value.

Dead features are revived with an **AuxK** auxiliary loss (Gao et al. "Scaling and
evaluating SAEs"): the currently-dead features try to explain the leftover residual.

This module is base-model-agnostic: it consumes captured activations of shape (N, L, d) for
the MLP inputs and (N, L, d) for the MLP outputs. See capture.py for how those are produced.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CLTConfig:
    n_layers: int            # L (8 for both Maia 5M and 23M)
    d_model: int             # 256 (5M) / 512 (23M)
    n_features: int          # F per layer (dictionary width)
    activation: str = "batchtopk"   # "batchtopk" | "jumprelu"
    k: int = 32              # BatchTopK target: avg TOTAL L0 per token across all layers
    per_layer_topk: bool = False  # if True, give each layer its own k/L budget (no cross-layer starvation)
    k_aux: int = 512         # AuxK: how many dead features try to fit the residual
    aux_alpha: float = 1.0 / 32  # AuxK loss weight
    dead_steps_threshold: int = 1000  # a feature is "dead" if unfired for this many steps
    theta_ema: float = 0.999  # EMA decay for the eval threshold (BatchTopK only)
    # --- JumpReLU as a SOFT threshold: a = relu(z - tau), tau learnable per feature ---
    # (This is the form that trained a Maia3 transcoder WITHOUT mass feature death. It is
    # fully differentiable -- no straight-through estimator, no absorbing-state freeze --
    # unlike the hard z*H(z-theta) gate.)
    tanh_c: float = 1.0                   # tanh sparsity-penalty scale c (Anthropic recipe)
    tau_init: float = 0.0                 # per-feature soft-threshold init (0 -> starts as ReLU)


class CrossLayerTranscoder(nn.Module):
    def __init__(self, cfg: CLTConfig):
        super().__init__()
        self.cfg = cfg
        L, d, Fdim = cfg.n_layers, cfg.d_model, cfg.n_features

        # --- Encoders: one (F, d) matrix per layer, stacked into (L, F, d) ---
        self.W_enc = nn.Parameter(torch.empty(L, Fdim, d))
        self.b_enc = nn.Parameter(torch.zeros(L, Fdim))
        # Per-layer input mean subtracted before encoding (SAE "pre-bias").
        self.b_pre = nn.Parameter(torch.zeros(L, d))
        # Per-output-layer reconstruction bias.
        self.b_out = nn.Parameter(torch.zeros(L, d))

        # --- Decoders: for encoder layer l, a (L-l, d, F) tensor writing into
        #     output layers l..L-1. Stored as a ParameterList (ragged over l). ---
        self.W_dec = nn.ParameterList(
            [nn.Parameter(torch.empty(L - l, d, Fdim)) for l in range(L)]
        )

        # JumpReLU soft threshold: a = relu(z - tau), tau learnable per feature (init 0 -> starts
        # as plain ReLU, sparsity penalty raises tau). Fully differentiable, no STE / absorbing state.
        self.tau = nn.Parameter(torch.full((L, Fdim), float(cfg.tau_init)))

        self._init_weights()

        # Dead-feature bookkeeping: steps since each (layer, feature) last fired.
        self.register_buffer("steps_since_fired", torch.zeros(L, Fdim, dtype=torch.long))
        # Per-layer eval threshold for BatchTopK (EMA of the per-batch cut).
        self.register_buffer("theta", torch.zeros(L))
        self.register_buffer("theta_initialized", torch.zeros((), dtype=torch.bool))

    def _init_weights(self):
        L = self.cfg.n_layers
        # Decoders: Kaiming init then unit-norm columns (per feature, per out-layer).
        for l in range(L):
            nn.init.kaiming_uniform_(self.W_dec[l], a=5 ** 0.5)
            with torch.no_grad():
                w = self.W_dec[l]
                w /= (w.norm(dim=1, keepdim=True) + 1e-8)
        # Tied init: encoder for layer l = transpose of its self-decoder (l -> l).
        # (Standard practice in Gemma Scope / sparsify; W_enc left untied afterwards.)
        with torch.no_grad():
            for l in range(L):
                self.W_enc[l].copy_(self.W_dec[l][0].t())   # (F,d) <- (d,F)^T

    @torch.no_grad()
    def normalize_decoder(self):
        """Renormalize decoder columns to unit norm. Call after every optimizer step.

        Prevents feature scales from drifting / collapsing (the missing ingredient that
        let features die past AuxK's reach). Used by Gemma Scope, sparsify, etc."""
        for l in range(self.cfg.n_layers):
            w = self.W_dec[l]
            w.div_(w.norm(dim=1, keepdim=True) + 1e-8)

    @torch.no_grad()
    def init_pre_bias(self, x_mean: torch.Tensor):
        """Set the pre-encoder bias to the data mean. x_mean: (L, d)."""
        self.b_pre.copy_(x_mean)

    def decoder_norms(self) -> torch.Tensor:
        """Per (layer l, feature) decoder power = MEAN over output layers of the L2 norm of
        that feature's decoder vector. Weights the tanh sparsity penalty (Anthropic). Grads
        flow to W_dec, so the penalty also discourages large decoders."""
        norms = [self.W_dec[l].norm(dim=1).mean(dim=0) for l in range(self.cfg.n_layers)]  # each (F,)
        return torch.stack(norms)                                                          # (L, F)

    # ------------------------------------------------------------------ encode
    def preactivations(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, L, d) MLP inputs -> preacts (N, L, F)."""
        # (N,L,d) - (L,d) -> center; einsum with (L,F,d) -> (N,L,F)
        xc = x - self.b_pre  # broadcast over N
        z = torch.einsum("nld,lfd->nlf", xc, self.W_enc) + self.b_enc
        return z

    def _topk_mask(self, relu_pre: torch.Tensor):
        """relu_pre: (N, L, F) nonneg. Returns (mask (N,L,F) bool, cut (L,)).

        Global BatchTopK keeps the top (k*N) activations pooled over all layers -> a single
        cut, so `k` is the avg total L0 per token. Per-layer gives each layer its own
        (k/L)*N budget -> a per-layer cut, preventing layers from starving each other."""
        N, L, Fdim = relu_pre.shape
        if self.cfg.per_layer_topk:
            k_l = max(1, self.cfg.k // L)
            kk = min(k_l * N, N * Fdim)
            per = relu_pre.permute(1, 0, 2).reshape(L, N * Fdim)   # (L, N*F)
            topvals, _ = torch.topk(per, kk, dim=1, sorted=False)  # (L, kk)
            cut = topvals.min(dim=1).values                        # (L,)
            mask = relu_pre >= cut.view(1, L, 1)
            return mask, cut.detach()
        # global
        kk = min(self.cfg.k * N, relu_pre.numel())
        if kk == 0:
            return torch.zeros_like(relu_pre, dtype=torch.bool), relu_pre.new_zeros(L)
        topvals, _ = torch.topk(relu_pre.reshape(-1), kk, sorted=False)
        cut = topvals.min()
        mask = relu_pre >= cut
        return mask, cut.detach().expand(L)                        # broadcast scalar cut -> (L,)

    def encode(self, x: torch.Tensor):
        """Returns (a, info). a: (N, L, F) sparse feature acts.

        info has: preact, relu_pre, mask (bool active), and (jumprelu) active_ste — the
        STE-differentiable H(z-theta) used for the L0 penalty."""
        z = self.preactivations(x)
        if self.cfg.activation == "jumprelu":
            return self._encode_jumprelu(z)
        return self._encode_batchtopk(z)

    def _encode_jumprelu(self, z):
        a = F.relu(z - self.tau)                                   # soft threshold, differentiable
        mask = a > 0                                               # bool, bookkeeping
        return a, {"preact": z, "relu_pre": F.relu(z), "mask": mask}

    def _encode_batchtopk(self, z):
        relu_pre = F.relu(z)
        L = relu_pre.shape[1]
        if self.training:
            mask, cut = self._topk_mask(relu_pre)                  # cut: (L,)
            a = relu_pre * mask
            # Running estimate of the per-layer cut; authoritative eval threshold is set by
            # calibrate_theta() from current weights (the cut drifts up as weights grow).
            with torch.no_grad():
                if not bool(self.theta_initialized):
                    self.theta.copy_(cut)
                    self.theta_initialized.fill_(True)
                else:
                    self.theta.mul_(self.cfg.theta_ema).add_(cut * (1 - self.cfg.theta_ema))
        else:
            mask = relu_pre >= self.theta.view(1, L, 1)             # per-layer JumpReLU at calib thresh
            a = relu_pre * mask
        return a, {"preact": z, "relu_pre": relu_pre, "mask": mask}

    @torch.no_grad()
    def calibrate_theta(self, x_batches) -> float:
        """Set the per-layer eval thresholds from the CURRENT weights.

        Averages the per-batch cut (per layer) over a handful of batches. Call after
        training (or periodically) before switching to eval(). `x_batches` is an iterable
        of (N, L, d) MLP-input tensors. Returns the mean theta (scalar, for logging)."""
        cuts = []
        for x in x_batches:
            relu_pre = F.relu(self.preactivations(x))
            _, cut = self._topk_mask(relu_pre)                     # (L,)
            cuts.append(cut)
        theta = torch.stack(cuts).mean(dim=0)                      # (L,)
        self.theta.copy_(theta)
        self.theta_initialized.fill_(True)
        return float(theta.mean())

    # ------------------------------------------------------------------ decode
    def decode(self, a: torch.Tensor) -> torch.Tensor:
        """a: (N, L, F) -> y_hat: (N, L, d). Cross-layer: feature at l writes to L'>=l."""
        N, L, _ = a.shape
        y_hat = self.b_out.unsqueeze(0).expand(N, L, self.cfg.d_model).clone()
        for l in range(L):
            # contributions of layer-l features to output layers l..L-1
            # einsum: (N,F) x (L-l, d, F) -> (N, L-l, d)
            contrib = torch.einsum("nf,odf->nod", a[:, l, :], self.W_dec[l])
            y_hat[:, l:, :] = y_hat[:, l:, :] + contrib
        return y_hat

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """x: (N,L,d) MLP inputs, y: (N,L,d) MLP outputs (targets).

        Returns loss COMPONENTS (recon_loss, aux_loss, l0_penalty) so the caller can
        assemble the total loss with an activation-appropriate, possibly annealed weight:
          * batchtopk: loss = recon_loss + aux_alpha * aux_loss     (l0_penalty = 0)
          * jumprelu:  loss = recon_loss + lambda * l0_penalty       (aux_loss = 0)
        A convenience `loss` is also returned for the batchtopk path / smoke tests."""
        a, info = self.encode(x)
        y_hat = self.decode(a)

        # per-layer normalized MSE (normalize by target variance for scale-free loss)
        err = y_hat - y
        mse_per_layer = err.pow(2).mean(dim=(0, 2))                    # (L,)
        var_per_layer = (y - y.mean(dim=0, keepdim=True)).pow(2).mean(dim=(0, 2)) + 1e-8
        fvu_per_layer = mse_per_layer / var_per_layer                  # (L,)
        recon_loss = fvu_per_layer.mean()

        # bookkeeping: which features fired this step (any token)
        fired = info["mask"].any(dim=0)                                # (L, F)
        if self.training:
            with torch.no_grad():
                self.steps_since_fired.add_(1)
                self.steps_since_fired[fired] = 0

        aux_loss = x.new_zeros(())
        tanh_penalty = x.new_zeros(())
        if self.cfg.activation == "jumprelu":
            # Anthropic tanh sparsity penalty, summed over features (mean over tokens): a strong,
            # responsive L0-like lever (~= expected active features). Decoder-norm weighted (norm
            # free), saturating. The soft relu(z-tau) keeps dead=0 even under this strong penalty.
            dec_norm = self.decoder_norms()                           # (L, F)
            arg = self.cfg.tanh_c * dec_norm.unsqueeze(0) * a         # (N, L, F)
            tanh_penalty = torch.tanh(arg).sum(dim=(1, 2)).mean()     # mean over tokens, sum over l,f
        elif self.training and self.cfg.aux_alpha > 0:
            aux_loss = self._aux_loss(info["relu_pre"], err.detach())

        loss = recon_loss + self.cfg.aux_alpha * aux_loss             # batchtopk convenience

        with torch.no_grad():
            l0 = info["mask"].sum().float() / x.shape[0]               # avg total L0 per token
            n_dead = (self.steps_since_fired > self.cfg.dead_steps_threshold).sum()

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "aux_loss": aux_loss,
            "tanh_penalty": tanh_penalty,
            "fvu_per_layer": fvu_per_layer.detach(),
            "l0": l0,
            "n_dead": n_dead,
            "y_hat": y_hat,
        }

    def _aux_loss(self, relu_pre: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Dead features try to reconstruct the leftover residual (detached)."""
        dead = (self.steps_since_fired > self.cfg.dead_steps_threshold)  # (L, F)
        if not bool(dead.any()):
            return relu_pre.new_zeros(())
        # keep only dead features' preacts, take top k_aux across the batch
        dead_pre = relu_pre * dead.unsqueeze(0)                          # (N, L, F)
        N = dead_pre.shape[0]
        kk = min(self.cfg.k_aux * N, int(dead.sum()) * N)
        if kk == 0:
            return relu_pre.new_zeros(())
        flat = dead_pre.reshape(-1)
        kk = min(kk, flat.numel())
        topvals, _ = torch.topk(flat, kk, sorted=False)
        cut = topvals.min()
        a_aux = dead_pre * (dead_pre >= cut)
        y_aux = self.decode(a_aux) - self.b_out.unsqueeze(0)            # aux recon (no bias double-count)
        return (y_aux - residual).pow(2).mean()


if __name__ == "__main__":
    # CPU smoke test: shapes, sparsity, a train step, and an eval step.
    torch.manual_seed(0)
    cfg = CLTConfig(n_layers=8, d_model=256, n_features=4096, k=32, dead_steps_threshold=0)
    clt = CrossLayerTranscoder(cfg)
    N = 512
    x = torch.randn(N, 8, 256)
    y = torch.randn(N, 8, 256) * 0.5

    clt.train()
    opt = torch.optim.AdamW(clt.parameters(), lr=1e-3)
    for step in range(5):
        out = clt(x, y)
        opt.zero_grad(); out["loss"].backward(); opt.step()
    print(f"train loss={out['loss'].item():.4f} recon={out['recon_loss'].item():.4f} "
          f"aux={out['aux_loss'].item():.4f} L0={out['l0'].item():.1f} dead={out['n_dead'].item()}")
    print("fvu/layer:", [round(v, 3) for v in out["fvu_per_layer"].tolist()])
    n_params = sum(p.numel() for p in clt.parameters())
    print(f"CLT params: {n_params/1e6:.1f}M  theta(mean)={clt.theta.mean().item():.4f}")

    clt.eval()
    with torch.no_grad():
        oe = clt(x, y)
    print(f"eval L0={oe['l0'].item():.1f} recon={oe['recon_loss'].item():.4f}")
    assert out["loss"].isfinite(), "non-finite loss"
    print("OK")
