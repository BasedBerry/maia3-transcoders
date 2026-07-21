"""Behavioral evaluation: splice the CLT in for Maia's MLPs and measure policy fidelity.

The transcoder is only useful if the *replacement model* (MLPs swapped for CLT
reconstructions) still plays like Maia. We measure, over held-out positions:
  * move-policy KL(clean || spliced) over legal moves
  * top-1 move agreement
  * WDL L1 shift (value head)

Modes:
  * 'clt'    : replace each layer's MLP output with the CLT cross-layer reconstruction
  * 'oracle' : replace with the TRUE captured MLP output (identity) -> must give KL~0
               (validates the splice machinery itself)
  * 'zero'   : ablate the MLP output to 0 (reference: how much the MLP mattered)

The CLT reconstruction is computed online during the forward: a pre-hook on each linear1
encodes x_L into features (eval JumpReLU at the calibrated theta); a post-hook on each
linear2 returns y_hat_L = b_out_L + sum_{l<=L} decode_{l->L}(a_l), using features from all
layers seen so far this forward.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import chess
import chess.pgn
import torch
import torch.nn.functional as F

from maia3.dataset import tokenize_board, get_historical_tokens, get_legal_moves_mask
from maia3.utils import get_all_possible_moves


class ReplacementModel:
    """Runs Maia with each block's MLP output replaced according to `mode`."""

    def __init__(self, base, clt, mode: str = "clt"):
        self.base = base.eval()
        self.clt = clt.eval() if clt is not None else None
        self.mode = mode
        self.n_layers = len(base.transformer.layers)
        self.d = base.transformer.layers[0].linear1.in_features
        self._a = {}       # layer -> feature acts (N_tok, F)
        self._handles = []

    def _encode_layer(self, l, x_flat):
        c = self.clt
        z = (x_flat - c.b_pre[l]) @ c.W_enc[l].t() + c.b_enc[l]
        if c.cfg.activation == "jumprelu":
            return F.relu(z - c.tau[l])                 # soft threshold (relu(z - tau))
        relu = F.relu(z)
        return relu * (relu >= c.theta[l])              # BatchTopK: per-layer calibrated threshold

    def _recon_layer(self, l, N):
        c = self.clt
        y = c.b_out[l].unsqueeze(0).expand(N, self.d).clone()
        for src in range(l + 1):
            # decoder src -> l : W_dec[src][l-src] is (d, F)
            y = y + self._a[src] @ c.W_dec[src][l - src].t()
        return y

    def __enter__(self):
        for li, blk in enumerate(self.base.transformer.layers):
            if self.mode == "clt":
                def pre(mod, args, _li=li):
                    x = args[0]
                    N = x.shape[0] * x.shape[1]
                    self._a[_li] = self._encode_layer(_li, x.reshape(N, self.d))
                def post(mod, args, output, _li=li):
                    N, S, d = output.shape
                    return self._recon_layer(_li, N * S).reshape(N, S, d).to(output.dtype)
                self._handles.append(blk.linear1.register_forward_pre_hook(pre))
                self._handles.append(blk.linear2.register_forward_hook(post))
            elif self.mode == "zero":
                def post_zero(mod, args, output):
                    return torch.zeros_like(output)
                self._handles.append(blk.linear2.register_forward_hook(post_zero))
            elif self.mode == "oracle":
                pass  # identity: leave the model untouched
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._a.clear()


@torch.no_grad()
def _policy_logits(base, tokens, self_elos, oppo_elos):
    logits_move, logits_value, _ = base(tokens, self_elos, oppo_elos)
    return logits_move, logits_value


def load_positions(pgn_path, n=256, history=8, min_ply=6, elo=1500):
    """Return list of (tokens, self_elo, oppo_elo, legal_mask). Fixed elo for eval determinism."""
    all_moves = get_all_possible_moves()
    mv_dict = {m: i for i, m in enumerate(all_moves)}

    class _C:  # minimal cfg for get_historical_tokens
        pass
    c = _C(); c.history = history; c.include_time_info = False

    out = []
    with open(pgn_path) as f:
        while len(out) < n:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            hist = deque(maxlen=history); hist.append(tokenize_board(board))
            ply = 0
            for mv in game.mainline_moves():
                if not board.is_legal(mv):
                    break
                if ply >= min_ply and len(out) < n:
                    toks = get_historical_tokens(deque(list(hist), maxlen=history), c,
                                                 base=0.0, inc=0.0, clk_left_before=0.0, clk_ponder=0.0)
                    mask = get_legal_moves_mask(board, mv_dict)
                    out.append((toks, elo, elo, mask))
                board.push(mv); hist.append(tokenize_board(board)); ply += 1
    return out


@torch.no_grad()
def evaluate(base, clt, positions, device="cpu"):
    tokens = torch.stack([p[0] for p in positions]).to(device)
    self_elos = torch.tensor([p[1] for p in positions], device=device)
    oppo_elos = torch.tensor([p[2] for p in positions], device=device)
    masks = torch.stack([p[3] for p in positions]).to(device)  # (B, n_moves) bool

    # pad legal mask to logit width (move logits are 4352 = 4096 + 256 promo)
    def masked_logprobs(logits):
        m = torch.zeros_like(logits, dtype=torch.bool)
        m[:, : masks.shape[1]] = masks
        logits = logits.masked_fill(~m, float("-inf"))
        return F.log_softmax(logits, dim=-1)

    clean_move, clean_val = _policy_logits(base, tokens, self_elos, oppo_elos)
    clean_lp = masked_logprobs(clean_move.float())
    clean_wdl = F.softmax(clean_val.float(), dim=-1)

    results = {}
    for mode in ["oracle", "zero", "clt"]:
        if mode == "clt" and clt is None:
            continue
        with ReplacementModel(base, clt, mode=mode):
            sp_move, sp_val = _policy_logits(base, tokens, self_elos, oppo_elos)
        sp_lp = masked_logprobs(sp_move.float())
        # KL(clean || spliced) over legal moves
        kl = (clean_lp.exp() * (clean_lp - sp_lp)).nansum(dim=-1).mean().item()
        top1 = (clean_lp.argmax(-1) == sp_lp.argmax(-1)).float().mean().item()
        wdl_l1 = (F.softmax(sp_val.float(), -1) - clean_wdl).abs().sum(-1).mean().item()
        results[mode] = {"kl": kl, "top1_agree": top1, "wdl_l1": wdl_l1}
    return results


if __name__ == "__main__":
    # Correctness test on random-weight 5M: oracle must give KL~0; zero should differ.
    import tempfile
    from types import SimpleNamespace
    from maia3.models import MAIA3Model
    from capture import build_cfg
    from clt import CLTConfig, CrossLayerTranscoder
    from data import make_synthetic_pgn

    torch.manual_seed(0)
    cfg = build_cfg("5m", device="cpu")
    base = MAIA3Model(cfg).eval()
    clt = CrossLayerTranscoder(CLTConfig(n_layers=8, d_model=256, n_features=1024, k=32))
    clt.theta.fill_(0.1); clt.theta_initialized.fill_(True)

    tmp = Path(tempfile.mkdtemp()) / "syn.pgn"
    make_synthetic_pgn(tmp, n_games=40)
    positions = load_positions(tmp, n=128)
    print(f"loaded {len(positions)} eval positions")
    res = evaluate(base, clt, positions)
    for mode, m in res.items():
        print(f"  {mode:7s} KL {m['kl']:.4f}  top1 {m['top1_agree']:.3f}  wdl_l1 {m['wdl_l1']:.4f}")
    assert res["oracle"]["kl"] < 1e-4, "oracle splice not identity — splice machinery broken"
    assert res["oracle"]["top1_agree"] > 0.999
    print("OK (oracle is faithful identity)")
