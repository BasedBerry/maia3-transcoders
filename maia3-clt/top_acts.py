#!/usr/bin/env python3
"""Feature dashboard export for a Maia-3 cross-layer transcoder.

Pass 1 (scan): stream positions, and for every CLT feature (layer, f) keep the top-K
   activating (position, square) pairs. Deduped by position, best square kept.
Pass 2 (effects): for each of those top positions, ablate the feature -- subtract its
   decoder contribution a_f * W_dec[src->L'][:,f] from the real MLP output (linear2) at
   the feature's square, for every output layer L' >= src -- and measure the change in the
   move policy over LEGAL moves (logit and softmax deltas, boost/suppress moves).

We deliberately do NOT collect sub-maximal / random pools -- only the max activators.

This uses OUR CLT (transcoders/clt.py: soft relu(z-tau), W_dec ParameterList) and captures
MLP inputs/outputs exactly as in training (no per-feature input standardization). Runs on a
single GPU. Output: JSON, feature_sets -> features -> top positions (+ effects).

Example (on the free GPU while the 23M trains on 0,1):
  CUDA_VISIBLE_DEVICES=2 python top_acts.py --model 5m \
    --clt-ckpt results/clt-5m-jr-anneal/clt_step12000.pt \
    --data '/grace/u/geilender/lichess_db_standard_rated_2022-07.pgn' \
    --max-positions 3000000 --top-k 20 --out-json /grace/u/geilender/maia3-clt/features_5m.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import chess
import chess.pgn
import torch

from clt import CLTConfig, CrossLayerTranscoder
from capture import build_cfg

from maia3.dataset import tokenize_board, get_historical_tokens, get_legal_moves_mask
from maia3.utils import get_all_possible_moves, mirror_move


# ----------------------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="5m")
    p.add_argument("--clt-ckpt", required=True)
    p.add_argument("--checkpoint-path", default=None, help="base .pt (else HF download)")
    p.add_argument("--trust-checkpoint", action="store_true")
    p.add_argument("--data", nargs="+", required=True, help="PGN glob(s)")
    p.add_argument("--out-json", required=True)

    p.add_argument("--max-positions", type=int, default=3_000_000)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--layers", type=int, nargs="+", default=None, help="subset of layers (default all)")
    p.add_argument("--min-score", type=float, default=1e-4, help="skip top slots / effects below this")

    p.add_argument("--scan-batch", type=int, default=256, help="positions per scan forward")
    p.add_argument("--effect-batch", type=int, default=128, help="(feature,position) jobs per ablation batch")
    p.add_argument("--history", type=int, default=8)
    p.add_argument("--min-ply", type=int, default=6)
    p.add_argument("--max-pos-per-game", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke-random-init", action="store_true")
    return p.parse_args()


# ------------------------------------------------------------------- load base + CLT
def load_base(args, mcfg):
    from maia3.models import MAIA3Model
    if args.smoke_random_init:
        return MAIA3Model(mcfg).to(mcfg.device).eval()
    from maia3.uci import load_model
    from maia3.model_registry import resolve_checkpoint_path
    mcfg.checkpoint_path = args.checkpoint_path or resolve_checkpoint_path(mcfg.model_spec)
    mcfg.trust_checkpoint = args.trust_checkpoint
    return load_model(mcfg)


def load_clt(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = CLTConfig(**ck["clt_config"])
    clt = CrossLayerTranscoder(cfg).to(device).eval()
    sd = {k: v for k, v in ck["state_dict"].items()
          if k not in ("theta", "theta_initialized", "steps_since_fired")}
    clt.load_state_dict(sd, strict=False)
    for pr in clt.parameters():
        pr.requires_grad_(False)
    return clt, cfg


# --------------------------------------------------------------- MLP-input capture
class MLPInputHooks:
    """Pre-hooks on every block.linear1 -> stash the MLP input x_L for all layers."""
    def __init__(self, base):
        self.layers = base.transformer.layers
        self._in = {}
        self._h = []
        for li, blk in enumerate(self.layers):
            self._h.append(blk.linear1.register_forward_pre_hook(self._mk(li)))

    def _mk(self, li):
        def pre(mod, args):
            self._in[li] = args[0]
        return pre

    def stack(self, n_layers, d):
        # dict -> (B, 64, L, d)
        xs = [self._in[li] for li in range(n_layers)]
        return torch.stack(xs, dim=2)

    def clear(self):
        self._in.clear()

    def remove(self):
        for h in self._h:
            h.remove()


@torch.no_grad()
def encode_all_layers(clt, x_b64ld):
    """x: (B,64,L,d) -> feature acts (B,64,L,F). Uses our CLT encode (relu(z-tau))."""
    B = x_b64ld.shape[0]
    L, d = clt.cfg.n_layers, clt.cfg.d_model
    x = x_b64ld.reshape(B * 64, L, d).float()
    a, _ = clt.encode(x)                      # (B*64, L, F)
    return a.reshape(B, 64, L, clt.cfg.n_features)


# --------------------------------------------------------------------- position gen
def stream_positions(pgn_paths, args):
    """Yield dict per position: tokens, self_elo, oppo_elo (game's real elos), fen, board,
    legal_mask, side. Global gid assigned by the caller."""
    all_moves = get_all_possible_moves()
    mv_dict = {m: i for i, m in enumerate(all_moves)}

    class _C:
        history = args.history
        include_time_info = False
    c = _C()

    for path in pgn_paths:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                try:
                    we = int(game.headers.get("WhiteElo", 1500))
                    be = int(game.headers.get("BlackElo", 1500))
                except ValueError:
                    we = be = 1500
                board = game.board()
                hist = deque(maxlen=args.history)
                hist.append(tokenize_board(board))
                emitted = 0
                for ply, mv in enumerate(game.mainline_moves()):
                    if not board.is_legal(mv):
                        break
                    if ply >= args.min_ply and emitted < args.max_pos_per_game:
                        toks = get_historical_tokens(deque(list(hist), maxlen=args.history), c,
                                                     base=0.0, inc=0.0, clk_left_before=0.0, clk_ponder=0.0)
                        self_elo, oppo_elo = (we, be) if board.turn == chess.WHITE else (be, we)
                        yield {
                            "tokens": toks, "self_elo": self_elo, "oppo_elo": oppo_elo,
                            "fen": board.fen(), "side": board.turn,
                            "legal_mask": get_legal_moves_mask(board, mv_dict),
                        }
                        emitted += 1
                    board.push(mv)
                    hist.append(tokenize_board(board))


def batched(gen, n):
    buf = []
    for x in gen:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


# -------------------------------------------------------------------------- display
def sq_to_abs(sq_rel, side):
    """Model squares are side-to-move perspective (mirrored for black). Return absolute square name."""
    sq = chess.square_mirror(sq_rel) if side == chess.BLACK else sq_rel
    return chess.SQUARE_NAMES[sq]


def move_to_abs(mi, moves, side):
    mv = moves[mi] if 0 <= mi < len(moves) else ""
    if side == chess.BLACK and mv:
        mv = mirror_move(mv)
    return mv


# ------------------------------------------------------------------------------ main
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dev = args.device
    mcfg = build_cfg(args.model, device=dev)
    base = load_base(args, mcfg)
    for pr in base.parameters():
        pr.requires_grad_(False)
    clt, ccfg = load_clt(args.clt_ckpt, dev)
    L, F = ccfg.n_layers, ccfg.n_features
    layers = args.layers if args.layers is not None else list(range(L))
    moves = get_all_possible_moves()
    paths = []
    import glob
    for g in args.data:
        paths.extend(sorted(glob.glob(g)))

    hooks = MLPInputHooks(base)

    # Vectorized running top-K per (layer, feature), kept on GPU: value + global gid + square.
    K = args.top_k
    topv = {l: torch.full((K, F), -1e30, device=dev) for l in layers}
    topg = {l: torch.full((K, F), -1, dtype=torch.long, device=dev) for l in layers}
    tops = {l: torch.zeros((K, F), dtype=torch.long, device=dev) for l in layers}
    # LIGHTWEIGHT per-gid metadata only (no tokens) -> tokens re-fetched by re-streaming in pass 2.
    meta = {}   # gid -> {fen, self_elo, oppo_elo, side, pred_move}

    # ---------------- PASS 1: scan ----------------
    print(f"[pass1] scanning up to {args.max_positions} positions, {len(layers)} layers x {F} feats", flush=True)
    gid = 0
    seen = 0
    with torch.no_grad():
        for batch in batched(stream_positions(paths, args), args.scan_batch):
            if seen >= args.max_positions:
                break
            tokens = torch.stack([b["tokens"] for b in batch]).to(dev)
            self_elos = torch.tensor([b["self_elo"] for b in batch], dtype=torch.long, device=dev)
            oppo_elos = torch.tensor([b["oppo_elo"] for b in batch], dtype=torch.long, device=dev)
            B = tokens.shape[0]

            hooks.clear()
            logits_move, _, _ = base(tokens, self_elos, oppo_elos)
            x = hooks.stack(L, ccfg.d_model)          # (B,64,L,d)
            a = encode_all_layers(clt, x)             # (B,64,L,F)

            # model's top predicted (legal) move per position -> stored for the "move" field
            legal_b = torch.stack([b["legal_mask"] for b in batch]).to(dev)
            pred = logits_move.masked_fill(~legal_b, float("-inf")).argmax(1).cpu().tolist()

            gid_start = gid
            for i, b in enumerate(batch):
                meta[gid_start + i] = {"fen": b["fen"], "self_elo": b["self_elo"],
                                       "oppo_elo": b["oppo_elo"], "side": b["side"],
                                       "pred_move": int(pred[i])}
            gid += B
            seen += B

            # vectorized running top-K merge (GPU): keep top-K (value, gid, square) per feature
            for l in layers:
                al = a[:, :, l, :].reshape(B * 64, F)                     # (B*64, F)
                bval, bidx = torch.topk(al, min(K, al.shape[0]), dim=0)   # (k_eff, F)
                bgid = gid_start + (bidx // 64)
                bsq = bidx % 64
                nv, nidx = torch.topk(torch.cat([topv[l], bval], 0), K, dim=0)
                topv[l] = nv
                topg[l] = torch.gather(torch.cat([topg[l], bgid], 0), 0, nidx)
                tops[l] = torch.gather(torch.cat([tops[l], bsq], 0), 0, nidx)
            if seen % (args.scan_batch * 40) < args.scan_batch:
                print(f"  scanned {seen} positions ({gid} gids)", flush=True)

    hooks.remove()
    # materialize heaps[l][f] = [(score, gid, sq), ...], deduped by gid, sorted desc
    heaps = {l: [[] for _ in range(F)] for l in layers}
    for l in layers:
        tv = topv[l].t().cpu().tolist(); tg = topg[l].t().cpu().tolist(); ts = tops[l].t().cpu().tolist()
        for f in range(F):
            best = {}
            for r in range(K):
                sc, g, sq = tv[f][r], tg[f][r], ts[f][r]
                if g < 0 or sc < args.min_score:
                    continue
                if g not in best or sc > best[g][0]:
                    best[g] = (sc, g, sq)
            heaps[l][f] = sorted(best.values(), key=lambda t: -t[0])

    # ---------------- PASS 2: effects ----------------
    # jobs grouped by source layer; each job = (layer, feature, gid, sq_rel)
    jobs_by_L = {l: [] for l in layers}
    for l in layers:
        for f in range(F):
            for sc, g, sq in heaps[l][f]:
                if sc >= args.min_score:
                    jobs_by_L[l].append((f, g, sq))
    n_jobs = sum(len(v) for v in jobs_by_L.values())
    print(f"[pass2] ablation effects for {n_jobs} (feature,position) jobs", flush=True)

    # Re-stream the data to fetch tokens + legal mask for just the TARGET positions (streaming
    # is deterministic, so gid N here == gid N in pass 1). Avoids holding tokens for all in RAM.
    target_gids = {g for l in layers for (_, g, _) in jobs_by_L[l]}
    tokens_by_gid = {}; legal_by_gid = {}
    if target_gids:
        g2 = 0
        for b in stream_positions(paths, args):
            if g2 >= gid:
                break
            if g2 in target_gids:
                tokens_by_gid[g2] = b["tokens"]
                legal_by_gid[g2] = b["legal_mask"]
                if len(tokens_by_gid) >= len(target_gids):
                    break
            g2 += 1
    print(f"[pass2] collected {len(tokens_by_gid)}/{len(target_gids)} target positions", flush=True)

    effects = {}   # (l,f,gid) -> effect dict
    lin2 = base.transformer.layers
    ablate_delta = {}   # layer_idx -> (B,64,d) tensor to add on linear2 output

    def ablate_hook(li):
        def hook(mod, args_, out):
            d = ablate_delta.get(li)
            if d is not None and d.shape == out.shape:
                return out + d
            return out
        return hook

    hook2 = MLPInputHooks(base)
    ab_handles = [lin2[li].linear2.register_forward_hook(ablate_hook(li)) for li in range(L)]
    d_model = ccfg.d_model

    with torch.no_grad():
        for src in layers:
            jlist = [(f, g, sq) for (f, g, sq) in jobs_by_L[src] if g in tokens_by_gid]
            for k0 in range(0, len(jlist), args.effect_batch):
                chunk = jlist[k0:k0 + args.effect_batch]
                Bc = len(chunk)
                gids_c = [g for (_, g, _) in chunk]
                feat_c = torch.tensor([f for (f, _, _) in chunk], device=dev)
                sq_c = torch.tensor([s for (_, _, s) in chunk], device=dev)

                tok = torch.stack([tokens_by_gid[g] for g in gids_c]).to(dev)
                se = torch.tensor([meta[g]["self_elo"] for g in gids_c], dtype=torch.long, device=dev)
                oe = torch.tensor([meta[g]["oppo_elo"] for g in gids_c], dtype=torch.long, device=dev)
                legal = torch.stack([legal_by_gid[g] for g in gids_c]).to(dev)

                # base forward (no ablation), capture MLP inputs
                ablate_delta.clear(); hook2.clear()
                logits_base, _, _ = base(tok, se, oe)
                x = hook2.stack(L, d_model)
                a = encode_all_layers(clt, x)                       # (Bc,64,L,F)
                a_scalar = a[torch.arange(Bc), sq_c, src, feat_c]   # (Bc,) feature act at its square

                # build ablation deltas: subtract feature's decoder contribution at its square,
                # for every output layer L' >= src.  W_dec[src] : (L-src, d, F)
                ablate_delta.clear()
                Wsrc = clt.W_dec[src]                               # (L-src, d, F)
                for o in range(Wsrc.shape[0]):
                    tgt = src + o
                    cols = Wsrc[o][:, feat_c].transpose(0, 1)       # (Bc, d)
                    contrib = a_scalar[:, None] * cols              # (Bc, d)
                    delta = torch.zeros(Bc, 64, d_model, device=dev)
                    delta[torch.arange(Bc), sq_c, :] = -contrib
                    ablate_delta[tgt] = delta

                hook2.clear()
                logits_ab, _, _ = base(tok, se, oe)
                ablate_delta.clear()

                lb = logits_base.masked_fill(~legal, float("-inf")).float()
                la = logits_ab.masked_fill(~legal, float("-inf")).float()
                dlog = lb - la                                     # base - ablated: feature's push
                pb = torch.softmax(lb, dim=1); pa = torch.softmax(la, dim=1)
                dprob = pb - pa
                # boost = move the feature most increases; suppress = most decreases. Two metrics.
                boost_l = dlog.masked_fill(~legal, float("-inf")).argmax(1)
                supp_l = dlog.masked_fill(~legal, float("inf")).argmin(1)
                boost_p = dprob.masked_fill(~legal, float("-inf")).argmax(1)
                supp_p = dprob.masked_fill(~legal, float("inf")).argmin(1)
                for bi in range(Bc):
                    f, g, sq = chunk[bi]
                    side = meta[g]["side"]
                    def pack(mi):
                        mi = int(mi)
                        return {"move": move_to_abs(mi, moves, side),
                                "delta_logit": float(dlog[bi, mi]), "delta_prob": float(dprob[bi, mi]),
                                "base_logit": float(lb[bi, mi]), "ablated_logit": float(la[bi, mi]),
                                "base_prob": float(pb[bi, mi]), "ablated_prob": float(pa[bi, mi])}
                    effects[(src, f, g)] = {
                        "a": float(a_scalar[bi]),
                        "boost_logit": pack(boost_l[bi]), "suppress_logit": pack(supp_l[bi]),
                        "boost_prob": pack(boost_p[bi]), "suppress_prob": pack(supp_p[bi]),
                    }

    for h in ab_handles:
        h.remove()
    hook2.remove()

    # ---------------- build JSON (schema matches the old exporter / frontend) ----------------
    def position_obj(l, f, sc, g, sq):
        m = meta[g]
        side = m["side"]
        sq_abs = chess.square_mirror(sq) if side == chess.BLACK else sq
        obj = {
            "fen": m["fen"],
            "move": move_to_abs(m["pred_move"], moves, side),
            "elo_s": int(m["self_elo"]),
            "elo_o": int(m["oppo_elo"]),
            "token": {"type": "square", "square": chess.SQUARE_NAMES[sq_abs], "token_idx": int(sq_abs)},
            "score": float(sc),
        }
        eff = effects.get((l, f, g))
        if eff is not None:
            obj["effects"] = eff
        return obj

    out = {"feature_sets": []}
    for l in sorted(layers):
        fs = {"name": f"Layer {l}", "layer": int(l), "features": []}
        for f in range(F):
            positions = [position_obj(l, f, sc, g, sq) for sc, g, sq in heaps[l][f]]
            if not positions:
                continue
            fs["features"].append({
                "id": f"L{l}F{f:04d}",
                "positions": positions,
                "random_positions_7p5": [],   # not collected (max activators only)
                "random_positions_25": [],
            })
        out["feature_sets"].append(fs)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(out, fh)
    n_feats = sum(len(fs["features"]) for fs in out["feature_sets"])
    print(f"[done] wrote {args.out_json}: {n_feats} features with >=1 top position, "
          f"{len(effects)} effects computed", flush=True)


if __name__ == "__main__":
    main()
