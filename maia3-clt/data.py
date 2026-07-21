"""Stream board positions from lichess PGNs into Maia-3 model inputs.

Yields (tokens, self_elo, oppo_elo) per sampled position, replaying each game so the
model sees faithful board *history* (matching uci.py's --use_uci_history behaviour:
each historical board is tokenized from its own side-to-move perspective).

Elo conditioning: per the project decision, Elo is **randomly sampled per position** from
a wide band (not the game's real ratings, and not enumerated) so features that are
Elo-gated are exercised across the whole skill range.

Handles plain .pgn and zstd-compressed .pgn.zst (lichess format).
"""

from __future__ import annotations

import io
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from maia3.dataset import tokenize_board, get_historical_tokens


@dataclass
class DataConfig:
    history: int = 8
    include_time_info: bool = False
    elo_low: int = 800            # wide band lower bound
    elo_high: int = 2800          # wide band upper bound
    min_ply: int = 6              # skip the first few book moves
    max_positions_per_game: int = 20  # cap + subsample to decorrelate within-game
    seed: int = 0


def _open_pgn(path: Path):
    """Return a text stream for a .pgn or .pgn.zst file."""
    if path.suffix == ".zst":
        import zstandard  # cluster env; not needed locally for plain .pgn
        dctx = zstandard.ZstdDecompressor()
        binary = dctx.stream_reader(path.open("rb"))
        return io.TextIOWrapper(binary, encoding="utf-8", errors="ignore")
    return path.open("r", encoding="utf-8", errors="ignore")


class PositionStream:
    """Iterable over (tokens, self_elo, oppo_elo). Elos sampled per position.

    `rank`/`world_size` shard games round-robin so DDP ranks see disjoint games.
    A private torch.Generator makes Elo sampling reproducible and rank-distinct.
    """

    def __init__(self, pgn_paths, cfg: DataConfig, rank: int = 0, world_size: int = 1):
        self.paths = [Path(p) for p in pgn_paths]
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        # SimpleNamespace-like view for get_historical_tokens (needs .history, .include_time_info)
        self._tok_cfg = cfg

    def _sample_elos(self, gen):
        lo, hi = self.cfg.elo_low, self.cfg.elo_high
        self_elo = int(torch.randint(lo, hi + 1, (1,), generator=gen).item())
        oppo_elo = int(torch.randint(lo, hi + 1, (1,), generator=gen).item())
        return self_elo, oppo_elo

    def __iter__(self):
        cfg = self.cfg
        gen = torch.Generator().manual_seed(cfg.seed * 100003 + self.rank)
        game_idx = -1
        for path in self.paths:
            stream = _open_pgn(path)
            while True:
                game = chess.pgn.read_game(stream)
                if game is None:
                    break
                game_idx += 1
                if game_idx % self.world_size != self.rank:
                    continue  # not this rank's shard

                board = game.board()
                history = deque(maxlen=cfg.history)
                history.append(tokenize_board(board))

                # collect eligible plies first, then subsample for within-game decorrelation
                positions = []
                ply = 0
                for move in game.mainline_moves():
                    if not board.is_legal(move):
                        break
                    if ply >= cfg.min_ply:
                        positions.append(_history_snapshot(history))
                    board.push(move)
                    history.append(tokenize_board(board))
                    ply += 1
                if not positions:
                    continue

                # subsample up to max_positions_per_game
                if len(positions) > cfg.max_positions_per_game:
                    idx = torch.randperm(len(positions), generator=gen)[: cfg.max_positions_per_game]
                    positions = [positions[i] for i in idx.tolist()]

                for hist in positions:
                    tokens = get_historical_tokens(
                        hist, cfg, base=0.0, inc=0.0, clk_left_before=0.0, clk_ponder=0.0
                    )
                    self_elo, oppo_elo = self._sample_elos(gen)
                    yield tokens, self_elo, oppo_elo


def _history_snapshot(history: deque):
    """Copy the current history deque so later pushes don't mutate captured positions."""
    return deque(list(history), maxlen=history.maxlen)


def make_synthetic_pgn(path, n_games=20, seed=0):
    """Write a small random-move PGN for offline testing (no real data needed)."""
    import chess.pgn
    g = torch.Generator().manual_seed(seed)
    with open(path, "w") as f:
        for gi in range(n_games):
            board = chess.Board()
            game = chess.pgn.Game()
            game.headers["WhiteElo"] = str(int(torch.randint(800, 2800, (1,), generator=g).item()))
            game.headers["BlackElo"] = str(int(torch.randint(800, 2800, (1,), generator=g).item()))
            node = game
            for _ in range(40):
                moves = list(board.legal_moves)
                if not moves or board.is_game_over():
                    break
                mv = moves[int(torch.randint(len(moves), (1,), generator=g).item())]
                board.push(mv)
                node = node.add_variation(mv)
            print(game, file=f, end="\n\n")


if __name__ == "__main__":
    import tempfile
    tmp = Path(tempfile.mkdtemp()) / "synthetic.pgn"
    make_synthetic_pgn(tmp, n_games=10)
    cfg = DataConfig(history=8, max_positions_per_game=8)
    stream = PositionStream([tmp], cfg)
    n = 0
    feat_dim = None
    for tokens, se, oe in stream:
        assert tokens.shape[0] == 64
        feat_dim = tokens.shape[1]
        assert cfg.elo_low <= se <= cfg.elo_high and cfg.elo_low <= oe <= cfg.elo_high
        n += 1
    print(f"streamed {n} positions, token feat_dim={feat_dim} (expect 12*8+1=97)")
    # DDP sharding: rank 0 and rank 1 of world_size 2 must be disjoint and cover all
    s0 = sum(1 for _ in PositionStream([tmp], cfg, rank=0, world_size=2))
    s1 = sum(1 for _ in PositionStream([tmp], cfg, rank=1, world_size=2))
    print(f"shard rank0={s0} rank1={s1} sum={s0+s1} (full={n})")
    assert s0 + s1 == n, "DDP game sharding lost or duplicated positions"
    print("OK")
