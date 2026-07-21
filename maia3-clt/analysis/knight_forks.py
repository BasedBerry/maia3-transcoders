#!/usr/bin/env python3
"""Find CLT features whose top-activating positions are disproportionately knight forks
(or one-move threats of forks) of valuable pieces (King, Rook, Queen)."""
import json, sys
import chess

VALUABLE = {chess.KING, chess.ROOK, chess.QUEEN}


def knight_targets(board, ksq, owner_color):
    """# of enemy K/R/Q attacked by the knight on ksq (owner_color's knight)."""
    enemy = not owner_color
    n = 0
    for t in board.attacks(ksq):                     # knight attack set (jumps, no blockers)
        p = board.piece_at(t)
        if p is not None and p.color == enemy and p.piece_type in VALUABLE:
            n += 1
    return n


def has_fork_now(board):
    """A knight of either color forks >=2 valuable enemy pieces on the current board."""
    for color in (chess.WHITE, chess.BLACK):
        for ksq in board.pieces(chess.KNIGHT, color):
            if knight_targets(board, ksq, color) >= 2:
                return True
    return False


def has_fork_threat(board):
    """Side to move has a LEGAL knight move that creates a fork of >=2 valuable pieces."""
    stm = board.turn
    knights = int(board.pieces(chess.KNIGHT, stm))
    if not knights:
        return False
    for mv in board.generate_legal_moves(from_mask=knights):   # only stm knight moves
        board.push(mv)                                          # applies any capture too
        forked = knight_targets(board, mv.to_square, stm)       # knight now at to_square
        board.pop()
        if forked >= 2:
            return True
    return False


def classify(fen):
    """Return (is_fork, kind) where kind in {'now','threat','none'}."""
    b = chess.Board(fen)
    if has_fork_now(b):
        return True, "now"
    if has_fork_threat(b):
        return True, "threat"
    return False, "none"


def fork_detail(fen):
    """Human-readable description of the fork/threat for eyeballing."""
    b = chess.Board(fen)
    for color in (chess.WHITE, chess.BLACK):
        for ksq in b.pieces(chess.KNIGHT, color):
            tg = [t for t in b.attacks(ksq)
                  if (p := b.piece_at(t)) and p.color != color and p.piece_type in VALUABLE]
            if len(tg) >= 2:
                who = "W" if color == chess.WHITE else "B"
                pcs = ",".join(chess.piece_symbol(b.piece_at(t).piece_type).upper() + chess.SQUARE_NAMES[t] for t in tg)
                return f"{who}N{chess.SQUARE_NAMES[ksq]} forks {pcs}"
    stm = b.turn
    for mv in b.generate_legal_moves(from_mask=int(b.pieces(chess.KNIGHT, stm))):
        b.push(mv)
        tg = [t for t in b.attacks(mv.to_square)
              if (p := b.piece_at(t)) and p.color != stm and p.piece_type in VALUABLE]
        b.pop()
        if len(tg) >= 2:
            return f"threat {mv.uci()} -> forks {len(tg)} pieces"
    return "?"


def analyze(path, min_count, top_n):
    d = json.load(open(path))
    cache = {}
    def is_fork(fen):
        r = cache.get(fen)
        if r is None:
            r = classify(fen)
            cache[fen] = r
        return r

    feats = []
    tot = tot_fork = tot_now = tot_threat = 0
    for fs in d["feature_sets"]:
        for ft in fs["features"]:
            pos = ft["positions"]
            n = len(pos)
            if n == 0:
                continue
            kinds = [is_fork(p["fen"]) for p in pos]
            k = sum(1 for f, _ in kinds if f)
            now = sum(1 for f, kind in kinds if kind == "now")
            thr = sum(1 for f, kind in kinds if kind == "threat")
            tot += n; tot_fork += k; tot_now += now; tot_threat += thr
            feats.append((fs["layer"], ft["id"], n, k, now, thr, k / n))

    base = tot_fork / tot
    print(f"\n===== {path.split('/')[-1]} =====")
    print(f"pooled base rate: {tot_fork}/{tot} = {base:.3%} fork positions "
          f"(present {tot_now/tot:.2%}, threat-only {tot_threat/tot:.2%})")

    cands = [x for x in feats if x[3] >= min_count]
    cands.sort(key=lambda x: (-x[6], -x[3]))
    print(f"\nTop {top_n} features by fork-fraction (>= {min_count}/{feats[0][2]} fork positions):")
    print(f"  {'id':>10} {'L':>2} {'n':>3} {'fork':>4} {'now':>3} {'thr':>3} {'frac':>6} {'enrich':>7}")
    for layer, fid, n, k, now, thr, frac in cands[:top_n]:
        print(f"  {fid:>10} {layer:>2} {n:>3} {k:>4} {now:>3} {thr:>3} {frac:>6.2%} {frac/base:>6.1f}x")

    # show fork details for the very top feature
    if cands:
        top = cands[0]
        ft = next(ft for fs in d["feature_sets"] if fs["layer"] == top[0]
                  for ft in fs["features"] if ft["id"] == top[1])
        print(f"\n  examples for {top[1]} (fork {top[3]}/{top[2]}):")
        shown = 0
        for p in ft["positions"]:
            f, kind = is_fork(p["fen"])
            if f:
                print(f"    [{kind:6}] act={p['score']:.2f} sq={p['token']['square']} | {fork_detail(p['fen'])}")
                print(f"             {p['fen']}")
                shown += 1
                if shown >= 4:
                    break
    return base, cands


if __name__ == "__main__":
    for path in sys.argv[1:]:
        analyze(path, min_count=int(sys.argv[0] and 8) or 8, top_n=25)
