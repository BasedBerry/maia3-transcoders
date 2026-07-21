# Maia-3 Cross-Layer Transcoders

Sparse **cross-layer transcoders (CLTs)** for the [Maia-3](https://github.com/CSSLab/maia3) chess
models, plus tooling to train them, verify them, and export interpretable feature dashboards.

A CLT replaces the model's MLP sublayers with a dictionary of sparse, per-square features.
Each layer has an encoder that reads the MLP input and produces a sparse feature vector; each
feature writes into the MLP **output of its own layer and every layer below it**. The result is
a set of human-interpretable features you can inspect and causally intervene on.

Because Maia-3 is a ViT over 64 board-square tokens, features are **per-square** — a feature's
top-activating examples render as *"this feature fires on this square of this board."*

---

## Install

```bash
pip install -r requirements.txt
pip install "maia3 @ git+https://github.com/CSSLab/maia3"   # base model + tokenizer + loader
```

CUDA GPU recommended for training and the feature sweep (CPU works for small tests).

---

## Train

Point `DATA` at a monthly [lichess](https://database.lichess.org/) PGN (`.pgn` or `.pgn.zst`).
The base Maia-3 checkpoint is downloaded automatically from Hugging Face on first run.

```bash
# 5M model, single GPU
DATA=/path/to/lichess_db_standard_rated_2022-07.pgn bash scripts/train_5m.sh

# 23M model, 2 GPUs (DDP)
DATA=/path/to/lichess_db_standard_rated_2022-07.pgn CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_23m.sh
```

Checkpoints (`clt_stepN.pt`, `clt_final.pt`) land in `--out`. The log prints per-step
reconstruction FVU, active-features-per-token (L0), and dead-feature count.

**The recipe** (see `train.py --help` for all flags):

- **Activation** — a soft-threshold JumpReLU, `a = relu(z − τ)`, with a learnable per-feature
  threshold `τ`. Fully differentiable; keeps the whole dictionary alive.
- **Sparsity** — a decoder-norm-weighted `tanh` penalty, `λ · Σ tanh(c·‖W_dec,f‖·a_f)`, with a
  **free** (unconstrained, only penalized) decoder norm.
- **λ schedule** — warm up `λ` from 0, hold, then anneal it down once reconstruction converges
  (`--lambda-anneal-*`). This removes late-training pressure that would otherwise consolidate
  activity into a few features.
- **Activations are generated on the fly** through a shuffle buffer (no activations are staged
  to disk), and Elo is randomly sampled from a wide band per position so Elo-gated features are
  exercised across skill levels.

Training is data-cheap: ~80–160M square-tokens (≈10–20k steps) is enough for these model sizes.

---

## Inspect & verify

**Dead-feature rate** over a large token budget (the honest measure of dictionary utilization):

```bash
python measure_dead.py --model 5m --clt-ckpt runs/clt-5m/clt_final.pt \
  --data /path/to/lichess.pgn --tokens 10000000
```

**Behavioral fidelity** — splice the CLT in for the MLPs and measure how well Maia's move policy
survives (move-policy KL, top-1 agreement, WDL shift), with `oracle` (identity) and `zero`
baselines:

```bash
python eval.py   # see the __main__ / functions; loads a CLT + base model and reports metrics
```

---

## Export feature dashboards

`top_acts.py` scans millions of positions and, for every feature, records its **top-activating
(position, square) examples** and a **causal effect** at each — it ablates the feature (removes
its decoder contribution) and measures the change in the move policy (boost/suppress moves,
logit and probability deltas).

```bash
python top_acts.py --model 5m \
  --clt-ckpt runs/clt-5m/clt_final.pt \
  --data /path/to/lichess.pgn \
  --max-positions 3000000 --top-k 20 \
  --out-json features_5m.json
```

Output JSON schema:

```
feature_sets[] : {name, layer, features[]}
  feature      : {id:"L{layer}F{feat:04d}", positions[], random_positions_7p5[], random_positions_25[]}
    position   : {fen, move, elo_s, elo_o, token:{type,square,token_idx}, score, effects}
      effects  : {a, boost_logit, suppress_logit, boost_prob, suppress_prob}
        each   : {move, delta_logit, delta_prob, base_logit, ablated_logit, base_prob, ablated_prob}
```

---

## Analysis: knight forks

`analysis/knight_forks.py` finds features whose top-activating positions are disproportionately
**knight forks (or one-move threats of forks) of valuable pieces (King, Rook, Queen)** and
cross-checks whether ablating them moves the model's logit on a knight move.

```bash
python -c "import analysis.knight_forks as K; K.analyze('features_5m.json', min_count=8, top_n=25)"
```

Precomputed summaries for the released models are in `results/` (small): the fork-selective
features with example positions and their causal moves.

---

## Trained models & full dashboards

The trained CLT checkpoints and the full feature-dashboard JSONs are large and hosted
separately (see the project's data release). Load a checkpoint with:

```python
import torch
from clt import CLTConfig, CrossLayerTranscoder
ck = torch.load("maia3-5m-clt.pt", map_location="cpu", weights_only=False)
clt = CrossLayerTranscoder(CLTConfig(**ck["clt_config"]))
clt.load_state_dict(ck["state_dict"], strict=False)
```

---

## File map

| File | Purpose |
|---|---|
| `clt.py` | The cross-layer transcoder module (encoders, decoders, activation, loss). |
| `capture.py` | Forward hooks to capture per-layer MLP inputs/outputs of the base model. |
| `data.py` | Stream board positions from lichess PGNs into model inputs. |
| `buffer.py` | On-the-fly activation shuffle buffer. |
| `train.py` | Training driver (single-GPU and DDP). |
| `eval.py` | Behavioral evaluation via CLT-for-MLP splicing. |
| `measure_dead.py` | Per-feature firing-frequency / dead-feature measurement. |
| `top_acts.py` | Feature-dashboard export (top activations + causal effects). |
| `analysis/knight_forks.py` | Knight-fork feature analysis. |
| `scripts/` | Ready-to-run training launchers. |
