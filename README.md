# GomokuZero

AlphaZero-style Gomoku (Five In A Row) engine using deep reinforcement learning with Monte Carlo Tree Search (MCTS) and a residual CNN.

## Features

- **AlphaZero architecture**: Combines deep neural networks with MCTS for superhuman play
- **Batched MCTS**: Virtual-loss parallel tree search with GPU-accelerated batch evaluation
- **Cython acceleration**: Optional C-level speedups for critical MCTS paths
- **Threat-plane backend fallback**: Cython (`mcts_accel`) -> Numba JIT -> NumPy fallback
- **Self-play training**: Generates training data through competitive self-play
- **Inline evaluation**: Automatic checkpoint comparison during training
- **Interactive play**: Terminal UI (`play.py`) and Qt GUI (`play_qt.py`)

## Requirements

- Python 3.8+
- TensorFlow 2.x (GPU recommended)
- NumPy
- Cython (optional, preferred for acceleration)
- Numba (optional fallback for threat planes when Cython is unavailable)
- PyQt6 (optional, for graphical play via `play_qt.py`)

## Installation

```bash
# Clone or navigate to the project directory
cd GomokuZero

# Create virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install tensorflow numpy cython numba pyqt6

# Build Cython acceleration (optional but recommended)
python setup_accel.py build_ext --inplace
```

## Usage

### Training

Train the network via self-play:

```bash
python train.py
```

`train.py` resumes automatically if `weights/gomoku_weights.weights.h5` exists.
To start fresh, delete `weights/` manually before running.

Training parameters can be adjusted in `train.py`. The script will:
- Generate self-play games using batched MCTS
- Train the network on policy and value targets
- Save checkpoints to `weights/`
- Keep plateau/autotune safeguards in-loop (inline eval/promotion disabled by default)
- Hand off checkpoint ranking/promotion to manual tournament runs

Runtime tuning is also available via environment variables (current defaults):
- `GZ_CONCURRENT_GAMES=24`
- `GZ_NUM_GAMES=5000`
- `GZ_BATCH_SIZE=1024`
- `GZ_TRAIN_STEPS_RATIO=4.0`
- `GZ_MCTS_BATCH_SIZE=10` (full-search moves)
- `GZ_MCTS_BATCH_SIZE_FAST=16` (fast/value-only moves)

Example:

```bash
GZ_CONCURRENT_GAMES=28 GZ_MCTS_BATCH_SIZE_FAST=20 python train.py
```

Run the training+tournament cycle until a total game goal:

```bash
./train_tournament_loop.sh --games-goal 50000

# Optional tournament folder and extra tournament args after --
./train_tournament_loop.sh --games-goal 120000 --tournament-dir botb-weights -- --mcmahon-rounds 6
```

`train_tournament_loop.sh` runs `train.py` (`GZ_NUM_GAMES` games/chunk; default 5,000),
then `eval_tournament.py --mode swiss-sprt`, and repeats until
`weights/train_state.pkl` reaches `--games-goal`. Both module calls in the loop
run with `2>/dev/null`.
By default in `swiss-sprt` mode, it also sets `--sprt-max-challengers 4`.
It auto-selects Swiss rounds per cycle by tournament player count:
`<=70 -> 6`, `71-140 -> 7`, `>140 -> 8`.
To override this, pass `--swiss-rounds` after `--`, e.g.
`./train_tournament_loop.sh --games-goal 120000 -- --swiss-rounds 7`.
You can also override mode explicitly, e.g.
`./train_tournament_loop.sh --games-goal 120000 -- --mode swiss-mcmahon`.
To override the challenger cap, pass e.g.
`./train_tournament_loop.sh --games-goal 120000 -- --sprt-max-challengers 8`.
To focus tournament evaluation on only the strongest nets, pass e.g.
`./train_tournament_loop.sh --games-goal 120000 -- --swiss-top-n 12`.

### Evaluation

Evaluate a checkpoint against baselines:

```bash
# Auto-detect latest checkpoint
python eval.py

# Evaluate specific checkpoint
python eval.py --checkpoint weights/gomoku_best.weights.h5

# Custom opening count
python eval.py --openings 200

# Calibrate strength between sim tiers on one checkpoint
python eval.py --checkpoint weights/gomoku_best.weights.h5 --calibrate-sims --sim-levels 100,400,1600

# Run eval without changing persistent ratings
python eval.py --no-rating-update

# Pretty-print a ratings table
python ratings_glicko2.py Mixed-competitor-weights/glicko2_ratings.pkl
python ratings_glicko2.py Mixed-competitor-weights/glicko2_ratings.pkl --min-games 40 --sort games
```

Standard eval runs update persistent Glicko-2 ratings in `weights/glicko2_ratings.pkl`.

### Tournament (`eval_tournament.py`)

Runs Swiss seeding across all checkpoints, then either:
- a McMahon final group selected by rating bar (`--mode swiss-mcmahon`, default), or
- a sequential SPRT challenger ladder over a Swiss shortlist (`--mode swiss-sprt`).

Outputs are written inside the tournament folder by default:
`<tournament-dir>/glicko2_ratings.pkl`, `<tournament-dir>/gomoku_best.weights.h5`,
`<tournament-dir>/best_checkpoint.pkl`.
Discovery automatically excludes `gomoku_best.weights.h5`, `gomoku_weights.weights.h5`,
and `gomoku_*_final.weights.h5`.

```bash
# Basic run (Swiss -> McMahon, default mode)
python eval_tournament.py --tournament-dir botb-weights

# Focus on the 12 strongest nets (Swiss + McMahon + SPRT all capped)
python eval_tournament.py --tournament-dir botb-weights --swiss-top-n 12

# Cap only the McMahon final to 10 players (Swiss still runs over all)
python eval_tournament.py --tournament-dir botb-weights --mcmahon-top-n 10

# Swiss+SPRT mode with top-N focus and challenger cap
python eval_tournament.py --tournament-dir botb-weights \
    --mode swiss-sprt --swiss-top-n 12 --sprt-top-n 8 --sprt-max-challengers 4

# Large field (hundreds of checkpoints), limit to known-strong group
python eval_tournament.py --tournament-dir botb-weights --swiss-top-n 20 --swiss-rounds 6

# Dry run: no ratings written, no best-checkpoint update
python eval_tournament.py --tournament-dir botb-weights --no-persist-ratings --no-promote-winner
```

#### All flags

**General**

| Flag | Default | Description |
|---|---|---|
| `--tournament-dir DIR` | *(required)* | Directory containing checkpoint weight files |
| `--mode MODE` | `swiss-mcmahon` | Tournament mode: `swiss-mcmahon` or `swiss-sprt` |
| `--certainty FLOAT` | `0.95` | SPRT mode only: overall confidence target for strongest-net selection |
| `--seed INT` | *(fixed)* | RNG seed for opening book generation |
| `--plies INT` | *(fixed)* | Random plies per opening position |

**Top-N group size (focus on the strongest nets)**

| Flag | Default | Description |
|---|---|---|
| `--swiss-top-n N` | all | Cap the Swiss field to the top N players by existing Glicko-2 rating. Also sets the default for `--mcmahon-top-n` and `--sprt-top-n` unless those are specified separately. Requires persistent ratings from a prior run to be meaningful. |
| `--mcmahon-top-n N` | all | Cap McMahon finalists to N players. Overrides `--mcmahon-max-players`. Defaults to `--swiss-top-n` if set. |
| `--sprt-top-n N` | all | Cap the SPRT finalist shortlist to N players (swiss-sprt mode). Defaults to `--swiss-top-n` if set. |

**Swiss seeding stage**

| Flag | Default | Description |
|---|---|---|
| `--swiss-rounds N` | `6` | Number of Swiss pairing rounds |
| `--swiss-sims N` | `50` | MCTS simulations per move during Swiss |
| `--swiss-openings N` | `4` | Opening positions per round (×2 = games per pairing) |
| `--swiss-batch-size N` | `16` | MCTS leaf batch size during Swiss |

**McMahon final stage** (`--mode swiss-mcmahon`)

| Flag | Default | Description |
|---|---|---|
| `--mcmahon-rounds N` | `5` | Number of McMahon pairing rounds |
| `--mcmahon-sims N` | `160` | MCTS simulations per move during McMahon |
| `--mcmahon-openings N` | `8` | Opening positions per round (×2 = games per pairing) |
| `--mcmahon-batch-size N` | `32` | MCTS leaf batch size during McMahon |
| `--mcmahon-bar-gap FLOAT` | `80.0` | Include players within this many rating points of the leader in the final group |
| `--mcmahon-min-players N` | `8` | Minimum finalists (expands group if bar-gap yields fewer) |
| `--mcmahon-max-players N` | `24` | Maximum finalists (truncates group if bar-gap yields more) |
| `--shared-gold-margin FLOAT` | `0.02` | Declare shared gold if top two are within this win-probability of 50/50 |
| `--shared-gold-min-games N` | `120` | Minimum head-to-head games before shared gold can be declared |

**SPRT ladder stage** (`--mode swiss-sprt`)

| Flag | Default | Description |
|---|---|---|
| `--sprt-max-challengers N` | `12` | Maximum challengers tested in the SPRT ladder (incumbent + N duels) |
| `--sprt-score0 FLOAT` | `0.50` | H0 expected score (null hypothesis: no improvement) |
| `--sprt-score1 FLOAT` | `0.55` | H1 expected score (alternative: challenger is stronger) |
| `--sprt-min-games N` | `32` | Minimum games before SPRT can reach a decision |
| `--sprt-max-games N` | `320` | Maximum games per SPRT duel |
| `--sprt-openings-step N` | `4` | Openings per SPRT update chunk (×2 = games per chunk) |
| `--sprt-sims N` | *(mcmahon-sims)* | MCTS sims per move for SPRT duels |
| `--sprt-batch-size N` | *(mcmahon-batch-size)* | MCTS batch size for SPRT duels |
| `--sprt-alpha FLOAT` | *(derived)* | Override SPRT Type-I error rate (default: derived from `--certainty`) |
| `--sprt-beta FLOAT` | *(derived)* | Override SPRT Type-II error rate (default: derived from `--certainty`) |

**Ratings and output**

| Flag | Default | Description |
|---|---|---|
| `--persist-ratings` / `--no-persist-ratings` | enabled | Load and save persistent Glicko-2 ratings file |
| `--ratings-file PATH` | `<tournament-dir>/glicko2_ratings.pkl` | Path to persistent ratings file |
| `--promote-winner` / `--no-promote-winner` | enabled | Copy tournament champion to `gomoku_best.weights.h5` |
| `--best-weights-file PATH` | `<tournament-dir>/gomoku_best.weights.h5` | Destination for promoted best weights |
| `--best-state-file PATH` | `<tournament-dir>/best_checkpoint.pkl` | Destination for promoted best-state metadata |

### Playing

Play against the trained AI:

```bash
python play.py

# Pick a difficulty tier
python play.py --difficulty easy    # 250 sims
python play.py --difficulty medium  # 500 sims (default)
python play.py --difficulty hard    # 2000 sims

# Or set a custom sim count directly
python play.py --difficulty 2500    # custom sims ("Custom" in UI)

# Use latest training weights
python play.py --latest

# Use a specific .h5 weights file
python play.py --weight_file weights/gomoku_20260223_001932_g00208.weights.h5
```

Use arrow keys to move cursor, Enter to place stone. `--difficulty` accepts
`easy|medium|hard` or any positive integer simulation count. The AI uses MCTS
with the trained network for move selection.

### Qt GUI (play_qt.py)

Play with a graphical board UI:

```bash
python play_qt.py
```

`play_qt.py` supports:
- Human vs AI and Human vs Human modes
- Auto-load of the current best checkpoint on startup (with fallback to latest/checkpoint)
- Difficulty tiers and custom simulation counts
- Analysis heatmap mode: continuous MCTS pondering on the human turn, with
  per-move shading and policy tooltips on empty intersections
- Winning-line highlight at game end

## Architecture

- **Board / rules**: 15×15 Gomoku, 5 in a row wins.
- **Input tensor**: `(15, 15, 6)` from the current player's perspective.
  - Plane 0: current player's stones
  - Plane 1: opponent stones
  - Plane 2: current player's open-four threat cells
  - Plane 3: opponent open-four threat cells
  - Plane 4: current player's open-three threat cells
  - Plane 5: opponent open-three threat cells
- **Backbone**:
  - 3×3 conv stem, 128 channels, BN, ReLU
  - 10 residual blocks (each: 3×3 conv -> BN -> ReLU -> 3×3 conv -> BN -> skip add -> ReLU)
  - Squeeze-and-Excitation on every other residual block (blocks 2, 4, 6, 8, 10)
- **Policy head**:
  - 1×1 conv (2 channels) -> BN -> ReLU -> Flatten -> Dense(225)
  - Outputs raw logits for all board cells (no softmax in-model; masking/softmax is applied in MCTS/training)
- **Value head**:
  - 1×1 conv (1 channel) -> BN -> ReLU -> Flatten -> Dense(128, ReLU) -> Dense(1, tanh)
  - Outputs expected game outcome in `[-1, 1]`
- **Search**: Batched MCTS with virtual loss, candidate-move pruning, and GPU batched inference.
- **Training targets**:
  - Policy target: MCTS visit distribution
  - Value target: final game outcome from the side-to-move perspective

## Files

- `gomoku.py` - Core game logic, neural network, and MCTS implementation
- `train.py` - Self-play training loop (inline eval/promotion disabled by default)
- `eval.py` - Checkpoint evaluation, calibration, and persistent Glicko-2 updates
- `eval_tournament.py` - Swiss + McMahon tournament runner with persistent ratings and best-checkpoint promotion
- `train_tournament_loop.sh` - Repeats `GZ_NUM_GAMES`-sized training chunks + tournament until a target total game count
- `play.py` - Interactive terminal UI for human vs AI
- `play_qt.py` - PyQt6 graphical UI (human vs AI / human vs human, analysis heatmap)
- `book_openings.py` - Opening book for evaluation consistency
- `mcts_accel.pyx` - Cython-accelerated MCTS functions
- `setup_accel.py` - Build script for Cython extensions

## Performance

Backend order for threat planes is Cython -> Numba -> NumPy. Typical 15x15 timing is ~3us for Cython/Numba and ~150us for NumPy fallback. Cython also accelerates key MCTS paths. GPU acceleration via TensorFlow is highly recommended for training and strong play.

## License

GNU General Public License v3.0 (GPLv3). See `LICENSE`.
