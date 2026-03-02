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
then `eval_tournament.py`, and repeats until `weights/train_state.pkl` reaches
`--games-goal`. Both module calls in the loop run with `2>/dev/null`.
It auto-selects Swiss rounds per cycle by tournament player count:
`<=70 -> 6`, `71-140 -> 7`, `>140 -> 8`.
To override this, pass `--swiss-rounds` after `--`, e.g.
`./train_tournament_loop.sh --games-goal 120000 -- --swiss-rounds 7`.

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

# Run tournament across all weight files in a folder
# (Swiss seeding -> McMahon final, no heads-up)
python eval_tournament.py --tournament-dir botb-weights

# Tournament with practical-tie (shared-gold) settings
python eval_tournament.py --tournament-dir botb-weights --shared-gold-margin 0.02 --shared-gold-min-games 120

# Large field example (hundreds of checkpoints)
python eval_tournament.py --tournament-dir botb-weights --swiss-rounds 6 --mcmahon-rounds 5 --mcmahon-max-players 24

# Tournament without touching persistent ratings / best checkpoint files
python eval_tournament.py --tournament-dir botb-weights --no-persist-ratings --no-promote-winner

# Run eval without changing persistent ratings
python eval.py --no-rating-update

# Pretty-print a ratings table
python ratings_glicko2.py Mixed-competitor-weights/glicko2_ratings.pkl
python ratings_glicko2.py Mixed-competitor-weights/glicko2_ratings.pkl --min-games 40 --sort games
```

Standard eval runs update persistent Glicko-2 ratings in `weights/glicko2_ratings.pkl`.
Tournament mode (`eval_tournament.py`) is persistent by default. It runs Swiss
seeding across all checkpoints, then a McMahon final group selected by rating
bar, and writes outputs inside the tournament folder by default:
`<tournament-dir>/glicko2_ratings.pkl`,
`<tournament-dir>/gomoku_best.weights.h5`,
`<tournament-dir>/best_checkpoint.pkl`.
Default tournament sims are `--swiss-sims 50` and `--mcmahon-sims 160`.
Default tournament MCTS batch sizes are `--swiss-batch-size 16` and
`--mcmahon-batch-size 8`.
Tournament discovery automatically excludes
`gomoku_best.weights.h5`, `gomoku_weights.weights.h5`, and
`gomoku_*_final.weights.h5`.
Use `--no-persist-ratings --no-promote-winner` for dry-run/transient behavior.
`eval.py --tournament-dir ...` remains supported as a compatibility wrapper.

### Playing

Play against the trained AI:

```bash
python play.py

# Pick a difficulty tier
python play.py --difficulty easy    # 100 sims
python play.py --difficulty medium  # 400 sims (default)
python play.py --difficulty hard    # 1600 sims

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
