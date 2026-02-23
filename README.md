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

Training parameters can be adjusted in `train.py`. The script will:
- Generate self-play games using batched MCTS
- Train the network on policy and value targets
- Save checkpoints to `weights/`
- Evaluate new checkpoints against previous versions

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
```

Standard eval runs update persistent Glicko-2 ratings in `weights/glicko2_ratings.pkl`.

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

- **Board**: 15×15 grid, standard Gomoku rules (5 in a row wins)
- **Network**: Residual CNN with policy and value heads
- **Input encoding**: 6 planes (2 stone planes + 4 threat planes)
- **MCTS**: Batched search with virtual loss for parallelization
- **Training**: Policy loss (cross-entropy) + value loss (MSE)

## Files

- `gomoku.py` - Core game logic, neural network, and MCTS implementation
- `train.py` - Self-play training loop with inline evaluation
- `eval.py` - Checkpoint evaluation and ELO calculation
- `play.py` - Interactive terminal UI for human vs AI
- `play_qt.py` - PyQt6 graphical UI (human vs AI / human vs human, analysis heatmap)
- `book_openings.py` - Opening book for evaluation consistency
- `mcts_accel.pyx` - Cython-accelerated MCTS functions
- `setup_accel.py` - Build script for Cython extensions

## Performance

Backend order for threat planes is Cython -> Numba -> NumPy. Typical 15x15 timing is ~3us for Cython/Numba and ~150us for NumPy fallback. Cython also accelerates key MCTS paths. GPU acceleration via TensorFlow is highly recommended for training and strong play.

## License

GNU General Public License v3.0 (GPLv3). See `LICENSE`.
