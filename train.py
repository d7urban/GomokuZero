#!/usr/bin/env python3
"""
Gomoku — AlphaZero-style training via MCTS self-play.

Each self-play game uses MCTS to generate move-probability targets.
The network is trained with:
    policy loss  = cross-entropy(predicted logits, MCTS visit distribution)
    value loss   = MSE(predicted value, game outcome)

Architecture: single-process, GPU-accelerated.
  • Self-play and training share one model in the main process.
  • Multiple games run concurrently with interleaved MCTS: leaf states
    from all active games are batched into a single GPU forward pass.
  • Inline evaluation after each checkpoint.
"""

import numpy as np
import math
import os, pickle, time, glob, signal, shutil
from datetime import datetime
from collections import deque

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf
from tensorflow import keras

tf.get_logger().setLevel("ERROR")

_gpus = tf.config.list_physical_devices("GPU")

from gomoku import (
    BOARD_SIZE, PLAYER1, NUM_INPUT_PLANES,
    GomokuGame, create_model, encode_state, make_predict_fn,
    mcts_policy, select_action,
    mcts_begin, mcts_expand_root, mcts_select_leaves, mcts_process_results,
)

from eval import (
    load_or_create_openings, find_checkpoints,
    run_match_sequential, elo_delta,
)

# ── Tunables ────────────────────────────────────────────────────────────────
CONCURRENT_GAMES  = 16           # games interleaved on GPU simultaneously
NUM_GAMES         = 5000         # total self-play games
C_PUCT            = 1.5
DIRICHLET_ALPHA   = 0.15
NOISE_FRAC_START  = 0.35         # Dirichlet noise fraction (anneals down)
NOISE_FRAC_END    = 0.25         # noise floor after NOISE_DECAY_GAMES
NOISE_DECAY_GAMES = 3000         # linear anneal over this many games
TEMP_THRESHOLD    = 40           # move number after which temperature drops
BATCH_SIZE        = 1024
REPLAY_SIZE       = 300_000      # positions — larger to reduce forgetting
REPLAY_RECENT_FRAC = 0.75       # fraction of each batch from recent positions
REPLAY_RECENT_SIZE = 50_000     # how many most-recent positions count as "recent"
TRAIN_STEPS_RATIO = 4.0          # gradient steps = new_positions * ratio / BATCH_SIZE
LR                = 1e-3
WEIGHT_DECAY      = 1e-4
VALUE_LOSS_COEFF  = 1.0          # weight on value loss (tune if v-loss dominates)
SAVE_INTERVAL     = 200          # games between checkpoints

# Batched MCTS — leaves evaluated per forward pass.
# Smaller batches = more backup rounds = better-informed UCB selection.
# GPU handles batch-8 in ~1ms so throughput is not the bottleneck.
MCTS_BATCH_SIZE   = 8

# Self-play simulation budget
MCTS_SIMS         = 200          # full-search sims (for policy training)

# Playout cap randomization: most moves get a fast search (value only),
# some get a full search (policy + value).  This generates more games
# for the value head while preserving policy quality.
PCAP_FAST_SIMS    = 40           # sims for fast-search moves
PCAP_FULL_FRAC    = 0.50         # fraction of moves that get full search

# Asymmetric self-play: in a fraction of games, one side is randomly
# weakened to produce decisive results and train the value head.
# The fraction decays linearly so training converges to symmetric play.
ASYM_WEAK_SIMS    = 80           # sims for the weakened side
ASYM_FRAC_START   = 0.5          # asymmetric fraction at game 0
ASYM_FRAC_END     = 0.1          # asymmetric fraction after decay period
ASYM_DECAY_GAMES  = 3000         # linear decay over this many games

# Random opening plies: play this many random moves before MCTS kicks in.
# Creates diverse starting positions and breaks defensive symmetry.
RANDOM_OPENING_PLIES = 6
BOOK_OPENING_FRAC    = 0.30    # fraction of games that use book openings

# Best-opponent play: fraction of games played against the current best
# checkpoint instead of self-play.  Forces the network to learn how to
# beat strategies it wouldn't encounter in pure self-play.
BEST_PLAY_FRAC       = 0.2

# Soft resignation: instead of ending games when hopeless, continue
# playing with fast search (value-only training).  Avoids censoring
# lost positions from the value head while not polluting policy.
RESIGN_THRESHOLD     = 0.85     # |v| above which a position is "hopeless"
RESIGN_CONSECUTIVE   = 5        # must be hopeless for this many moves in a row
RESIGN_MIN_MOVES     = 30       # don't check before this many moves

# ── Inline evaluation ──────────────────────────────────────────────────────
# Two-tier eval system:
# 1. Diagnostic: lightweight t-1 check every checkpoint (SAVE_INTERVAL).
#    Just prints Elo trend — no promotion gate.
# 2. Promotion: eval vs current best, triggered when diagnostic win%
#    exceeds threshold for consecutive checkpoints.  Promotes if Wilson
#    95% CI lower bound > 50%.
INLINE_EVAL_OPENINGS  = 30       # 30 openings × 2 colors = 60 games (diagnostic)
PROMOTE_EVAL_OPENINGS = 60       # 60 openings × 2 colors = 120 games (promotion)
DIAG_EVAL_SIMS        = 50       # fewer sims for fast diagnostic
PROMOTE_EVAL_SIMS     = 100      # full sims for promotion decision
PROMOTE_DIAG_BLACK_PCT = 80.0    # Black win% that counts as "strong" diagnostic
PROMOTE_CONSEC_NEEDED = 2        # consecutive strong diagnostics to trigger
EMERG_LR_BLACK_PCT    = 40.0    # Black win% below which mild regression (2 consec)
EMERG_LR_SEVERE_PCT   = 25.0   # Black win% below which severe regression (immediate)
PROMOTE_COOLDOWN      = 400      # min games between promotion evals
PROMOTE_CI_LEVEL      = 0.95     # confidence level for Wilson interval
EMERG_LR_FACTOR       = 0.3     # multiply LR by this on emergency
EMERG_LR_MIN_GAMES    = 2000    # don't fire emergency before this many games

# ── Plateau detection ────────────────────────────────────────────────────────
class PlateauDetector:
    """
    Detects loss plateaus using a rolling window slope test.
    Emits a message when conditions persist for several checks.
    """

    def __init__(self, window=80, check_every=10, warmup_batches=40,
                 slope_rel_thresh=3e-4, v_improve_rel=5e-4,
                 min_replay=20_000, cooldown=50, persist=3):
        self.window = window
        self.check_every = check_every
        self.warmup_batches = warmup_batches
        self.slope_rel_thresh = slope_rel_thresh
        self.v_improve_rel = v_improve_rel
        self.min_replay = min_replay
        self.cooldown = cooldown
        self.persist = persist

        self._ploss = []
        self._vloss = []
        self._ent = []
        self._dec = []
        self.batches = 0
        self._persist_counts = {}
        self._last_emit = -cooldown

    def _rel_slope(self, y):
        n = len(y)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        m = np.polyfit(x, y, 1)[0]
        mean_abs = np.mean(np.abs(y)) + 1e-12
        return m / mean_abs

    def update(self, ploss, vloss, target_entropy, decisive_rate, replay_len):
        self._ploss.append(ploss)
        self._vloss.append(vloss)
        self._ent.append(target_entropy)
        self._dec.append(decisive_rate)
        for buf in (self._ploss, self._vloss, self._ent, self._dec):
            if len(buf) > self.window:
                buf.pop(0)
        self.batches += 1

        if self.batches < self.warmup_batches:
            return None
        if self.batches % self.check_every != 0:
            return None
        if replay_len < self.min_replay:
            return None
        if len(self._ploss) < self.window:
            return None

        pl_s = self._rel_slope(self._ploss)
        vl_s = self._rel_slope(self._vloss)
        ent_s = self._rel_slope(self._ent)
        dec_s = self._rel_slope(self._dec)

        decisive_now = self._dec[-1] if self._dec else 0.0
        ent_now = self._ent[-1] if self._ent else 0.0

        thr = self.slope_rel_thresh
        v_thr = self.v_improve_rel

        p_flat = abs(pl_s) < thr
        v_flat = abs(vl_s) < thr
        v_improving = vl_s < -v_thr
        ent_flat = abs(ent_s) < thr
        dec_flat = abs(dec_s) < thr and decisive_now < 0.9  # saturated is fine
        dec_rising = dec_s > thr

        msg = None
        cond_key = None

        if p_flat and v_improving and dec_flat:
            cond_key = "capacity"
            msg = ("PLATEAU: policy flat, value improving, decisive rate stable "
                   "→ policy head capacity-limited. Consider more res-blocks/filters.")
        elif p_flat and v_improving and dec_rising:
            self._persist_counts.clear()
            return None
        elif p_flat and v_flat and ent_flat and decisive_now < 0.3:
            cond_key = "stalemate"
            msg = ("PLATEAU: all flat, low decisive rate "
                   "→ defensive stalemate. Increase asymmetry or exploration.")
        elif p_flat and v_flat and ent_s < -thr:
            cond_key = "mcts_sharp"
            msg = ("PLATEAU: losses flat, entropy falling "
                   "→ MCTS sharpening but net can't follow. Lower LR or add capacity.")
        elif p_flat and ent_s > thr:
            cond_key = "ent_rising"
            msg = ("PLATEAU: policy flat, entropy rising "
                   "→ targets diversifying. Lower LR or add capacity.")
        elif p_flat and v_flat:
            cond_key = "generic"
            msg = ("PLATEAU: policy and value both flat "
                   "→ generic stall. Check data quality, LR, exploration.")

        if cond_key is None:
            self._persist_counts.clear()
            return None

        self._persist_counts.setdefault(cond_key, 0)
        self._persist_counts[cond_key] += 1
        for k in list(self._persist_counts):
            if k != cond_key:
                del self._persist_counts[k]

        if self._persist_counts[cond_key] < self.persist:
            return None
        if (self.batches - self._last_emit) < self.cooldown:
            return None

        self._last_emit = self.batches
        return (msg +
                f"  [pl_s={pl_s:+.2e}, vl_s={vl_s:+.2e}, "
                f"ent_s={ent_s:+.2e}, dec_s={dec_s:+.2e}, "
                f"entropy={ent_now:.2f}, decisive={decisive_now:.2f}]")


class LRScheduler:
    """Reduce learning rate when policy learning stalls.

    Tracks P - H (policy loss minus target entropy ≈ KL divergence from
    MCTS policy to network policy).  This gap measures how well the
    network matches the MCTS output, independent of target difficulty.
    When MCTS targets sharpen, both P and H drop together — the gap is
    what indicates actual learning progress.

    When the rolling average of the gap hasn't improved by `rel_threshold`
    for `patience` checks, LR is multiplied by `factor`.

    Usage:
        lr_sched = LRScheduler(optimizer)
        msg = lr_sched.update(ploss, entropy)
        if msg: print(msg)
    """

    def __init__(self, optimizer, window=200, check_every=50,
                 patience=5, factor=0.3, min_lr=1e-5, warmup=100,
                 rel_threshold=0.005, warmup_start_frac=0.3):
        self.optimizer = optimizer
        self.window = window
        self.check_every = check_every
        self.patience = patience        # checks without improvement
        self.factor = factor             # LR *= factor on reduction
        self.min_lr = min_lr
        self.warmup = warmup
        self.rel_threshold = rel_threshold  # relative improvement needed
        self.warmup_start_frac = warmup_start_frac  # start at 10% of target LR
        self._target_lr = float(optimizer.learning_rate.numpy())

        self._gap_buf = []
        self._best_avg = float("inf")
        self._checks_without_improvement = 0
        self._batches = 0
        self._reductions = 0
        self._warmup_done = False

    @property
    def current_lr(self):
        return float(self.optimizer.learning_rate.numpy())

    def update(self, ploss, entropy=0.0):
        gap = ploss - entropy
        self._gap_buf.append(gap)
        if len(self._gap_buf) > self.window:
            self._gap_buf.pop(0)
        self._batches += 1

        # Linear LR warmup: ramp from target*start_frac to target
        if not self._warmup_done and self._batches <= self.warmup:
            frac = self.warmup_start_frac + (
                (1.0 - self.warmup_start_frac) * self._batches / self.warmup)
            self.optimizer.learning_rate.assign(self._target_lr * frac)
            if self._batches == self.warmup:
                self._warmup_done = True
            return None

        if self._batches % self.check_every != 0:
            return None
        if len(self._gap_buf) < self.window:
            return None

        avg = float(np.mean(self._gap_buf))

        # Relative improvement: needs rel_threshold drop from best
        if avg < self._best_avg * (1 - self.rel_threshold):
            self._best_avg = avg
            self._checks_without_improvement = 0
            return None

        self._checks_without_improvement += 1

        if self._checks_without_improvement >= self.patience:
            old_lr = self.current_lr
            if old_lr <= self.min_lr:
                self._checks_without_improvement = 0  # reset to avoid spam
                return (f"⚠ P-H gap plateaued at {avg:.4f} with LR at floor "
                        f"({old_lr:.0e}) after {self._reductions} reductions — "
                        f"consider a larger model (more res-blocks or filters)")
            new_lr = max(old_lr * self.factor, self.min_lr)
            self.optimizer.learning_rate.assign(new_lr)
            self._checks_without_improvement = 0
            self._best_avg = avg  # reset baseline after reduction
            self._reductions += 1
            return (f"LR reduced: {old_lr:.2e} → {new_lr:.2e} "
                    f"(P-H gap {avg:.4f}, reduction #{self._reductions})")

        return None

    def get_state(self):
        return {
            "gap_buf": list(self._gap_buf),
            "best_avg": self._best_avg,
            "checks_without_improvement": self._checks_without_improvement,
            "batches": self._batches,
            "reductions": self._reductions,
            "current_lr": self.current_lr,
            "target_lr": self._target_lr,
            "warmup_done": self._warmup_done,
        }

    def load_state(self, state):
        if state is None:
            return
        # Accept both old "ploss_buf" and new "gap_buf" keys
        self._gap_buf = state.get("gap_buf", state.get("ploss_buf", []))
        self._best_avg = state.get("best_avg", float("inf"))
        self._checks_without_improvement = state.get(
            "checks_without_improvement", 0)
        self._batches = state.get("batches", 0)
        self._reductions = state.get("reductions", 0)
        self._warmup_done = state.get("warmup_done", True)  # assume done for old states
        self._target_lr = state.get("target_lr", self._target_lr)
        saved_lr = state.get("current_lr")
        if saved_lr is not None:
            self.optimizer.learning_rate.assign(saved_lr)
            print(f"  Restored LR: {saved_lr:.2e} "
                  f"({self._reductions} prior reductions)")


# ── Symmetry augmentation ────────────────────────────────────────────────
def _augment_batch(states, policies):
    """Apply random D4 symmetry + random translation to each batch.

    Step 1 — D4 symmetry (one transform for whole batch, 8× coverage).
    Step 2 — Random translation per sample, forcing the CNN to learn
             position-invariant patterns instead of "play near center".
             Keeps all content (stones + nearby policy mass) on board.
    """
    bs = BOARD_SIZE

    # ── Step 1: D4 symmetry (batch-wide) ──
    k = np.random.randint(8)
    s, p = states, policies
    if k > 0:
        rot = k % 4
        flip = k >= 4
        if flip:
            s = s[:, :, ::-1, :]
        if rot:
            s = np.rot90(s, rot, axes=(1, 2))
        p2 = p.reshape(-1, bs, bs)
        if flip:
            p2 = p2[:, :, ::-1]
        if rot:
            p2 = np.rot90(p2, rot, axes=(1, 2))
        p = p2.reshape(-1, bs * bs)
        s = np.ascontiguousarray(s)
        p = np.ascontiguousarray(p)

    # ── Step 2: Random translation (per-sample) ──
    batch_n = s.shape[0]
    s_out = np.zeros_like(s)
    p2d = p.reshape(batch_n, bs, bs)
    p_out = np.zeros((batch_n, bs, bs), dtype=p.dtype)

    _MARGIN = 3  # keep stones + candidate moves off the very edge

    for i in range(batch_n):
        occ = (s[i, :, :, 0] != 0) | (s[i, :, :, 1] != 0)
        rows, cols = np.where(occ)
        if len(rows) == 0:
            s_out[i] = s[i]
            p_out[i] = p2d[i]
            continue

        # Bounding box of content + margin for policy spread
        r0 = max(0, int(rows.min()) - _MARGIN)
        r1 = min(bs - 1, int(rows.max()) + _MARGIN)
        c0 = max(0, int(cols.min()) - _MARGIN)
        c1 = min(bs - 1, int(cols.max()) + _MARGIN)

        # Random offset keeping padded bbox on board
        dr_lo, dr_hi = -r0, bs - 1 - r1
        dc_lo, dc_hi = -c0, bs - 1 - c1
        if dr_lo >= dr_hi and dc_lo >= dc_hi:
            # Content already spans nearly the whole board
            s_out[i] = s[i]
            p_out[i] = p2d[i]
            continue

        dr = np.random.randint(dr_lo, dr_hi + 1)
        dc = np.random.randint(dc_lo, dc_hi + 1)

        if dr == 0 and dc == 0:
            s_out[i] = s[i]
            p_out[i] = p2d[i]
            continue

        # Copy with translation
        sr = slice(max(0, -dr), min(bs, bs - dr))
        sc = slice(max(0, -dc), min(bs, bs - dc))
        dr_s = slice(max(0, dr), min(bs, bs + dr))
        dc_s = slice(max(0, dc), min(bs, bs + dc))

        s_out[i, dr_s, dc_s, :] = s[i, sr, sc, :]
        p_out[i, dr_s, dc_s] = p2d[i, sr, sc]

    # Renormalize policy (tiny mass may be lost at edges)
    p_flat = p_out.reshape(batch_n, bs * bs)
    p_sums = p_flat.sum(axis=1, keepdims=True)
    p_sums = np.maximum(p_sums, 1e-8)
    p_flat /= p_sums

    return s_out, p_flat


def _sample_replay(replay, batch_size):
    """Sample from replay buffer with recency bias.

    REPLAY_RECENT_FRAC of the batch comes from the most recent
    REPLAY_RECENT_SIZE positions; the rest from the entire buffer.
    This reduces off-policy drift while preserving diversity.
    Falls back to uniform when the buffer is smaller than the
    recent window.
    """
    n = len(replay)
    recent_window = min(REPLAY_RECENT_SIZE, n)

    if recent_window >= n:
        # Buffer smaller than recent window → uniform
        return np.random.choice(n, batch_size, replace=False)

    n_recent = int(batch_size * REPLAY_RECENT_FRAC)
    n_old = batch_size - n_recent

    # Recent: sample from the tail of the deque
    recent_start = n - recent_window
    recent_idxs = recent_start + np.random.choice(
        recent_window, n_recent, replace=False)

    # Old: sample from entire buffer (may overlap with recent, that's fine)
    old_idxs = np.random.choice(n, n_old, replace=False)

    return np.concatenate([recent_idxs, old_idxs])


# ── Persistence ─────────────────────────────────────────────────────────────
_STATE_PATH  = "weights/train_state.pkl"
_REPLAY_PATH = "weights/replay_buffer.pkl"


def _get_optimizer_state(optimizer):
    if hasattr(optimizer, "get_weights"):
        return optimizer.get_weights()
    return [v.numpy() for v in optimizer.variables]


def _set_optimizer_state(optimizer, state):
    if hasattr(optimizer, "set_weights"):
        optimizer.set_weights(state)
    else:
        for var, val in zip(optimizer.variables, state):
            var.assign(val)


def _save_training_state(game_count, optimizer, replay, lr_scheduler=None):
    """Persist everything needed for a seamless resume."""
    opt_weights = _get_optimizer_state(optimizer)

    state = {
        "board_size": BOARD_SIZE,
        "total_games": game_count,
        "optimizer_weights": opt_weights,
    }
    if lr_scheduler is not None:
        state["lr_scheduler"] = lr_scheduler.get_state()

    with open(_STATE_PATH, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(_REPLAY_PATH, "wb") as f:
        pickle.dump(list(replay), f, protocol=pickle.HIGHEST_PROTOCOL)

    with open("weights/model_config.pkl", "wb") as f:
        pickle.dump({"board_size": BOARD_SIZE, "total_games": game_count}, f)


def _load_training_state(model, optimizer, replay):
    """Restore optimizer state and replay buffer.  Returns (starting_game, lr_state)."""
    starting_game = 0
    lr_state = None

    cfg_path = _STATE_PATH
    if not os.path.exists(cfg_path):
        cfg_path = "weights/model_config.pkl"

    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "rb") as f:
                cfg = pickle.load(f)
            starting_game = cfg.get("total_games", 0)
            print(f"  Resuming from game {starting_game}")

            opt_weights = cfg.get("optimizer_weights")
            lr_state = cfg.get("lr_scheduler")
            if opt_weights is not None:
                saved_w = [w.numpy() for w in model.trainable_variables]
                dummy_s = np.zeros((1, BOARD_SIZE, BOARD_SIZE, NUM_INPUT_PLANES), dtype=np.float32)
                dummy_p = np.zeros((1, BOARD_SIZE * BOARD_SIZE), dtype=np.float32)
                dummy_p[0, 0] = 1.0
                dummy_v = np.zeros((1,), dtype=np.float32)
                dummy_w = np.ones((1,), dtype=np.float32)
                _train_step(model, optimizer, dummy_s, dummy_p, dummy_v, dummy_w)
                for var, w in zip(model.trainable_variables, saved_w):
                    var.assign(w)
                try:
                    _set_optimizer_state(optimizer, opt_weights)
                    print("  Restored optimizer state (momentum/variance).")
                except Exception as e:
                    print(f"  Warning: could not restore optimizer state: {e}")
                    print("  Continuing with fresh optimizer.")
        except Exception as e:
            print(f"  Warning: could not load config: {e}")

    if os.path.exists(_REPLAY_PATH):
        try:
            with open(_REPLAY_PATH, "rb") as f:
                saved_replay = pickle.load(f)
            for item in saved_replay:
                replay.append(item)
            print(f"  Restored replay buffer: {len(replay):,} positions.")
        except Exception as e:
            print(f"  Warning: could not load replay buffer: {e}")
            print("  Starting with empty buffer (will refill via self-play).")

    return starting_game, lr_state


# ── Interleaved self-play (multiple games, shared GPU inference) ────────────
def _play_games_interleaved(predict_fn, game_configs, best_predict_fn=None,
                            noise_frac=0.25):
    """Play multiple self-play games with shared GPU inference.

    All games contribute leaf states to a single batched forward pass each
    round.  This fills GPU idle time that sequential play would waste.

    predict_fn: compiled prediction function from make_predict_fn()
    game_configs: list of (sims, weak_sims, weak_player, opening, vs_best)
                  opening: int (random plies) or list of (r,c) (book moves)
    best_predict_fn: predict function for best checkpoint (None if no best)
    Returns: list of (trajectory, winner, resigned)
    """

    # Per-game state
    class _G:
        __slots__ = ('game', 'sims', 'weak_sims', 'weak_player',
                     'trajectory', 'winner', 'finished', 'move_num',
                     'ctx', 'root_state', 'phase', 'move_sims',
                     'resign_count', 'vs_best', 'training_player',
                     'is_full_search', 'soft_resigned')

    all_games = []
    for cfg in game_configs:
        sims, weak_sims, weak_player, opening, vs_best = cfg
        g = _G()
        g.game = GomokuGame()
        g.sims = sims
        g.weak_sims = weak_sims
        g.weak_player = weak_player
        g.trajectory = []
        g.winner = 0
        g.finished = False
        g.ctx = None
        g.root_state = None
        g.phase = "new_move"
        g.move_sims = sims
        g.move_num = 0
        g.resign_count = 0
        g.is_full_search = True
        g.soft_resigned = False
        g.vs_best = vs_best and best_predict_fn is not None
        # Training model plays a random side in vs_best games
        g.training_player = (PLAYER1 if np.random.random() < 0.5
                             else -PLAYER1) if g.vs_best else None

        # Opening: book moves (list) or random plies (int)
        if isinstance(opening, list):
            for r, c in opening:
                reward, done = g.game.make_move(r, c)
                if done:
                    g.finished = True; break
        else:
            for _ in range(opening):
                moves = g.game.get_valid_moves()
                if not moves:
                    g.finished = True; break
                r, c = moves[np.random.randint(len(moves))]
                reward, done = g.game.make_move(r, c)
                if done:
                    g.finished = True; break

        g.move_num = len(g.game.move_history)
        all_games.append(g)

    def _is_best_turn(g):
        """True if the best model should play this move."""
        return g.vs_best and g.game.current_player != g.training_player

    # Main interleaving loop
    while True:
        active = [g for g in all_games if not g.finished]
        if not active:
            break

        # Two separate batches: one for training model, one for best model
        train_states = []
        train_root_evals = []
        train_leaf_evals = []
        best_states = []
        best_root_evals = []
        best_leaf_evals = []

        for g in active:
            is_best = _is_best_turn(g)
            states = best_states if is_best else train_states
            r_evals = best_root_evals if is_best else train_root_evals
            l_evals = best_leaf_evals if is_best else train_leaf_evals

            if g.phase == "new_move":
                # Determine sims for this move
                if is_best:
                    g.move_sims = g.sims  # best plays at full strength
                    g.is_full_search = True
                elif g.soft_resigned:
                    # Soft resign: fast search, value-only training
                    g.move_sims = PCAP_FAST_SIMS
                    g.is_full_search = False
                elif (g.weak_player is not None and
                        g.game.current_player == g.weak_player):
                    g.move_sims = g.weak_sims
                    g.is_full_search = False  # weak side always "fast"
                else:
                    # Playout cap randomization: most moves get fast
                    # search (value only), some get full (policy + value)
                    g.is_full_search = (np.random.random() < PCAP_FULL_FRAC)
                    g.move_sims = g.sims if g.is_full_search else PCAP_FAST_SIMS

                # No noise for: best model, fast searches, soft-resigned
                add_noise = (not is_best) and g.is_full_search
                g.ctx, g.root_state = mcts_begin(
                    g.game,
                    num_simulations=g.move_sims,
                    batch_size=MCTS_BATCH_SIZE,
                    c_puct=C_PUCT,
                    add_noise=add_noise,
                    dirichlet_alpha=DIRICHLET_ALPHA,
                    noise_frac=noise_frac,
                )
                idx = len(states)
                states.append(g.root_state)
                r_evals.append((g, idx))
                g.phase = "root_pending"

            elif g.phase == "root_pending":
                pass

            elif g.phase == "searching":
                leaf_states = mcts_select_leaves(g.ctx)
                if leaf_states:
                    start = len(states)
                    states.extend(leaf_states)
                    l_evals.append((g, start, len(leaf_states)))
                else:
                    mcts_process_results(g.ctx)

        # ── GPU forward passes ──────────────────────────────────────
        for states, r_evals, l_evals, fn in [
            (train_states, train_root_evals, train_leaf_evals, predict_fn),
            (best_states, best_root_evals, best_leaf_evals, best_predict_fn),
        ]:
            if not states:
                continue
            batch = np.array(states, dtype=np.float32)
            logits_np, values_np = fn(batch)
            values_np = values_np.ravel()

            for g, idx in r_evals:
                mcts_expand_root(g.ctx, logits_np[idx], values_np[idx])
                g.phase = "searching"

            for g, start, count in l_evals:
                mcts_process_results(
                    g.ctx,
                    logits_np[start:start + count],
                    values_np[start:start + count],
                )

        # ── Advance games where MCTS search is complete ─────────────
        for g in active:
            if g.ctx is not None and g.ctx["sims_done"] >= g.ctx["sims_target"]:
                root = g.ctx["root"]
                is_best = _is_best_turn(g)

                # Temperature: full searches use exploration temp,
                # fast searches and best model use low temp for strength
                if is_best or not g.is_full_search:
                    temp = 0.1
                else:
                    temp = 1.0 if g.move_num < TEMP_THRESHOLD else 0.1
                pi = mcts_policy(root, temperature=temp)

                # Record position — is_full_search determines policy weight
                g.trajectory.append((
                    g.root_state,
                    pi,
                    g.game.current_player,
                    g.move_sims,
                    g.is_full_search,
                ))

                # Soft resign check: if hopeless for several moves,
                # switch to fast play (value-only training data).
                if (not is_best and not g.soft_resigned
                        and g.move_num >= RESIGN_MIN_MOVES):
                    if root.q_value < -RESIGN_THRESHOLD:
                        g.resign_count += 1
                    else:
                        g.resign_count = 0

                    if g.resign_count >= RESIGN_CONSECUTIVE:
                        g.soft_resigned = True

                row, col = select_action(pi)
                reward, done = g.game.make_move(row, col)
                g.move_num += 1

                if done:
                    if reward == 1:
                        g.winner = g.game.current_player
                    elif reward == -1:
                        g.winner = -g.game.current_player
                    g.finished = True
                else:
                    g.phase = "new_move"

                g.ctx = None

    return [(g.trajectory, g.winner, g.soft_resigned)
            for g in all_games]


# ── Training step ──────────────────────────────────────────────────────────
@tf.function
def _train_step(model, optimizer, states, target_pi, target_v, pi_weights):
    with tf.GradientTape() as tape:
        logits, value = model(states, training=True)
        value = tf.squeeze(value, axis=1)

        per_sample_ploss = tf.nn.softmax_cross_entropy_with_logits(
            labels=target_pi, logits=logits)
        # Normalized policy loss: independent of PCAP_FULL_FRAC.
        # sum(w * CE) / sum(w) gives the mean CE over full-search
        # positions only, regardless of how many fast-search (w=0)
        # samples are in the batch.
        w_sum = tf.reduce_sum(pi_weights) + 1e-8
        policy_loss = tf.reduce_sum(pi_weights * per_sample_ploss) / w_sum

        value_loss = tf.reduce_mean(tf.square(target_v - value))
        loss = policy_loss + VALUE_LOSS_COEFF * value_loss

    grads = tape.gradient(loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 5.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return policy_loss, value_loss, loss


# ── Best checkpoint tracking ──────────────────────────────────────────────
BEST_WEIGHTS_FILE = "weights/gomoku_best.weights.h5"
BEST_STATE_FILE   = "weights/best_checkpoint.pkl"

def _load_best_state():
    if os.path.exists(BEST_STATE_FILE):
        try:
            with open(BEST_STATE_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return {"path": None, "game_count": 0, "elo_vs_long": None}


def _save_best_state(state):
    with open(BEST_STATE_FILE, "wb") as f:
        pickle.dump(state, f)


def _wilson_lower(wins, n, z=1.96):
    """Wilson score lower confidence bound.

    Returns the lower bound of the confidence interval for the true
    win probability.  z=1.96 → 95% CI, z=1.645 → 90% CI.
    Draws count as half a win.
    """
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - spread) / denom


def _run_diagnostic_eval(model, game_count, eval_openings, opp_model):
    """Lightweight eval vs previous checkpoint — prints per-color trends.

    Returns result dict or None if eval couldn't run.
    """
    checkpoints = find_checkpoints()
    if not checkpoints:
        return None

    current_idx = None
    for i, (gc, fp) in enumerate(checkpoints):
        if abs(gc - game_count) <= 10:
            current_idx = i
            break
    if current_idx is None or current_idx < 1:
        return None

    gc_prev, fp_prev = checkpoints[current_idx - 1]
    label = f"t-1 (g{gc_prev})"

    print(f"  ── Diag g{game_count} vs {label} ──", flush=True)
    opp_model.load_weights(fp_prev)
    result = run_match_sequential(
        model, opp_model, label,
        openings=eval_openings,
        sims=DIAG_EVAL_SIMS,
        batch_size=MCTS_BATCH_SIZE,
    )
    return result


def _run_promotion_eval(model, game_count, all_openings, best_state, opp_model):
    """Promotion eval vs current best checkpoint — promotes if both colors improved.

    Triggered when consecutive diagnostic Black win% exceeds threshold.
    Uses full sims (100) and a fresh random subset of openings.

    Promotion gate (per-color):
      - Black: Wilson lower CI of Black win rate > 60%
        (confirms strong attack; easily met if true rate is 85%+)
      - White: at least 1 win, OR avg loss survival > 35 moves
        (confirms defensive ability hasn't collapsed)
    """
    best_gc = best_state.get("game_count", 0)

    # Find the checkpoint we just saved
    checkpoints = find_checkpoints()
    cp_file = None
    for gc, fp in checkpoints:
        if abs(gc - game_count) <= 10:
            cp_file = fp
            break
    if cp_file is None:
        return best_state

    if not os.path.exists(BEST_WEIGHTS_FILE):
        # No best yet — auto-promote
        shutil.copy2(cp_file, BEST_WEIGHTS_FILE)
        best_state = {
            "path": cp_file,
            "game_count": game_count,
            "elo_vs_long": None,
        }
        _save_best_state(best_state)
        print(f"  ★ Auto-promoted to best (no previous best)", flush=True)
        return best_state

    if abs(game_count - best_gc) < PROMOTE_COOLDOWN:
        print(f"  (skip promotion: current best g{best_gc:05d} is recent)",
              flush=True)
        return best_state

    # Sample a fresh subset of openings (reproducible per game_count)
    rng = np.random.RandomState(game_count)
    n_openings = min(PROMOTE_EVAL_OPENINGS, len(all_openings))
    indices = rng.choice(len(all_openings), size=n_openings, replace=False)
    eval_openings = [all_openings[i] for i in indices]

    label = f"best (g{best_gc})"
    print(f"  ── Promotion eval g{game_count} vs {label} ──", flush=True)

    opp_model.load_weights(BEST_WEIGHTS_FILE)
    result = run_match_sequential(
        model, opp_model, label,
        openings=eval_openings,
        sims=PROMOTE_EVAL_SIMS,
        batch_size=MCTS_BATCH_SIZE,
    )

    z = 1.96 if PROMOTE_CI_LEVEL >= 0.95 else 1.645

    # ── Per-color gates ──
    # Black: Wilson lower CI on Black win rate must exceed 60%
    bw = result["black_wins"]
    bl = result["black_losses"]
    bn = bw + bl  # ignoring draws (rare in Gomoku)
    black_lower = _wilson_lower(bw, bn, z=z) if bn > 0 else 0.0

    # White: must show some defensive capability
    ww = result["white_wins"]
    wl = result["white_losses"]
    white_survival = result["avg_white_loss_moves"]
    white_ok = (ww > 0) or (white_survival >= 35)

    black_ok = black_lower > 0.60
    promoted = black_ok and white_ok

    if promoted:
        shutil.copy2(cp_file, BEST_WEIGHTS_FILE)
        best_state = {
            "path": cp_file,
            "game_count": game_count,
            "elo_vs_long": result["elo_delta"],
        }
        _save_best_state(best_state)
        white_reason = (f"W wins={ww}" if ww > 0
                        else f"W survival={white_survival:.0f}")
        print(f"  ★ Promoted to best "
              f"(B CI>{black_lower:.0%}, {white_reason}, "
              f"Elo Δ{result['elo_delta']:+.0f})", flush=True)
    else:
        reasons = []
        if not black_ok:
            reasons.append(f"B CI={black_lower:.0%}≤60%")
        if not white_ok:
            reasons.append(f"W wins=0, survival={white_survival:.0f}<35")
        print(f"  (best remains g{best_gc:05d} — {', '.join(reasons)})",
              flush=True)

    return best_state


# ── Main loop ──────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    print("=" * 60)
    print("Gomoku — AlphaZero-style MCTS Training")
    print("  (single-process GPU self-play + training)")
    print("=" * 60)

    if _gpus:
        print(f"GPU: {_gpus[0].name}")
    else:
        print("WARNING: No GPU detected — training will be slow.")

    model = create_model()
    optimizer = keras.optimizers.AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY)

    os.makedirs("weights", exist_ok=True)
    weights_file = "weights/gomoku_weights.weights.h5"
    starting_game = 0
    replay = deque(maxlen=REPLAY_SIZE)

    # Resume from checkpoint if available
    if os.path.exists(weights_file):
        choice = input(f"\nFound {weights_file} — continue training? "
                       "(press Enter to continue, 'n' to start fresh): "
                       ).strip().lower()
        if choice == "n":
            shutil.rmtree("weights", ignore_errors=True)
            os.makedirs("weights", exist_ok=True)
            lr_state = None
            print("  Deleted weights/ — starting fresh.\n")
        else:
            model.load_weights(weights_file)
            starting_game, lr_state = _load_training_state(
                model, optimizer, replay)
            print("  Loaded existing weights.\n")
    else:
        lr_state = None
        print("  No existing weights — starting fresh.\n")

    # Automatic LR scheduling: warmup ramp then plateau-based reduction
    lr_scheduler = LRScheduler(optimizer)
    if lr_state is not None:
        lr_scheduler.load_state(lr_state)
    else:
        # Fresh run: start at warmup fraction of target LR
        optimizer.learning_rate.assign(LR * lr_scheduler.warmup_start_frac)

    total = model.count_params()
    print(f"Model: {NUM_INPUT_PLANES}ch input, 6 res-blocks × 128 filters + SE, "
          f"{total:,} parameters")
    print(f"MCTS: {MCTS_SIMS} full / {PCAP_FAST_SIMS} fast sims "
          f"({PCAP_FULL_FRAC:.0%} full), batch size {MCTS_BATCH_SIZE} (GPU)")
    print(f"Asymmetric play: {ASYM_FRAC_START:.0%}→{ASYM_FRAC_END:.0%} "
          f"over {ASYM_DECAY_GAMES} games, weak side {ASYM_WEAK_SIMS} sims")
    print(f"Openings: {BOOK_OPENING_FRAC:.0%} book, "
          f"{1-BOOK_OPENING_FRAC:.0%} random ({RANDOM_OPENING_PLIES} plies)")
    print(f"Soft resign: threshold {RESIGN_THRESHOLD}, "
          f"{RESIGN_CONSECUTIVE} consecutive, after move {RESIGN_MIN_MOVES} "
          f"(value-only, {PCAP_FAST_SIMS} sims)")
    from gomoku import _USE_ACCEL, _HAS_NUMBA
    print(f"Cython acceleration: {'enabled' if _USE_ACCEL else 'disabled (run setup_accel.py build_ext --inplace)'}")
    if _USE_ACCEL:
        print("Threat planes: Cython (~3μs)")
    elif _HAS_NUMBA:
        print("Threat planes: numba (~3μs, JIT warmup on first call)")
    else:
        print("Threat planes: numpy fallback (~150μs, rebuild Cython or pip install numba)")
    print(f"Eval: diagnostic vs t-1 every {SAVE_INTERVAL}g ({DIAG_EVAL_SIMS} sims), "
          f"promotion vs best after {PROMOTE_CONSEC_NEEDED}× Black≥{PROMOTE_DIAG_BLACK_PCT:.0f}% "
          f"({PROMOTE_EVAL_SIMS} sims, per-color Wilson CI)")
    print(f"Concurrent games: {CONCURRENT_GAMES}, target: {NUM_GAMES}")
    print(f"Best-opponent: {BEST_PLAY_FRAC:.0%} of games vs best checkpoint")
    print(f"Noise: {NOISE_FRAC_START:.2f}→{NOISE_FRAC_END:.2f} over {NOISE_DECAY_GAMES}g")
    print(f"Replay buffer: {REPLAY_SIZE:,} positions, batch size: {BATCH_SIZE}, "
          f"recent bias: {REPLAY_RECENT_FRAC:.0%} from last {REPLAY_RECENT_SIZE:,}")
    print(f"Loss: policy + {VALUE_LOSS_COEFF}×value")
    if lr_scheduler._warmup_done:
        print(f"LR: {lr_scheduler.current_lr:.2e} (tracks P-H gap, "
              f"patience={lr_scheduler.patience}, factor={lr_scheduler.factor}, "
              f"floor={lr_scheduler.min_lr:.0e})")
    else:
        print(f"LR: {lr_scheduler.current_lr:.2e} → {lr_scheduler._target_lr:.2e} "
              f"(warmup {lr_scheduler.warmup} batches, then tracks P-H gap, "
              f"patience={lr_scheduler.patience}, factor={lr_scheduler.factor}, "
              f"floor={lr_scheduler.min_lr:.0e})")
    print()

    plateau = PlateauDetector(
        window=80,
        check_every=10,
        warmup_batches=500,         # ~500 games before detection starts
        slope_rel_thresh=3e-4,
        v_improve_rel=5e-4,
        min_replay=20_000,
        cooldown=50,
        persist=3,
    )

    # Eval: fixed opening book persisted to disk
    full_openings = load_or_create_openings(
        max(PROMOTE_EVAL_OPENINGS, 150))
    diag_openings = full_openings[:INLINE_EVAL_OPENINGS]
    # Promotion samples from full pool each time (see _run_promotion_eval)

    # Training: book openings for plausible midgame starts
    try:
        from book_openings import get_book_openings
        book_openings = get_book_openings()
        print(f"Book openings: {len(book_openings)} positions "
              f"({BOOK_OPENING_FRAC:.0%} of games)")
    except ImportError:
        book_openings = []
        print("Book openings: not available (book_openings.py missing)")

    best_state = _load_best_state()
    if best_state["path"]:
        elo_str = (f"Elo Δ{best_state['elo_vs_long']:+.0f}"
                   if best_state["elo_vs_long"] is not None else "no Elo")
        print(f"Current best: g{best_state['game_count']:05d} ({elo_str})")
    else:
        print("No best checkpoint yet — first promotion after initial evals.")

    # Reusable opponent model for eval (swap weights per matchup)
    opp_model = create_model()

    # Best-opponent model for self-play (if best checkpoint exists)
    best_model = None
    best_predict_fn = None
    if os.path.exists(BEST_WEIGHTS_FILE):
        best_model = create_model()
        best_model.load_weights(BEST_WEIGHTS_FILE)
        best_predict_fn = make_predict_fn(best_model)
        best_predict_fn(np.zeros((1, BOARD_SIZE, BOARD_SIZE, NUM_INPUT_PLANES),
                                 dtype=np.float32))
        print(f"Best-opponent loaded from {BEST_WEIGHTS_FILE} "
              f"(g{best_state.get('game_count', '?')})")
    else:
        print(f"No best checkpoint — vs-best games disabled until first promotion")
    print()

    game_count = starting_game
    target = starting_game + NUM_GAMES
    interrupted = False
    consec_strong_diags = 0  # consecutive diagnostics above win% threshold
    emerg_lr_fired = False   # emergency LR cut: fires once per regression episode
    consec_mild_regress = 0  # consecutive diagnostics in 30-40% range

    # Warm up TF graph with compiled predict function.
    # input_signature=[None, ...] traces once for all batch sizes.
    predict_fn = make_predict_fn(model)
    predict_fn(np.zeros((1, BOARD_SIZE, BOARD_SIZE, NUM_INPUT_PLANES), dtype=np.float32))

    elapsed = time.time() - t_start
    print(f"Ready. (startup took {elapsed:.0f}s)\n")

    try:
        while game_count < target:
            # ── Self-play batch ─────────────────────────────────────────
            t0 = time.time()
            batch_games = min(CONCURRENT_GAMES, target - game_count)

            # Asymmetry fraction decays linearly over training
            progress = min(1.0, game_count / ASYM_DECAY_GAMES)
            asym_frac = ASYM_FRAC_START + (ASYM_FRAC_END - ASYM_FRAC_START) * progress

            # Noise fraction anneals down as policy matures
            noise_progress = min(1.0, game_count / NOISE_DECAY_GAMES)
            noise_frac = NOISE_FRAC_START + (NOISE_FRAC_END - NOISE_FRAC_START) * noise_progress

            # Build per-game configs
            game_configs = []
            n_asym = 0
            n_vs_best = 0
            for _ in range(batch_games):
                # Choose opening: book or random
                if book_openings and np.random.random() < BOOK_OPENING_FRAC:
                    opening = book_openings[
                        np.random.randint(len(book_openings))]
                else:
                    opening = RANDOM_OPENING_PLIES  # int → random plies

                # vs-best games: play against best checkpoint
                if best_predict_fn is not None and np.random.random() < BEST_PLAY_FRAC:
                    game_configs.append((MCTS_SIMS, MCTS_SIMS,
                                         None, opening, True))
                    n_vs_best += 1
                elif np.random.random() < asym_frac:
                    weak = PLAYER1 if np.random.random() < 0.5 else -PLAYER1
                    game_configs.append((MCTS_SIMS, ASYM_WEAK_SIMS,
                                         weak, opening, False))
                    n_asym += 1
                else:
                    game_configs.append((MCTS_SIMS, MCTS_SIMS,
                                         None, opening, False))

            results = _play_games_interleaved(predict_fn, game_configs,
                                              best_predict_fn, noise_frac)
            sp_time = time.time() - t0

            new_positions = 0
            move_counts = []
            n_resigned = 0

            for trajectory, winner, resigned in results:
                if not trajectory:
                    continue
                if resigned:
                    n_resigned += 1
                move_counts.append(len(trajectory))
                for state, pi, player, move_sims, is_full in trajectory:
                    if winner == 0:
                        outcome = 0.0
                    elif player == winner:
                        outcome = 1.0
                    else:
                        outcome = -1.0
                    # Full searches train policy + value; fast only value
                    if is_full:
                        pi_w = np.float32(np.sqrt(move_sims / MCTS_SIMS))
                    else:
                        pi_w = np.float32(0.0)
                    replay.append((state, pi, np.float32(outcome), pi_w))
                    new_positions += 1
                game_count += 1
                if game_count >= target:
                    break

            # ── Training ────────────────────────────────────────────────
            t1 = time.time()
            n_steps = max(1, int(new_positions * TRAIN_STEPS_RATIO / BATCH_SIZE))
            total_ploss = total_vloss = 0.0
            total_ent = 0.0
            ent_steps = 0

            did_train = False
            total_mean_w = 0.0
            if len(replay) >= BATCH_SIZE:
                did_train = True
                for _ in range(n_steps):
                    idxs = _sample_replay(replay, BATCH_SIZE)
                    batch = [replay[i] for i in idxs]
                    s = np.array([b[0] for b in batch])
                    p = np.array([b[1] for b in batch])
                    v = np.array([b[2] for b in batch])
                    w = np.array([b[3] for b in batch])
                    s, p = _augment_batch(s, p)
                    eps = 1e-12
                    ent = -np.mean(np.sum(p * np.log(p + eps), axis=1))
                    total_ent += float(ent)
                    total_mean_w += float(w.mean())
                    ent_steps += 1

                    pl, vl, _ = _train_step(model, optimizer, s, p, v, w)
                    total_ploss += float(pl)
                    total_vloss += float(vl)
                total_ploss /= n_steps
                total_vloss /= n_steps
                total_mean_w /= n_steps
                if ent_steps:
                    total_ent /= ent_steps
                else:
                    total_ent = 0.0

            train_time = time.time() - t1

            avg_moves = float(np.mean(move_counts)) if move_counts else 0.0
            kept = [(traj, wn, res) for (traj, wn, res) in results if traj]
            kept_n = len(kept)
            n_decisive = sum(1 for (_, wn, _) in kept if wn != 0)
            decisive_rate = n_decisive / max(1, kept_n)

            resign_str = f"  SR {n_resigned}" if n_resigned else ""
            best_str = f"  B {n_vs_best}" if n_vs_best else ""
            lr_str = f"  lr {lr_scheduler.current_lr:.1e}" if lr_scheduler._reductions > 0 else ""
            print(
                f"Game {game_count:5d}/{target} | "
                f"Asym {n_asym}/{batch_games} ({asym_frac:.0%}) | "
                f"Moves {avg_moves:5.1f} | "
                f"Win {n_decisive}/{kept_n}{resign_str}{best_str} | "
                f"P {total_ploss:.4f} V {total_vloss:.4f} "
                f"w̄ {total_mean_w:.2f} H {total_ent:.2f} "
                f"gap {total_ploss - total_ent:.3f} | "
                f"Buf {len(replay):6d} | "
                f"SP {sp_time:.1f}s  Tr {train_time:.1f}s{lr_str}"
            , flush=True)
            if did_train:
                rec = plateau.update(
                    ploss=total_ploss,
                    vloss=total_vloss,
                    target_entropy=total_ent,
                    decisive_rate=decisive_rate,
                    replay_len=len(replay),
                )
                if rec:
                    print("  " + rec, flush=True)
                lr_msg = lr_scheduler.update(total_ploss, total_ent)
                if lr_msg:
                    print("  " + lr_msg, flush=True)

            # ── Checkpoint ──────────────────────────────────────────────
            prev_cp = (game_count - len(move_counts)) // SAVE_INTERVAL
            curr_cp = game_count // SAVE_INTERVAL
            if curr_cp > prev_cp:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                cp_file = f"weights/gomoku_{ts}_g{game_count:05d}.weights.h5"
                model.save_weights(cp_file)
                model.save_weights(weights_file)
                _save_training_state(game_count, optimizer, replay, lr_scheduler)
                print(f"  → Checkpoint {cp_file}", flush=True)

                # Auto-promote first checkpoint as best so vs-best can start
                if not os.path.exists(BEST_WEIGHTS_FILE):
                    shutil.copy2(cp_file, BEST_WEIGHTS_FILE)
                    best_state = {
                        "path": cp_file,
                        "game_count": game_count,
                        "elo_vs_long": None,
                    }
                    _save_best_state(best_state)
                    if best_model is None:
                        best_model = create_model()
                    best_model.load_weights(BEST_WEIGHTS_FILE)
                    best_predict_fn = make_predict_fn(best_model)
                    best_predict_fn(np.zeros(
                        (1, BOARD_SIZE, BOARD_SIZE, NUM_INPUT_PLANES),
                        dtype=np.float32))
                    print(f"  ★ Auto-promoted g{game_count} as first best",
                          flush=True)

                # Diagnostic: lightweight t-1 eval every checkpoint
                diag_result = _run_diagnostic_eval(
                    model, game_count, diag_openings, opp_model)

                # Extract per-color metrics
                diag_black_pct = (diag_result["black_win_pct"]
                                  if diag_result else None)

                # Emergency LR drop on eval regression (per Black win%)
                # Black should win 90%+ vs t-1; dropping means regression.
                # (skip early: random-vs-random evals are meaningless)
                if (diag_black_pct is not None and not emerg_lr_fired
                        and game_count >= EMERG_LR_MIN_GAMES):
                    if diag_black_pct <= EMERG_LR_SEVERE_PCT:
                        # Severe regression: can't even win as Black
                        old_lr = lr_scheduler.current_lr
                        new_lr = max(old_lr * EMERG_LR_FACTOR, lr_scheduler.min_lr)
                        lr_scheduler.optimizer.learning_rate.assign(new_lr)
                        lr_scheduler._warmup_done = True
                        lr_scheduler._target_lr = new_lr
                        if lr_scheduler._gap_buf:
                            lr_scheduler._best_avg = float(np.mean(lr_scheduler._gap_buf))
                        lr_scheduler._checks_without_improvement = 0
                        lr_scheduler._reductions += 1
                        emerg_lr_fired = True
                        consec_mild_regress = 0
                        print(f"  ⚠ Emergency LR cut: {old_lr:.2e} → {new_lr:.2e} "
                              f"(Black {diag_black_pct:.0f}% ≤ {EMERG_LR_SEVERE_PCT:.0f}%)",
                              flush=True)
                    elif diag_black_pct < EMERG_LR_BLACK_PCT:
                        # Mild regression: need 2 consecutive
                        consec_mild_regress += 1
                        if consec_mild_regress >= 2:
                            old_lr = lr_scheduler.current_lr
                            new_lr = max(old_lr * EMERG_LR_FACTOR, lr_scheduler.min_lr)
                            lr_scheduler.optimizer.learning_rate.assign(new_lr)
                            lr_scheduler._warmup_done = True
                            lr_scheduler._target_lr = new_lr
                            if lr_scheduler._gap_buf:
                                lr_scheduler._best_avg = float(np.mean(lr_scheduler._gap_buf))
                            lr_scheduler._checks_without_improvement = 0
                            lr_scheduler._reductions += 1
                            emerg_lr_fired = True
                            consec_mild_regress = 0
                            print(f"  ⚠ Emergency LR cut: {old_lr:.2e} → {new_lr:.2e} "
                                  f"(Black <{EMERG_LR_BLACK_PCT:.0f}% for 2 checkpoints)",
                                  flush=True)
                    else:
                        consec_mild_regress = 0
                if (diag_black_pct is not None
                        and diag_black_pct >= PROMOTE_DIAG_BLACK_PCT):
                    emerg_lr_fired = False
                    consec_mild_regress = 0

                # Track consecutive strong diagnostics (Black win% based)
                if (diag_black_pct is not None
                        and diag_black_pct >= PROMOTE_DIAG_BLACK_PCT):
                    consec_strong_diags += 1
                else:
                    consec_strong_diags = 0

                # Promotion: if N consecutive strong diagnostics
                if consec_strong_diags >= PROMOTE_CONSEC_NEEDED:
                    old_best_gc = best_state.get("game_count", 0)
                    best_state = _run_promotion_eval(
                        model, game_count, full_openings,
                        best_state, opp_model)
                    consec_strong_diags = 0  # reset after attempt

                    # Reload best model if promotion happened
                    if best_state.get("game_count", 0) != old_best_gc:
                        if best_model is None:
                            best_model = create_model()
                        best_model.load_weights(BEST_WEIGHTS_FILE)
                        best_predict_fn = make_predict_fn(best_model)
                        best_predict_fn(np.zeros(
                            (1, BOARD_SIZE, BOARD_SIZE, NUM_INPUT_PLANES),
                            dtype=np.float32))

    except KeyboardInterrupt:
        interrupted = True
        print("\n\n⚠ Interrupted — saving training state …", flush=True)

    # ── Cleanup & final save ────────────────────────────────────────────
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "interrupted" if interrupted else "final"
    model.save_weights(f"weights/gomoku_{ts}_{tag}.weights.h5")
    model.save_weights(weights_file)

    _save_training_state(game_count, optimizer, replay, lr_scheduler)

    if interrupted:
        print(f"\n✓ State saved at game {game_count}.  Resume with: python train.py")
    else:
        print(f"\n✓ Training complete — {game_count} games.  Weights → {weights_file}")

    if best_state.get("path"):
        elo_str = (f"Elo Δ{best_state['elo_vs_long']:+.0f}"
                   if best_state.get("elo_vs_long") is not None else "no Elo")
        print(f"  Best checkpoint: g{best_state['game_count']:05d} "
              f"({elo_str})  → {BEST_WEIGHTS_FILE}")

    elapsed = time.time() - t_start
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    print(f"  Total time: {h}h {m}m {s}s")


if __name__ == "__main__":
    main()