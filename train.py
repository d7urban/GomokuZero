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
  • Inline evaluation at a separate cadence from checkpoint saves.
"""

import numpy as np
import math
import os, pickle, time, signal, shutil
from datetime import datetime
from collections import deque

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf
from tensorflow import keras

tf.get_logger().setLevel("ERROR")

_gpus = tf.config.list_physical_devices("GPU")

from gomoku import (
    BOARD_SIZE, PLAYER1, NUM_INPUT_PLANES,
    GomokuGame, create_model, make_predict_fn,
    mcts_policy, select_action,
    mcts_begin, mcts_expand_root, mcts_select_leaves, mcts_process_results,
)
from book_openings import load_or_create_openings

from eval import (
    find_checkpoints,
    run_match_sequential,
)
from ratings_glicko2 import (
    apply_match_glicko2_update,
    format_glicko2_entry,
    get_glicko2_entry,
    load_glicko2_ratings,
    save_glicko2_ratings,
)


def _env_int(name, default, min_value=1):
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        val = int(raw)
    except Exception:
        return int(default)
    return max(int(min_value), val)


def _env_float(name, default, min_value=0.0):
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        val = float(raw)
    except Exception:
        return float(default)
    return max(float(min_value), val)


# ── Tunables ────────────────────────────────────────────────────────────────
CONCURRENT_GAMES  = _env_int("GZ_CONCURRENT_GAMES", 24)  # interleaved games per batch
NUM_GAMES         = _env_int("GZ_NUM_GAMES", 5000)       # total self-play games
TRAIN_NUM_RES_BLOCKS = 10        # fixed training architecture (10x128)
TRAIN_NUM_FILTERS = 128
C_PUCT            = 1.5
DIRICHLET_ALPHA   = 0.15
NOISE_FRAC_START  = 0.35         # Dirichlet noise fraction (anneals down)
NOISE_FRAC_END    = 0.25         # noise floor after NOISE_DECAY_GAMES
NOISE_DECAY_GAMES = 3000         # linear anneal over this many games
TEMP_THRESHOLD    = 40           # move number after which temperature drops
BATCH_SIZE        = _env_int("GZ_BATCH_SIZE", 1024)
REPLAY_SIZE       = 300_000      # positions — larger to reduce forgetting
REPLAY_RECENT_FRAC = 0.75       # fraction of each batch from recent positions
REPLAY_RECENT_SIZE = 50_000     # how many most-recent positions count as "recent"
TRAIN_STEPS_RATIO = _env_float(
    "GZ_TRAIN_STEPS_RATIO", 4.0, min_value=0.01
)  # gradient steps = new_positions * ratio / BATCH_SIZE
LR                = 7e-4
WEIGHT_DECAY      = 1e-4
VALUE_LOSS_COEFF  = 1.0          # weight on value loss (tune if v-loss dominates)
SAVE_INTERVAL     = 1000         # games between checkpoint saves
EVAL_INTERVAL     = 200          # games between diagnostic/promotion checks

# Batched MCTS — leaves evaluated per forward pass.
# Smaller batches = more backup rounds = better-informed UCB selection.
# GPU handles batch-8 in ~1ms so throughput is not the bottleneck.
MCTS_BATCH_SIZE   = _env_int("GZ_MCTS_BATCH_SIZE", 10)
# Value-only moves (fast/weak/resigned) can use larger leaf batches to
# reduce forward-pass rounds without affecting policy targets.
MCTS_BATCH_SIZE_FAST = _env_int("GZ_MCTS_BATCH_SIZE_FAST", 16)

# Self-play simulation budget
MCTS_SIMS         = 160          # full-search sims (for policy training)
# Fresh-run warmup: start cheaper, then switch to full budget once replay has
# enough diversity. Set warmup sims >= MCTS_SIMS or replay threshold <= 0 to disable.
MCTS_SIMS_WARMUP         = 100
MCTS_SIMS_WARMUP_REPLAY  = 50_000

# Playout cap randomization: most moves get a fast search (value only),
# some get a full search (policy + value).  This generates more games
# for the value head while preserving policy quality.
PCAP_FAST_SIMS    = 32           # sims for fast-search moves
PCAP_FULL_FRAC    = 0.50         # fraction of moves that get full search

# Asymmetric self-play: in a fraction of games, one side is randomly
# weakened to produce decisive results and train the value head.
# The fraction decays linearly so training converges to symmetric play.
ASYM_WEAK_SIMS    = 64           # sims for the weakened side
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
# 1. Diagnostic: lightweight check every EVAL_INTERVAL.
#    Uses t-1 when evaluating a freshly saved checkpoint, otherwise compares
#    current in-memory weights vs the latest saved checkpoint.
#    Tracks a rolling signal (2 strong checks in latest up-to-3 checks),
#    arming as soon as the second strong check appears.
# 2. Promotion: eval vs current best when diagnostic signal is strong.
INLINE_EVAL_OPENINGS  = 30       # 30 openings × 2 colors = 60 games (diagnostic)
PROMOTE_EVAL_OPENINGS = 60       # 60 openings × 2 colors = 120 games (promotion)
DIAG_EVAL_SIMS        = 160      # lighter than self-play but enough for signal
PROMOTE_EVAL_SIMS     = 320      # full sims for promotion decisions
PROMOTE_DIAG_CI_LEVEL = 0.90     # confidence for diagnostic Black CI signal
PROMOTE_DIAG_BLACK_LOWER = 0.50  # strong diagnostic if Black Wilson lower > this
PROMOTE_DIAG_WINDOW = 3          # rolling diagnostic history length
PROMOTE_DIAG_MIN_STRONG = 2      # require >= this many strong checks in window
EMERG_LR_BLACK_PCT    = 40.0    # Black win% below which mild regression (2 consec)
EMERG_LR_SEVERE_PCT   = 25.0   # Black win% below which severe regression (immediate)
PROMOTE_COOLDOWN      = 400      # min games between promotion evals
PROMOTE_CI_LEVEL      = 0.95     # confidence level for Wilson interval
PROMOTE_BLACK_LOWER   = 0.60     # promotion requires Black Wilson lower > this
PROMOTE_WHITE_SURVIVAL_MIN = 35.0   # avg moves survived in White losses
PROMOTE_WHITE_NON_LOSS_MIN = 0.10   # min White non-loss rate (wins+draws)
PROMOTE_WHITE_QUICK_LOSS_MAX = 85.0 # max quick-loss share among White losses
PROMOTE_WHITE_QUICK_LOSS_MIN_LOSSES = 10  # apply quick-loss gate only with enough losses
PROMOTE_WHITE_QUICK_LOSS_CHECK_NON_LOSS_MAX = 0.50  # apply quick-loss only if White non-loss is weak
EMERG_LR_FACTOR       = 0.3     # multiply LR by this on emergency
EMERG_LR_MIN_GAMES    = 1000    # don't fire emergency before this many games
INLINE_EVAL_PROMOTION_ENABLED = False  # run manual eval_tournament.py after training chunks

# Plateau auto-tuning (conservative, reversible in-run controls only)
PLATEAU_AUTOTUNE_ENABLED = True
PLATEAU_AUTOTUNE_COOLDOWN_GAMES = 800
PLATEAU_AUTOTUNE_LR_FACTOR = 0.80
PLATEAU_AUTOTUNE_STALEMATE_ASYM_BOOST = 0.10
PLATEAU_AUTOTUNE_STALEMATE_NOISE_BOOST = 0.05
PLATEAU_AUTOTUNE_STALEMATE_DURATION_GAMES = 400

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
        self._last_event = None

    def _rel_slope(self, y):
        n = len(y)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        m = np.polyfit(x, y, 1)[0]
        mean_abs = np.mean(np.abs(y)) + 1e-12
        return m / mean_abs

    def _append_metrics(self, ploss, vloss, target_entropy, decisive_rate):
        self._ploss.append(ploss)
        self._vloss.append(vloss)
        self._ent.append(target_entropy)
        self._dec.append(decisive_rate)
        for buf in (self._ploss, self._vloss, self._ent, self._dec):
            if len(buf) > self.window:
                buf.pop(0)
        self.batches += 1

    def _ready_to_check(self, replay_len):
        if self.batches < self.warmup_batches:
            return False
        if self.batches % self.check_every != 0:
            return False
        if replay_len < self.min_replay:
            return False
        if len(self._ploss) < self.window:
            return False
        return True

    def _compute_plateau_stats(self):
        pl_s = self._rel_slope(self._ploss)
        vl_s = self._rel_slope(self._vloss)
        ent_s = self._rel_slope(self._ent)
        dec_s = self._rel_slope(self._dec)
        return {
            "pl_s": pl_s,
            "vl_s": vl_s,
            "ent_s": ent_s,
            "dec_s": dec_s,
            "decisive_now": self._dec[-1] if self._dec else 0.0,
            "ent_now": self._ent[-1] if self._ent else 0.0,
        }

    def _classify_plateau(self, stats):
        thr = self.slope_rel_thresh
        v_thr = self.v_improve_rel

        p_flat = abs(stats["pl_s"]) < thr
        v_flat = abs(stats["vl_s"]) < thr
        v_improving = stats["vl_s"] < -v_thr
        ent_flat = abs(stats["ent_s"]) < thr
        dec_flat = abs(stats["dec_s"]) < thr and stats["decisive_now"] < 0.9
        dec_rising = stats["dec_s"] > thr

        if p_flat and v_improving and dec_flat:
            return (
                "capacity",
                "[CAPACITY WARNING] PLATEAU: policy flat, value improving, "
                "decisive rate stable → likely policy-head capacity limit. "
                "Consider more res-blocks/filters.",
                False,
            )
        if p_flat and v_improving and dec_rising:
            return None, None, True
        if p_flat and v_flat and ent_flat and stats["decisive_now"] < 0.3:
            return (
                "stalemate",
                "PLATEAU: all flat, low decisive rate "
                "→ defensive stalemate. Increase asymmetry or exploration.",
                False,
            )
        if p_flat and v_flat and stats["ent_s"] < -thr:
            return (
                "mcts_sharp",
                "PLATEAU: losses flat, entropy falling "
                "→ MCTS sharpening but net can't follow. Lower LR or add capacity.",
                False,
            )
        if p_flat and stats["ent_s"] > thr:
            return (
                "ent_rising",
                "PLATEAU: policy flat, entropy rising "
                "→ targets diversifying. Lower LR or add capacity.",
                False,
            )
        if p_flat and v_flat:
            return (
                "generic",
                "PLATEAU: policy and value both flat "
                "→ generic stall. Check data quality, LR, exploration.",
                False,
            )
        return None, None, True

    def _plateau_emit_ready(self, cond_key):
        self._persist_counts.setdefault(cond_key, 0)
        self._persist_counts[cond_key] += 1
        for k in list(self._persist_counts):
            if k != cond_key:
                del self._persist_counts[k]
        if self._persist_counts[cond_key] < self.persist:
            return False
        if (self.batches - self._last_emit) < self.cooldown:
            return False
        self._last_emit = self.batches
        return True

    def _format_plateau_message(self, msg, stats):
        return (
            msg
            + f"  [pl_s={stats['pl_s']:+.2e}, vl_s={stats['vl_s']:+.2e}, "
            f"ent_s={stats['ent_s']:+.2e}, dec_s={stats['dec_s']:+.2e}, "
            f"entropy={stats['ent_now']:.2f}, decisive={stats['decisive_now']:.2f}]"
        )

    def last_event(self):
        return self._last_event

    def update(self, ploss, vloss, target_entropy, decisive_rate, replay_len):
        self._last_event = None
        self._append_metrics(ploss, vloss, target_entropy, decisive_rate)
        if not self._ready_to_check(replay_len):
            return None

        stats = self._compute_plateau_stats()
        cond_key, msg, clear_counts = self._classify_plateau(stats)
        if clear_counts:
            self._persist_counts.clear()
            return None

        if not self._plateau_emit_ready(cond_key):
            return None

        full_msg = self._format_plateau_message(msg, stats)
        self._last_event = {
            "cond_key": cond_key,
            "message": full_msg,
            "stats": stats,
            "batches": self.batches,
        }
        return full_msg


class PlateauAutoTuner:
    """Apply conservative reversible changes in response to plateau events."""

    def __init__(
        self,
        lr_scheduler,
        enabled=PLATEAU_AUTOTUNE_ENABLED,
        cooldown_games=PLATEAU_AUTOTUNE_COOLDOWN_GAMES,
        lr_factor=PLATEAU_AUTOTUNE_LR_FACTOR,
        stalemate_asym_boost=PLATEAU_AUTOTUNE_STALEMATE_ASYM_BOOST,
        stalemate_noise_boost=PLATEAU_AUTOTUNE_STALEMATE_NOISE_BOOST,
        stalemate_duration_games=PLATEAU_AUTOTUNE_STALEMATE_DURATION_GAMES,
    ):
        self.lr_scheduler = lr_scheduler
        self.enabled = bool(enabled)
        self.cooldown_games = int(cooldown_games)
        self.lr_factor = float(lr_factor)
        self.stalemate_asym_boost = float(stalemate_asym_boost)
        self.stalemate_noise_boost = float(stalemate_noise_boost)
        self.stalemate_duration_games = int(stalemate_duration_games)
        self._last_action_game = -10**9

    def _in_cooldown(self, game_count):
        return (game_count - self._last_action_game) < self.cooldown_games

    def _apply_lr_reduction(self, game_count, cond_key):
        old_lr = self.lr_scheduler.current_lr
        if old_lr <= self.lr_scheduler.min_lr:
            return (f"[AutoTune:{cond_key}] skipped LR step-down: already at floor "
                    f"{old_lr:.0e}")

        new_lr = max(old_lr * self.lr_factor, self.lr_scheduler.min_lr)
        if new_lr >= old_lr:
            return (f"[AutoTune:{cond_key}] skipped LR step-down: no smaller LR "
                    f"available ({old_lr:.2e})")

        self.lr_scheduler.optimizer.learning_rate.assign(new_lr)
        self.lr_scheduler._warmup_done = True
        self.lr_scheduler._target_lr = new_lr
        if self.lr_scheduler._gap_buf:
            self.lr_scheduler._best_avg = float(np.mean(self.lr_scheduler._gap_buf))
        self.lr_scheduler._checks_without_improvement = 0
        self.lr_scheduler._reductions += 1
        self._last_action_game = game_count
        return (f"[AutoTune:{cond_key}] LR step-down: {old_lr:.2e} -> {new_lr:.2e} "
                f"(cooldown {self.cooldown_games}g)")

    def _apply_stalemate_boost(self, game_count, loop_state):
        until = game_count + self.stalemate_duration_games
        loop_state["autotune_asym_boost_until"] = max(
            int(loop_state.get("autotune_asym_boost_until", 0)),
            until,
        )
        loop_state["autotune_asym_boost_value"] = max(
            float(loop_state.get("autotune_asym_boost_value", 0.0)),
            self.stalemate_asym_boost,
        )
        loop_state["autotune_noise_boost_until"] = max(
            int(loop_state.get("autotune_noise_boost_until", 0)),
            until,
        )
        loop_state["autotune_noise_boost_value"] = max(
            float(loop_state.get("autotune_noise_boost_value", 0.0)),
            self.stalemate_noise_boost,
        )
        self._last_action_game = game_count
        return (f"[AutoTune:stalemate] temporary exploration boost for "
                f"{self.stalemate_duration_games}g "
                f"(asym +{100.0 * self.stalemate_asym_boost:.0f}%, "
                f"noise +{self.stalemate_noise_boost:.2f})")

    def on_plateau(self, event, game_count, loop_state):
        if not self.enabled or not event:
            return None
        cond_key = event.get("cond_key")
        if not cond_key:
            return None

        if cond_key == "capacity":
            return None

        if self._in_cooldown(game_count):
            return None

        if cond_key in ("mcts_sharp", "ent_rising", "generic"):
            return self._apply_lr_reduction(game_count, cond_key)
        if cond_key == "stalemate":
            return self._apply_stalemate_boost(game_count, loop_state)
        return None


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
                 patience=7, factor=0.5, min_lr=3e-6, warmup=200,
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
def _augment_apply_d4(states, policies, board_size):
    k = np.random.randint(8)
    s, p = states, policies
    if k <= 0:
        return s, p

    rot = k % 4
    flip = k >= 4
    if flip:
        s = s[:, :, ::-1, :]
    if rot:
        s = np.rot90(s, rot, axes=(1, 2))

    p2 = p.reshape(-1, board_size, board_size)
    if flip:
        p2 = p2[:, :, ::-1]
    if rot:
        p2 = np.rot90(p2, rot, axes=(1, 2))
    p = p2.reshape(-1, board_size * board_size)
    return np.ascontiguousarray(s), np.ascontiguousarray(p)


def _augment_translate_one(s, p2d, s_out, p_out, idx, board_size, margin):
    occ = (s[idx, :, :, 0] != 0) | (s[idx, :, :, 1] != 0)
    rows, cols = np.where(occ)
    if len(rows) == 0:
        s_out[idx] = s[idx]
        p_out[idx] = p2d[idx]
        return

    r0 = max(0, int(rows.min()) - margin)
    r1 = min(board_size - 1, int(rows.max()) + margin)
    c0 = max(0, int(cols.min()) - margin)
    c1 = min(board_size - 1, int(cols.max()) + margin)

    dr_lo, dr_hi = -r0, board_size - 1 - r1
    dc_lo, dc_hi = -c0, board_size - 1 - c1
    if dr_lo >= dr_hi and dc_lo >= dc_hi:
        s_out[idx] = s[idx]
        p_out[idx] = p2d[idx]
        return

    dr = np.random.randint(dr_lo, dr_hi + 1)
    dc = np.random.randint(dc_lo, dc_hi + 1)
    if dr == 0 and dc == 0:
        s_out[idx] = s[idx]
        p_out[idx] = p2d[idx]
        return

    sr = slice(max(0, -dr), min(board_size, board_size - dr))
    sc = slice(max(0, -dc), min(board_size, board_size - dc))
    dr_s = slice(max(0, dr), min(board_size, board_size + dr))
    dc_s = slice(max(0, dc), min(board_size, board_size + dc))
    s_out[idx, dr_s, dc_s, :] = s[idx, sr, sc, :]
    p_out[idx, dr_s, dc_s] = p2d[idx, sr, sc]


def _augment_translate_batch(states, policies, board_size, margin=3):
    batch_n = states.shape[0]
    s_out = np.zeros_like(states)
    p2d = policies.reshape(batch_n, board_size, board_size)
    p_out = np.zeros((batch_n, board_size, board_size), dtype=policies.dtype)
    for idx in range(batch_n):
        _augment_translate_one(
            states, p2d, s_out, p_out, idx, board_size, margin
        )
    return s_out, p_out


def _augment_renormalize_policy(policies_2d, board_size):
    batch_n = policies_2d.shape[0]
    p_flat = policies_2d.reshape(batch_n, board_size * board_size)
    p_sums = p_flat.sum(axis=1, keepdims=True)
    p_sums = np.maximum(p_sums, 1e-8)
    p_flat /= p_sums
    return p_flat


def _augment_batch(states, policies):
    """Apply random D4 symmetry + random translation to each batch."""
    board_size = BOARD_SIZE
    s, p = _augment_apply_d4(states, policies, board_size)
    s_out, p_out = _augment_translate_batch(s, p, board_size, margin=3)
    return s_out, _augment_renormalize_policy(p_out, board_size)


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


class ReplayBuffer:
    """Fixed-capacity ring buffer with O(1) append and indexing."""

    def __init__(self, capacity):
        self.capacity = max(1, int(capacity))
        self._data = [None] * self.capacity
        self._start = 0
        self._size = 0

    def __len__(self):
        return self._size

    def append(self, item):
        if self._size < self.capacity:
            idx = (self._start + self._size) % self.capacity
            self._data[idx] = item
            self._size += 1
            return
        self._data[self._start] = item
        self._start = (self._start + 1) % self.capacity

    def __getitem__(self, index):
        i = int(index)
        if i < 0:
            i += self._size
        if i < 0 or i >= self._size:
            raise IndexError(index)
        return self._data[(self._start + i) % self.capacity]

    def __iter__(self):
        for i in range(self._size):
            yield self._data[(self._start + i) % self.capacity]

    def to_list(self):
        if self._size == 0:
            return []
        end = self._start + self._size
        if end <= self.capacity:
            return list(self._data[self._start:end])
        split = self.capacity - self._start
        return list(self._data[self._start:]) + list(self._data[:self._size - split])


def _train_arch_label():
    return f"{TRAIN_NUM_RES_BLOCKS}x{TRAIN_NUM_FILTERS}"


def _create_train_model():
    return create_model(
        num_res_blocks=TRAIN_NUM_RES_BLOCKS,
        num_filters=TRAIN_NUM_FILTERS,
    )


def _weight_error_reason(exc):
    text = str(exc)
    return text.splitlines()[0] if text else type(exc).__name__


def _load_weights_strict(model, weight_path, role):
    try:
        model.load_weights(weight_path)
    except Exception as e:
        reason = _weight_error_reason(e)
        raise ValueError(
            f"Failed to load {role} weights from '{weight_path}' into "
            f"{_train_arch_label()} model: {reason}. "
            f"Training now expects only {_train_arch_label()} checkpoints."
        ) from e


def _load_weights_optional(model, weight_path, role):
    try:
        model.load_weights(weight_path)
        return True
    except Exception as e:
        reason = _weight_error_reason(e)
        print(
            f"  Skipping {role} '{weight_path}': incompatible with "
            f"{_train_arch_label()} ({reason})",
            flush=True,
        )
        return False


def _get_optimizer_state(optimizer):
    if hasattr(optimizer, "get_weights"):
        return optimizer.get_weights()
    return [v.numpy() for v in optimizer.variables]


def _set_optimizer_state(optimizer, state):
    if hasattr(optimizer, "set_weights"):
        optimizer.set_weights(state)
    else:
        for var, val in zip(optimizer.variables, state, strict=False):
            var.assign(val)


def _save_training_state(game_count, optimizer, replay, lr_scheduler=None,
                         eval_state=None):
    """Persist everything needed for a seamless resume."""
    opt_weights = _get_optimizer_state(optimizer)

    state = {
        "board_size": BOARD_SIZE,
        "total_games": game_count,
        "optimizer_weights": opt_weights,
    }
    if lr_scheduler is not None:
        state["lr_scheduler"] = lr_scheduler.get_state()
    if eval_state is not None:
        state["eval_state"] = eval_state

    with open(_STATE_PATH, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(_REPLAY_PATH, "wb") as f:
        if hasattr(replay, "to_list"):
            replay_dump = replay.to_list()
        else:
            replay_dump = list(replay)
        pickle.dump(replay_dump, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open("weights/model_config.pkl", "wb") as f:
        pickle.dump(
            {
                "board_size": BOARD_SIZE,
                "total_games": game_count,
                "num_res_blocks": TRAIN_NUM_RES_BLOCKS,
                "num_filters": TRAIN_NUM_FILTERS,
            },
            f,
        )


def _load_training_state(model, optimizer, replay):
    """Restore optimizer/replay state. Returns (starting_game, lr_state, eval_state)."""
    starting_game = 0
    lr_state = None
    eval_state = None

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
            eval_state = cfg.get("eval_state")
            if opt_weights is not None:
                saved_w = [w.numpy() for w in model.trainable_variables]
                dummy_s = np.zeros((1, BOARD_SIZE, BOARD_SIZE, NUM_INPUT_PLANES), dtype=np.float32)
                dummy_p = np.zeros((1, BOARD_SIZE * BOARD_SIZE), dtype=np.float32)
                dummy_p[0, 0] = 1.0
                dummy_v = np.zeros((1,), dtype=np.float32)
                dummy_w = np.ones((1,), dtype=np.float32)
                _train_step(model, optimizer, dummy_s, dummy_p, dummy_v, dummy_w)
                for var, w in zip(model.trainable_variables, saved_w, strict=False):
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

    return starting_game, lr_state, eval_state


# ── Interleaved self-play (multiple games, shared GPU inference) ────────────
class _InterleavedGame:
    __slots__ = (
        "game", "sims", "fast_sims", "weak_sims", "weak_player",
        "trajectory", "winner", "finished", "move_num",
        "ctx", "root_state", "phase", "move_sims",
        "resign_count", "vs_best", "training_player",
        "is_full_search", "soft_resigned",
    )


_ZERO_POLICY = np.zeros((BOARD_SIZE * BOARD_SIZE,), dtype=np.float32)


def _interleaved_is_best_turn(game_state):
    """True if the best model should play this move."""
    return game_state.vs_best and game_state.game.current_player != game_state.training_player


def _apply_interleaved_opening(game_state, opening):
    if isinstance(opening, list):
        for row, col in opening:
            _, done = game_state.game.make_move(row, col)
            if done:
                game_state.finished = True
                break
        return

    for _ in range(opening):
        moves = game_state.game.get_valid_moves()
        if not moves:
            game_state.finished = True
            break
        row, col = moves[np.random.randint(len(moves))]
        _, done = game_state.game.make_move(row, col)
        if done:
            game_state.finished = True
            break


def _init_interleaved_game(cfg, has_best_predict):
    sims, fast_sims, weak_sims, weak_player, opening, vs_best = cfg
    game_state = _InterleavedGame()
    game_state.game = GomokuGame()
    game_state.sims = sims
    game_state.fast_sims = fast_sims
    game_state.weak_sims = weak_sims
    game_state.weak_player = weak_player
    game_state.trajectory = []
    game_state.winner = 0
    game_state.finished = False
    game_state.ctx = None
    game_state.root_state = None
    game_state.phase = "new_move"
    game_state.move_sims = sims
    game_state.move_num = 0
    game_state.resign_count = 0
    game_state.is_full_search = True
    game_state.soft_resigned = False
    game_state.vs_best = vs_best and has_best_predict
    game_state.training_player = (
        PLAYER1 if np.random.random() < 0.5 else -PLAYER1
    ) if game_state.vs_best else None

    _apply_interleaved_opening(game_state, opening)
    game_state.move_num = len(game_state.game.move_history)
    return game_state


def _set_interleaved_move_budget(game_state, is_best_turn):
    if is_best_turn:
        game_state.move_sims = game_state.sims
        game_state.is_full_search = True
        return

    if game_state.soft_resigned:
        game_state.move_sims = game_state.fast_sims
        game_state.is_full_search = False
        return

    if (game_state.weak_player is not None and
            game_state.game.current_player == game_state.weak_player):
        game_state.move_sims = game_state.weak_sims
        game_state.is_full_search = False
        return

    game_state.is_full_search = np.random.random() < PCAP_FULL_FRAC
    game_state.move_sims = (
        game_state.sims if game_state.is_full_search else game_state.fast_sims
    )


def _queue_interleaved_new_move(game_state, bundle, noise_frac, is_best_turn):
    _set_interleaved_move_budget(game_state, is_best_turn)
    add_noise = (not is_best_turn) and game_state.is_full_search
    batch_size = MCTS_BATCH_SIZE if game_state.is_full_search else MCTS_BATCH_SIZE_FAST
    game_state.ctx, game_state.root_state = mcts_begin(
        game_state.game,
        num_simulations=game_state.move_sims,
        batch_size=batch_size,
        c_puct=C_PUCT,
        add_noise=add_noise,
        dirichlet_alpha=DIRICHLET_ALPHA,
        noise_frac=noise_frac,
    )
    idx = len(bundle["states"])
    bundle["states"].append(game_state.root_state)
    bundle["root_evals"].append((game_state, idx))
    game_state.phase = "root_pending"


def _queue_interleaved_search_leaves(game_state, bundle):
    leaf_states = mcts_select_leaves(game_state.ctx)
    if leaf_states:
        start = len(bundle["states"])
        bundle["states"].extend(leaf_states)
        bundle["leaf_evals"].append((game_state, start, len(leaf_states)))
    else:
        mcts_process_results(game_state.ctx)


def _collect_interleaved_eval_bundles(active_games, noise_frac):
    train_bundle = {"states": [], "root_evals": [], "leaf_evals": []}
    best_bundle = {"states": [], "root_evals": [], "leaf_evals": []}

    for game_state in active_games:
        is_best_turn = _interleaved_is_best_turn(game_state)
        bundle = best_bundle if is_best_turn else train_bundle

        if game_state.phase == "new_move":
            _queue_interleaved_new_move(game_state, bundle, noise_frac, is_best_turn)
        elif game_state.phase == "searching":
            _queue_interleaved_search_leaves(game_state, bundle)

    return train_bundle, best_bundle


def _run_interleaved_eval_bundle(bundle, predict_fn):
    if not bundle["states"]:
        return

    states = bundle["states"]
    if len(states) == 1:
        batch = states[0][np.newaxis, ...]
    else:
        batch = np.stack(states, axis=0)
    if batch.dtype != np.float32:
        batch = batch.astype(np.float32, copy=False)
    logits_np, values_np = predict_fn(batch)
    values_np = values_np.ravel()

    for game_state, idx in bundle["root_evals"]:
        mcts_expand_root(game_state.ctx, logits_np[idx], values_np[idx])
        game_state.phase = "searching"

    for game_state, start, count in bundle["leaf_evals"]:
        mcts_process_results(
            game_state.ctx,
            logits_np[start:start + count],
            values_np[start:start + count],
        )


def _run_interleaved_forward_passes(
    train_bundle, best_bundle, predict_fn, best_predict_fn,
):
    for bundle, fn in (
        (train_bundle, predict_fn),
        (best_bundle, best_predict_fn),
    ):
        _run_interleaved_eval_bundle(bundle, fn)


def _interleaved_temperature(game_state, is_best_turn):
    if is_best_turn or not game_state.is_full_search:
        return 0.1
    return 1.0 if game_state.move_num < TEMP_THRESHOLD else 0.1


def _update_interleaved_soft_resign(game_state, root, is_best_turn):
    if (is_best_turn or game_state.soft_resigned
            or game_state.move_num < RESIGN_MIN_MOVES):
        return
    if root.q_value < -RESIGN_THRESHOLD:
        game_state.resign_count += 1
    else:
        game_state.resign_count = 0
    if game_state.resign_count >= RESIGN_CONSECUTIVE:
        game_state.soft_resigned = True


def _interleaved_winner_from_reward(game, reward):
    if reward == 1:
        return game.current_player
    if reward == -1:
        return -game.current_player
    return 0


def _set_interleaved_post_move_state(game_state, reward, done):
    if done:
        game_state.winner = _interleaved_winner_from_reward(game_state.game, reward)
        game_state.finished = True
        return
    game_state.phase = "new_move"


def _apply_interleaved_action(game_state, root, pi):
    row, col = select_action(pi)
    assert game_state.game.board[row, col] == 0, (
        f"ILLEGAL MOVE ({row},{col}) at move {game_state.move_num}, "
        f"board[{row},{col}]={game_state.game.board[row, col]}, "
        f"MCTS children={len(root.children) if hasattr(root, 'children') else '?'}, "
        f"stones={int(np.count_nonzero(game_state.game.board))}"
    )

    reward, done = game_state.game.make_move(row, col)
    game_state.move_num += 1
    _set_interleaved_post_move_state(game_state, reward, done)


def _advance_interleaved_completed_games(active_games):
    for game_state in active_games:
        if game_state.ctx is None:
            continue
        if game_state.ctx["sims_done"] < game_state.ctx["sims_target"]:
            continue

        root = game_state.ctx["root"]
        is_best_turn = _interleaved_is_best_turn(game_state)
        temp = _interleaved_temperature(game_state, is_best_turn)
        pi_play = mcts_policy(root, temperature=temp)
        pi_train = pi_play if game_state.is_full_search else _ZERO_POLICY
        game_state.trajectory.append((
            game_state.root_state,
            pi_train,
            game_state.game.current_player,
            game_state.move_sims,
            game_state.is_full_search,
            game_state.sims,
        ))
        _update_interleaved_soft_resign(game_state, root, is_best_turn)
        _apply_interleaved_action(game_state, root, pi_play)
        game_state.ctx = None


def _play_games_interleaved(predict_fn, game_configs, best_predict_fn=None,
                            noise_frac=0.25):
    """Play multiple self-play games with shared GPU inference."""
    all_games = [
        _init_interleaved_game(cfg, has_best_predict=best_predict_fn is not None)
        for cfg in game_configs
    ]

    while True:
        active_games = [g for g in all_games if not g.finished]
        if not active_games:
            break
        train_bundle, best_bundle = _collect_interleaved_eval_bundles(
            active_games, noise_frac
        )
        _run_interleaved_forward_passes(
            train_bundle,
            best_bundle,
            predict_fn,
            best_predict_fn,
        )
        _advance_interleaved_completed_games(active_games)

    return [(g.trajectory, g.winner, g.soft_resigned) for g in all_games]


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
    optimizer.apply_gradients(zip(grads, model.trainable_variables, strict=False))
    return policy_loss, value_loss, loss


# ── Best checkpoint tracking ──────────────────────────────────────────────
BEST_WEIGHTS_FILE = "weights/gomoku_best.weights.h5"
BEST_STATE_FILE   = "weights/best_checkpoint.pkl"

def _load_best_state():
    default = {"path": None, "game_count": 0, "glicko2_vs_long": None}
    if os.path.exists(BEST_STATE_FILE):
        try:
            with open(BEST_STATE_FILE, "rb") as f:
                state = pickle.load(f)
            if isinstance(state, dict):
                # Backward compatibility with older saved key name.
                if "glicko2_vs_long" not in state:
                    state["glicko2_vs_long"] = state.get("elo_vs_long")
                return {
                    "path": state.get("path"),
                    "game_count": int(state.get("game_count", 0) or 0),
                    "glicko2_vs_long": state.get("glicko2_vs_long"),
                }
        except Exception:
            pass
    return default


def _save_best_state(state):
    with open(BEST_STATE_FILE, "wb") as f:
        pickle.dump(state, f)


def _save_timestamped_checkpoint(model, game_count):
    """Save a timestamped checkpoint and return the file path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cp_file = f"weights/gomoku_{ts}_g{game_count:05d}.weights.h5"
    model.save_weights(cp_file)
    return cp_file


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
    """Lightweight eval vs recent checkpoint — prints per-color trends.

    Returns result dict or None if eval couldn't run.
    """
    checkpoints = find_checkpoints()
    if not checkpoints:
        return None

    # If evaluating immediately after a save, compare the new checkpoint
    # against t-1. Otherwise compare current in-memory weights against the
    # latest saved checkpoint.
    near_idx = None
    for i in range(len(checkpoints) - 1, -1, -1):
        gc = checkpoints[i][0]
        if abs(gc - game_count) <= 10:
            near_idx = i
            break

    if near_idx is not None:
        if near_idx < 1:
            return None
        gc_prev, fp_prev = checkpoints[near_idx - 1]
        label = f"t-1 (g{gc_prev})"
    else:
        older_idx = None
        for i in range(len(checkpoints) - 1, -1, -1):
            gc = checkpoints[i][0]
            if gc < game_count:
                older_idx = i
                break
        if older_idx is None:
            return None
        gc_prev, fp_prev = checkpoints[older_idx]
        label = f"latest cp (g{gc_prev})"

    print(f"  ── Diag g{game_count} vs {label} ──", flush=True)
    if not _load_weights_optional(
        opp_model, fp_prev, role="diagnostic opponent checkpoint"
    ):
        print("  Diagnostic eval skipped: no compatible opponent checkpoint.", flush=True)
        return None
    result = run_match_sequential(
        model, opp_model, label,
        openings=eval_openings,
        sims=DIAG_EVAL_SIMS,
        batch_size=MCTS_BATCH_SIZE,
    )
    return result


def _sample_eval_openings(all_openings, n_openings, seed):
    """Sample a reproducible subset of openings."""
    if not all_openings:
        return []
    rng = np.random.RandomState(seed)
    n_openings = min(n_openings, len(all_openings))
    idxs = rng.choice(len(all_openings), size=n_openings, replace=False)
    return [all_openings[i] for i in idxs]


def _find_checkpoint_near_game_count(checkpoints, game_count):
    for gc, fp in reversed(checkpoints):
        if abs(gc - game_count) <= 10:
            return fp
    return None


def _resolve_promotion_checkpoint_file(model, game_count, glicko2_table):
    checkpoints = find_checkpoints()
    cp_file = _find_checkpoint_near_game_count(checkpoints, game_count)
    if glicko2_table is not None and cp_file is None:
        cp_file = _save_timestamped_checkpoint(model, game_count)
        print(f"  ↳ Saved checkpoint for Glicko-2 tracking: {cp_file}", flush=True)
    return cp_file


def _run_promotion_match(model, game_count, all_openings, best_gc, opp_model):
    eval_openings = _sample_eval_openings(
        all_openings, PROMOTE_EVAL_OPENINGS, seed=game_count
    )
    label = f"best (g{best_gc})"
    print(f"  ── Promotion eval g{game_count} vs {label} ──", flush=True)
    if not _load_weights_optional(
        opp_model, BEST_WEIGHTS_FILE, role="best checkpoint for promotion eval"
    ):
        print("  Promotion eval skipped: best checkpoint is incompatible.", flush=True)
        return None
    return run_match_sequential(
        model, opp_model, label,
        openings=eval_openings,
        sims=PROMOTE_EVAL_SIMS,
        batch_size=MCTS_BATCH_SIZE,
    )


def _apply_promotion_rating_update(
    glicko2_table, cp_file, best_state, result,
):
    if glicko2_table is None or cp_file is None or not best_state.get("path"):
        return None

    upd = apply_match_glicko2_update(
        glicko2_table,
        cp_file,
        best_state.get("path"),
        wins=result["wins"],
        losses=result["losses"],
        draws=result["draws"],
    )
    if not upd:
        return None

    cand_entry = upd["candidate"]["entry"]
    opp_entry = upd["opponent"]["entry"]
    cand_rating_delta = upd["candidate"]["delta"]
    print(f"  ↳ Rating update: cand {format_glicko2_entry(cand_entry)}  "
          f"|  best {format_glicko2_entry(opp_entry)}", flush=True)
    save_glicko2_ratings(glicko2_table)
    return cand_rating_delta


def _compute_promotion_gate_metrics(result):
    z = 1.96 if PROMOTE_CI_LEVEL >= 0.95 else 1.645

    bw = result["black_wins"]
    bl = result["black_losses"]
    bd = result.get("black_draws", 0)
    bn = bw + bl + bd
    black_score = bw + 0.5 * bd
    black_lower = _wilson_lower(black_score, bn, z=z) if bn > 0 else 0.0

    ww = result["white_wins"]
    wl = result["white_losses"]
    wd = result.get("white_draws", 0)
    white_total = ww + wl + wd
    white_non_loss_rate = (ww + wd) / max(1, white_total)
    white_quick_loss_rate = result["white_quick_loss_rate"]
    white_survival = result["avg_white_loss_moves"]
    white_quick_loss_required = (
        wl >= PROMOTE_WHITE_QUICK_LOSS_MIN_LOSSES
        and white_non_loss_rate < PROMOTE_WHITE_QUICK_LOSS_CHECK_NON_LOSS_MAX
    )
    white_quick_loss_ok = (
        not white_quick_loss_required
        or white_quick_loss_rate <= PROMOTE_WHITE_QUICK_LOSS_MAX
    )
    white_ok = (
        white_non_loss_rate >= PROMOTE_WHITE_NON_LOSS_MIN
        and white_quick_loss_ok
        and (ww > 0 or white_survival >= PROMOTE_WHITE_SURVIVAL_MIN)
    )
    black_ok = black_lower > PROMOTE_BLACK_LOWER
    return {
        "black_ok": black_ok,
        "black_lower": black_lower,
        "ww": ww,
        "wl": wl,
        "white_non_loss_rate": white_non_loss_rate,
        "white_quick_loss_rate": white_quick_loss_rate,
        "white_quick_loss_required": white_quick_loss_required,
        "white_survival": white_survival,
        "white_ok": white_ok,
        "promoted": black_ok and white_ok,
    }


def _promotion_rejection_reasons(gates):
    reasons = []
    if not gates["black_ok"]:
        reasons.append(
            f"B lower={gates['black_lower']:.0%}≤{PROMOTE_BLACK_LOWER:.0%}"
        )
    if not gates["white_ok"]:
        if gates["white_non_loss_rate"] < PROMOTE_WHITE_NON_LOSS_MIN:
            reasons.append(
                f"W non-loss={100.0 * gates['white_non_loss_rate']:.0f}%"
                f"<{100.0 * PROMOTE_WHITE_NON_LOSS_MIN:.0f}%"
            )
        if (gates["white_quick_loss_required"]
                and gates["white_quick_loss_rate"] > PROMOTE_WHITE_QUICK_LOSS_MAX):
            reasons.append(
                f"W quick-loss={gates['white_quick_loss_rate']:.0f}%"
                f">{PROMOTE_WHITE_QUICK_LOSS_MAX:.0f}%"
            )
        if gates["ww"] == 0 and gates["white_survival"] < PROMOTE_WHITE_SURVIVAL_MIN:
            reasons.append(
                f"W wins=0, survival={gates['white_survival']:.0f}"
                f"<{PROMOTE_WHITE_SURVIVAL_MIN:.0f}"
            )
    return reasons


def _refresh_best_state_rating_snapshot(best_state, glicko2_table):
    if glicko2_table is None or not best_state.get("path"):
        return
    best_entry = get_glicko2_entry(
        glicko2_table, best_state["path"], create=False)
    if best_entry is not None:
        best_state["glicko2_vs_long"] = best_entry.get("rating")


def _promote_to_best_state(
    model,
    game_count,
    cp_file,
    result,
    gates,
    glicko2_table,
    cand_rating_delta,
):
    if cp_file is None:
        cp_file = _save_timestamped_checkpoint(model, game_count)
    shutil.copy2(cp_file, BEST_WEIGHTS_FILE)
    best_state = {
        "path": cp_file,
        "game_count": game_count,
        "glicko2_vs_long": result["glicko2_delta"],
    }
    if glicko2_table is not None:
        best_entry = get_glicko2_entry(
            glicko2_table, best_state["path"], create=False)
        if best_entry is not None:
            best_state["glicko2_vs_long"] = best_entry.get("rating")
    _save_best_state(best_state)
    white_reason = (
        f"W non-loss={100.0 * gates['white_non_loss_rate']:.0f}% "
        f"quick-loss={gates['white_quick_loss_rate']:.0f}% "
        + (f"wins={gates['ww']}"
           if gates["ww"] > 0
           else f"survival={gates['white_survival']:.0f}")
    )
    rating_delta = (
        cand_rating_delta
        if cand_rating_delta is not None
        else result["glicko2_delta"]
    )
    print(f"  ★ Promoted to best "
          f"(B lower={gates['black_lower']:.0%}>{PROMOTE_BLACK_LOWER:.0%}, "
          f"{white_reason}, "
          f"Glicko-2 Δ{rating_delta:+.0f})",
          flush=True)
    return best_state


def _run_promotion_eval(model, game_count, all_openings, best_state, opp_model,
                        glicko2_table=None):
    """Promotion eval vs current best checkpoint — promotes if both colors improved.

    Triggered by rolling diagnostic signal (2 strong checks out of latest 3).
    Uses full sims and a fresh random subset of openings.

    Promotion gate (per-color):
      - Black: Wilson lower CI of Black score (win=1/draw=0.5/loss=0)
        must exceed PROMOTE_BLACK_LOWER.
      - White: must not collapse defensively
        (minimum non-loss rate, bounded quick-loss rate, and wins or survival).
    """
    if not best_state.get("path") or not os.path.exists(BEST_WEIGHTS_FILE):
        print("  Promotion eval skipped: no best checkpoint file. "
              "Run eval_tournament.py first.", flush=True)
        return best_state

    best_gc = best_state.get("game_count", 0)
    cp_file = _resolve_promotion_checkpoint_file(model, game_count, glicko2_table)

    result = _run_promotion_match(
        model, game_count, all_openings, best_gc, opp_model
    )
    if result is None:
        return best_state
    cand_rating_delta = _apply_promotion_rating_update(
        glicko2_table, cp_file, best_state, result
    )
    gates = _compute_promotion_gate_metrics(result)
    if gates["promoted"]:
        return _promote_to_best_state(
            model,
            game_count,
            cp_file,
            result,
            gates,
            glicko2_table,
            cand_rating_delta,
        )

    reasons = _promotion_rejection_reasons(gates)
    print(f"  (best remains g{best_gc:05d} — {', '.join(reasons)})",
          flush=True)
    _refresh_best_state_rating_snapshot(best_state, glicko2_table)
    return best_state


# ── Main loop ──────────────────────────────────────────────────────────────
def _setup_training_run():
    t_start = time.time()
    print("=" * 60)
    print("Gomoku — AlphaZero-style MCTS Training")
    print("  (single-process GPU self-play + training)")
    print("=" * 60)

    if _gpus:
        print(f"GPU: {_gpus[0].name}")
    else:
        print("WARNING: No GPU detected — training will be slow.")

    model = _create_train_model()
    optimizer = keras.optimizers.AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY)

    os.makedirs("weights", exist_ok=True)
    weights_file = "weights/gomoku_weights.weights.h5"
    starting_game = 0
    replay = ReplayBuffer(REPLAY_SIZE)

    # Resume automatically from checkpoint if available.
    if os.path.exists(weights_file):
        _load_weights_strict(model, weights_file, role="main training")
        starting_game, lr_state, eval_state = _load_training_state(
            model, optimizer, replay)
        print("  Loaded existing weights.\n")
    else:
        lr_state = None
        eval_state = None
        print("  No existing weights — starting fresh.\n")

    # Automatic LR scheduling: warmup ramp then plateau-based reduction
    lr_scheduler = LRScheduler(optimizer)
    if lr_state is not None:
        lr_scheduler.load_state(lr_state)
    else:
        # Fresh run: start at warmup fraction of target LR
        optimizer.learning_rate.assign(LR * lr_scheduler.warmup_start_frac)

    total = model.count_params()
    warmup_enabled = (
        MCTS_SIMS_WARMUP_REPLAY > 0 and MCTS_SIMS_WARMUP > 0 and MCTS_SIMS_WARMUP < MCTS_SIMS
    )
    warmup_fast = _scaled_budget_for_sims(MCTS_SIMS_WARMUP, PCAP_FAST_SIMS)
    warmup_weak = _scaled_budget_for_sims(MCTS_SIMS_WARMUP, ASYM_WEAK_SIMS)
    print(f"Model: {NUM_INPUT_PLANES}ch input, {TRAIN_NUM_RES_BLOCKS} res-blocks × "
          f"{TRAIN_NUM_FILTERS} filters + SE, "
          f"{total:,} parameters")
    print(f"MCTS: {MCTS_SIMS} full / {PCAP_FAST_SIMS} fast sims "
          f"({PCAP_FULL_FRAC:.0%} full), "
          f"batch size {MCTS_BATCH_SIZE} full / {MCTS_BATCH_SIZE_FAST} fast (GPU)")
    if warmup_enabled:
        print(f"MCTS warmup: {MCTS_SIMS_WARMUP} full / {warmup_fast} fast / {warmup_weak} weak "
              f"until replay reaches {MCTS_SIMS_WARMUP_REPLAY:,} positions")
    print(f"Asymmetric play: {ASYM_FRAC_START:.0%}→{ASYM_FRAC_END:.0%} "
          f"over {ASYM_DECAY_GAMES} games, weak side {ASYM_WEAK_SIMS} sims")
    print(f"Openings: {BOOK_OPENING_FRAC:.0%} book, "
          f"{1-BOOK_OPENING_FRAC:.0%} random ({RANDOM_OPENING_PLIES} plies)")
    print(f"Soft resign: threshold {RESIGN_THRESHOLD}, "
          f"{RESIGN_CONSECUTIVE} consecutive, after move {RESIGN_MIN_MOVES} "
          f"(value-only, fast sims from current self-play budget; target {PCAP_FAST_SIMS})")
    from gomoku import _USE_ACCEL, _HAS_NUMBA
    print(f"Cython acceleration: {'enabled' if _USE_ACCEL else 'disabled (run setup_accel.py build_ext --inplace)'}")
    if _USE_ACCEL:
        print("Threat planes: Cython (~3μs)")
    elif _HAS_NUMBA:
        print("Threat planes: numba (~3μs, JIT warmup on first call)")
    else:
        print("Threat planes: numpy fallback (~150μs, rebuild Cython or pip install numba)")
    print(f"Checkpoints: save every {SAVE_INTERVAL}g")
    if INLINE_EVAL_PROMOTION_ENABLED:
        print(f"Eval: diagnostic every {EVAL_INTERVAL}g ({DIAG_EVAL_SIMS} sims), "
              f"promotion trigger: {PROMOTE_DIAG_MIN_STRONG} strong checks in rolling "
              f"{PROMOTE_DIAG_WINDOW} (arm immediately) with "
              f"Black lower>{PROMOTE_DIAG_BLACK_LOWER:.0%} "
              f"({PROMOTE_EVAL_SIMS} sims, per-color Wilson CI)")
    else:
        print("Eval: inline diagnostic/promotion disabled")
        print("      Run manual tournament after run/chunk, e.g.:")
        print("      python eval_tournament.py --tournament-dir weights")
    print(f"Concurrent games: {CONCURRENT_GAMES}, target: {NUM_GAMES}")
    print(f"Best-opponent: {BEST_PLAY_FRAC:.0%} of games vs best checkpoint")
    print(f"Noise: {NOISE_FRAC_START:.2f}→{NOISE_FRAC_END:.2f} over {NOISE_DECAY_GAMES}g")
    print(f"Replay buffer: {REPLAY_SIZE:,} positions, batch size: {BATCH_SIZE}, "
          f"recent bias: {REPLAY_RECENT_FRAC:.0%} from last {REPLAY_RECENT_SIZE:,}")
    current_full, current_fast, current_weak = _current_self_play_budgets(len(replay))
    warmup_active = current_full < MCTS_SIMS
    if warmup_active:
        print(f"Self-play sims: warmup active ({current_full}/{current_fast}/{current_weak}) "
              f"at replay {len(replay):,}/{MCTS_SIMS_WARMUP_REPLAY:,}")
    else:
        print(f"Self-play sims: full budget ({current_full}/{current_fast}/{current_weak})")
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
        warmup_batches=800,         # ~800 games before detection starts
        slope_rel_thresh=3e-4,
        v_improve_rel=5e-4,
        min_replay=40_000,
        cooldown=50,
        persist=3,
    )
    plateau_autotuner = PlateauAutoTuner(lr_scheduler)

    # Eval: fixed opening book persisted to disk
    full_openings = load_or_create_openings(
        max(PROMOTE_EVAL_OPENINGS, 150))
    # Diagnostic and promotion both sample from this pool each eval tick.

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
    glicko2_table = load_glicko2_ratings()
    if best_state["path"]:
        best_entry = get_glicko2_entry(
            glicko2_table, best_state.get("path"), create=False)
        rating_str = format_glicko2_entry(best_entry) if best_entry else "unrated"
        print(f"Current best: g{best_state['game_count']:05d} ({rating_str})")
    else:
        print("No best checkpoint yet — run eval_tournament.py to create one.")

    # Reusable opponent model for eval (swap weights per matchup)
    opp_model = _create_train_model()

    # Best-opponent model for self-play (if best checkpoint exists)
    best_model = None
    best_predict_fn = None
    if os.path.exists(BEST_WEIGHTS_FILE):
        best_model = _create_train_model()
        if _load_weights_optional(
            best_model, BEST_WEIGHTS_FILE, role="best-opponent checkpoint"
        ):
            best_predict_fn = make_predict_fn(best_model)
            best_predict_fn(np.zeros((1, BOARD_SIZE, BOARD_SIZE, NUM_INPUT_PLANES),
                                     dtype=np.float32))
            print(f"Best-opponent loaded from {BEST_WEIGHTS_FILE} "
                  f"(g{best_state.get('game_count', '?')})")
        else:
            best_model = None
            best_predict_fn = None
            print("Best-opponent disabled until a compatible best checkpoint exists.")
    else:
        print("No best checkpoint — vs-best games disabled until tournament promotion")
    print()

    game_count = starting_game
    target = starting_game + NUM_GAMES
    diag_strong_history = deque(maxlen=PROMOTE_DIAG_WINDOW)
    promotion_pending = False
    if eval_state:
        saved_hist = eval_state.get("diag_strong_history", [])
        if isinstance(saved_hist, (list, tuple)):
            for v in saved_hist[-PROMOTE_DIAG_WINDOW:]:
                diag_strong_history.append(bool(v))
        promotion_pending = bool(eval_state.get("promotion_pending", False))
        if diag_strong_history or promotion_pending:
            print("  Restored promotion trigger state: "
                  f"window={int(sum(diag_strong_history))}/{len(diag_strong_history)} strong, "
                  f"pending={'yes' if promotion_pending else 'no'}")
    emerg_lr_fired = False   # emergency LR cut: fires once per regression episode
    consec_mild_regress = 0  # consecutive diagnostics in 30-40% range

    # Warm up TF graph with compiled predict function.
    # input_signature=[None, ...] traces once for all batch sizes.
    predict_fn = make_predict_fn(model)
    predict_fn(np.zeros((1, BOARD_SIZE, BOARD_SIZE, NUM_INPUT_PLANES), dtype=np.float32))

    elapsed = time.time() - t_start
    print(f"Ready. (startup took {elapsed:.0f}s)\n")
    return {
        "t_start": t_start,
        "model": model,
        "optimizer": optimizer,
        "weights_file": weights_file,
        "replay": replay,
        "lr_scheduler": lr_scheduler,
        "plateau": plateau,
        "plateau_autotuner": plateau_autotuner,
        "full_openings": full_openings,
        "book_openings": book_openings,
        "best_state": best_state,
        "glicko2_table": glicko2_table,
        "opp_model": opp_model,
        "best_model": best_model,
        "best_predict_fn": best_predict_fn,
        "predict_fn": predict_fn,
        "game_count": game_count,
        "target": target,
        "diag_strong_history": diag_strong_history,
        "promotion_pending": promotion_pending,
        "emerg_lr_fired": emerg_lr_fired,
        "consec_mild_regress": consec_mild_regress,
        "autotune_asym_boost_until": 0,
        "autotune_noise_boost_until": 0,
        "autotune_asym_boost_value": PLATEAU_AUTOTUNE_STALEMATE_ASYM_BOOST,
        "autotune_noise_boost_value": PLATEAU_AUTOTUNE_STALEMATE_NOISE_BOOST,
        "sims_warmup_active": warmup_active,
    }


def _scaled_budget_for_sims(base_sims, ref_budget):
    base_sims = max(1, int(base_sims))
    if MCTS_SIMS <= 0:
        return max(1, min(base_sims, int(ref_budget)))
    scaled = int(round(float(ref_budget) * float(base_sims) / float(MCTS_SIMS)))
    return max(1, min(base_sims, scaled))


def _current_self_play_sims(replay_len):
    warmup_sims = int(MCTS_SIMS_WARMUP)
    warmup_replay = int(MCTS_SIMS_WARMUP_REPLAY)
    if warmup_sims <= 0 or warmup_replay <= 0 or warmup_sims >= MCTS_SIMS:
        return int(MCTS_SIMS)
    if int(replay_len) < warmup_replay:
        return warmup_sims
    return int(MCTS_SIMS)


def _current_self_play_budgets(replay_len):
    full_sims = _current_self_play_sims(replay_len)
    fast_sims = _scaled_budget_for_sims(full_sims, PCAP_FAST_SIMS)
    weak_sims = _scaled_budget_for_sims(full_sims, ASYM_WEAK_SIMS)
    return full_sims, fast_sims, weak_sims


def _loop_batch_schedule(game_count, target, loop_state=None):
    batch_games = min(CONCURRENT_GAMES, target - game_count)
    progress = min(1.0, game_count / ASYM_DECAY_GAMES)
    asym_frac = ASYM_FRAC_START + (ASYM_FRAC_END - ASYM_FRAC_START) * progress
    noise_progress = min(1.0, game_count / NOISE_DECAY_GAMES)
    noise_frac = NOISE_FRAC_START + (NOISE_FRAC_END - NOISE_FRAC_START) * noise_progress

    if loop_state:
        if game_count < int(loop_state.get("autotune_asym_boost_until", 0)):
            asym_boost = float(loop_state.get(
                "autotune_asym_boost_value", PLATEAU_AUTOTUNE_STALEMATE_ASYM_BOOST
            ))
            asym_frac = min(1.0, asym_frac + asym_boost)
        if game_count < int(loop_state.get("autotune_noise_boost_until", 0)):
            noise_boost = float(loop_state.get(
                "autotune_noise_boost_value", PLATEAU_AUTOTUNE_STALEMATE_NOISE_BOOST
            ))
            noise_frac = min(1.0, noise_frac + noise_boost)
    return batch_games, asym_frac, noise_frac


def _select_self_play_opening(book_openings):
    if book_openings and np.random.random() < BOOK_OPENING_FRAC:
        return book_openings[np.random.randint(len(book_openings))]
    return RANDOM_OPENING_PLIES


def _build_self_play_game_configs(
    batch_games,
    book_openings,
    best_predict_fn,
    asym_frac,
    full_sims,
    fast_sims,
    weak_sims,
):
    game_configs = []
    n_asym = 0
    n_vs_best = 0
    for _ in range(batch_games):
        opening = _select_self_play_opening(book_openings)
        if best_predict_fn is not None and np.random.random() < BEST_PLAY_FRAC:
            game_configs.append((full_sims, fast_sims, full_sims, None, opening, True))
            n_vs_best += 1
        elif np.random.random() < asym_frac:
            weak = PLAYER1 if np.random.random() < 0.5 else -PLAYER1
            game_configs.append((full_sims, fast_sims, weak_sims, weak, opening, False))
            n_asym += 1
        else:
            game_configs.append((full_sims, fast_sims, full_sims, None, opening, False))
    return game_configs, n_asym, n_vs_best


def _run_self_play_batch(predict_fn, game_configs, best_predict_fn, noise_frac):
    t0 = time.time()
    results = _play_games_interleaved(
        predict_fn, game_configs, best_predict_fn, noise_frac
    )
    return results, time.time() - t0


def _trajectory_outcome(winner, player):
    if winner == 0:
        return 0.0
    if player == winner:
        return 1.0
    return -1.0


def _policy_weight_for_sample(move_sims, is_full, full_sims):
    if is_full:
        base = max(1, int(full_sims))
        return np.float32(np.sqrt(float(move_sims) / float(base)))
    return np.float32(0.0)


def _ingest_self_play_results(results, replay, game_count, target):
    new_positions = 0
    move_counts = []
    n_resigned = 0

    for trajectory, winner, resigned in results:
        if not trajectory:
            continue
        if resigned:
            n_resigned += 1
        move_counts.append(len(trajectory))
        for state, pi, player, move_sims, is_full, full_sims in trajectory:
            outcome = _trajectory_outcome(winner, player)
            pi_w = _policy_weight_for_sample(move_sims, is_full, full_sims)
            replay.append((state, pi, np.float32(outcome), pi_w))
            new_positions += 1
        game_count += 1
        if game_count >= target:
            break

    return new_positions, move_counts, n_resigned, game_count


def _run_training_updates(model, optimizer, replay, new_positions):
    def _build_train_batch_arrays(replay_buf, idxs):
        batch_n = len(idxs)
        s = np.empty(
            (batch_n, BOARD_SIZE, BOARD_SIZE, NUM_INPUT_PLANES),
            dtype=np.float32,
        )
        p = np.empty((batch_n, BOARD_SIZE * BOARD_SIZE), dtype=np.float32)
        v = np.empty((batch_n,), dtype=np.float32)
        w = np.empty((batch_n,), dtype=np.float32)
        for j, ridx in enumerate(idxs):
            state, pi, outcome, pi_w = replay_buf[int(ridx)]
            s[j] = state
            p[j] = pi
            v[j] = outcome
            w[j] = pi_w
        return s, p, v, w

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
            s, p, v, w = _build_train_batch_arrays(replay, idxs)
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
        total_ent = (total_ent / ent_steps) if ent_steps else 0.0

    return {
        "did_train": did_train,
        "total_ploss": total_ploss,
        "total_vloss": total_vloss,
        "total_ent": total_ent,
        "total_mean_w": total_mean_w,
        "train_time": time.time() - t1,
    }


def _self_play_summary_stats(results, move_counts):
    avg_moves = float(np.mean(move_counts)) if move_counts else 0.0
    kept = [(traj, winner, resigned) for (traj, winner, resigned) in results if traj]
    kept_n = len(kept)
    n_decisive = sum(1 for (_, winner, _) in kept if winner != 0)
    decisive_rate = n_decisive / max(1, kept_n)
    return avg_moves, kept_n, n_decisive, decisive_rate


def _print_training_batch_status(
    game_count,
    target,
    n_asym,
    batch_games,
    asym_frac,
    full_sims,
    fast_sims,
    weak_sims,
    avg_moves,
    n_decisive,
    kept_n,
    n_resigned,
    n_vs_best,
    train_stats,
    replay_len,
    sp_time,
    lr_scheduler,
):
    resign_str = f"  SR {n_resigned}" if n_resigned else ""
    best_str = f"  B {n_vs_best}" if n_vs_best else ""
    lr_str = f"  lr {lr_scheduler.current_lr:.1e}" if lr_scheduler._reductions > 0 else ""
    print(
        f"Game {game_count:5d}/{target} | "
        f"Asym {n_asym}/{batch_games} ({asym_frac:.0%}) | "
        f"Sims {full_sims}/{fast_sims}/{weak_sims} | "
        f"Moves {avg_moves:5.1f} | "
        f"Win {n_decisive}/{kept_n}{resign_str}{best_str} | "
        f"P {train_stats['total_ploss']:.4f} V {train_stats['total_vloss']:.4f} "
        f"w̄ {train_stats['total_mean_w']:.2f} H {train_stats['total_ent']:.2f} "
        f"gap {train_stats['total_ploss'] - train_stats['total_ent']:.3f} | "
        f"Buf {replay_len:6d} | "
        f"SP {sp_time:.1f}s  Tr {train_stats['train_time']:.1f}s{lr_str}"
    , flush=True)


def _apply_train_feedback_updates(
    plateau,
    lr_scheduler,
    train_stats,
    decisive_rate,
    replay_len,
    game_count,
    loop_state,
    plateau_autotuner,
):
    if not train_stats["did_train"]:
        return

    rec = plateau.update(
        ploss=train_stats["total_ploss"],
        vloss=train_stats["total_vloss"],
        target_entropy=train_stats["total_ent"],
        decisive_rate=decisive_rate,
        replay_len=replay_len,
    )
    if rec:
        print("  " + rec, flush=True)

    if plateau_autotuner is not None:
        auto_msg = plateau_autotuner.on_plateau(
            plateau.last_event(),
            game_count,
            loop_state,
        )
        if auto_msg:
            print("  " + auto_msg, flush=True)

    lr_msg = lr_scheduler.update(train_stats["total_ploss"], train_stats["total_ent"])
    if lr_msg:
        print("  " + lr_msg, flush=True)


def _load_best_predict_fn(best_model):
    if best_model is None:
        best_model = _create_train_model()
    if not _load_weights_optional(
        best_model, BEST_WEIGHTS_FILE, role="best-opponent checkpoint"
    ):
        return None, None
    best_predict_fn = make_predict_fn(best_model)
    best_predict_fn(np.zeros(
        (1, BOARD_SIZE, BOARD_SIZE, NUM_INPUT_PLANES),
        dtype=np.float32,
    ))
    return best_model, best_predict_fn


def _maybe_save_checkpoint(
    state,
    model,
    optimizer,
    replay,
    lr_scheduler,
    weights_file,
    glicko2_table,
    game_count,
    prev_game_count,
):
    prev_save_tick = prev_game_count // SAVE_INTERVAL
    curr_save_tick = game_count // SAVE_INTERVAL
    if curr_save_tick <= prev_save_tick:
        return

    cp_file = _save_timestamped_checkpoint(model, game_count)
    model.save_weights(weights_file)
    save_glicko2_ratings(glicko2_table)
    _save_training_state(
        game_count, optimizer, replay, lr_scheduler,
        eval_state={
            "diag_strong_history": list(state["diag_strong_history"]),
            "promotion_pending": bool(state["promotion_pending"]),
        },
    )
    print(f"  → Checkpoint {cp_file}", flush=True)


def _diag_eval_signal(model, game_count, full_openings, opp_model, diag_strong_history):
    diag_openings = _sample_eval_openings(
        full_openings, INLINE_EVAL_OPENINGS, seed=game_count
    )
    diag_result = _run_diagnostic_eval(
        model, game_count, diag_openings, opp_model)
    diag_black_pct = (diag_result["black_win_pct"] if diag_result else None)
    diag_strong = False
    if diag_result is None:
        return diag_black_pct, diag_strong

    diag_bw = diag_result["black_wins"]
    diag_bl = diag_result["black_losses"]
    diag_bd = diag_result.get("black_draws", 0)
    diag_bn = diag_bw + diag_bl + diag_bd
    diag_black_score = diag_bw + 0.5 * diag_bd
    diag_z = 1.96 if PROMOTE_DIAG_CI_LEVEL >= 0.95 else 1.645
    diag_black_lower = (
        _wilson_lower(diag_black_score, diag_bn, z=diag_z)
        if diag_bn > 0 else 0.0
    )
    diag_strong = (diag_black_lower >= PROMOTE_DIAG_BLACK_LOWER)
    diag_strong_history.append(diag_strong)
    strong_n = int(sum(diag_strong_history))
    print(
        f"  Diag trigger signal: Black lower={diag_black_lower:.0%} "
        f"({'strong' if diag_strong else 'weak'}), "
        f"window={strong_n}/{len(diag_strong_history)} strong",
        flush=True,
    )
    return diag_black_pct, diag_strong


def _maybe_emergency_lr_cut(
    lr_scheduler, game_count, diag_black_pct, emerg_lr_fired, consec_mild_regress,
):
    if (diag_black_pct is None or emerg_lr_fired
            or game_count < EMERG_LR_MIN_GAMES):
        return emerg_lr_fired, consec_mild_regress

    if diag_black_pct <= EMERG_LR_SEVERE_PCT:
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
        return emerg_lr_fired, consec_mild_regress

    if diag_black_pct < EMERG_LR_BLACK_PCT:
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
                  f"(Black <{EMERG_LR_BLACK_PCT:.0f}% for 2 diagnostics)",
                  flush=True)
    else:
        consec_mild_regress = 0
    return emerg_lr_fired, consec_mild_regress


def _maybe_arm_promotion_pending(promotion_pending, diag_strong_history):
    if (len(diag_strong_history) >= PROMOTE_DIAG_MIN_STRONG
            and sum(diag_strong_history) >= PROMOTE_DIAG_MIN_STRONG):
        if not promotion_pending:
            print("  → Promotion trigger armed (rolling diagnostics)", flush=True)
        promotion_pending = True
    return promotion_pending


def _maybe_run_pending_promotion(state, model, game_count, full_openings, opp_model, glicko2_table):
    if not state["promotion_pending"]:
        return

    best_gc = state["best_state"].get("game_count", 0)
    since_best = abs(game_count - best_gc)
    if since_best < PROMOTE_COOLDOWN:
        wait = PROMOTE_COOLDOWN - since_best
        print(
            f"  (promotion pending: cooldown {since_best}/{PROMOTE_COOLDOWN}g, "
            f"wait {wait}g)",
            flush=True,
        )
        return

    old_best_gc = state["best_state"].get("game_count", 0)
    state["best_state"] = _run_promotion_eval(
        model, game_count, full_openings,
        state["best_state"], opp_model,
        glicko2_table=glicko2_table,
    )
    state["promotion_pending"] = False
    state["diag_strong_history"].clear()
    if state["best_state"].get("game_count", 0) != old_best_gc:
        state["best_model"], state["best_predict_fn"] = _load_best_predict_fn(
            state["best_model"]
        )


def _maybe_run_eval_tick(
    state,
    model,
    game_count,
    full_openings,
    opp_model,
    lr_scheduler,
    glicko2_table,
    prev_game_count,
):
    prev_eval_tick = prev_game_count // EVAL_INTERVAL
    curr_eval_tick = game_count // EVAL_INTERVAL
    if curr_eval_tick <= prev_eval_tick:
        return

    diag_black_pct, diag_strong = _diag_eval_signal(
        model,
        game_count,
        full_openings,
        opp_model,
        state["diag_strong_history"],
    )
    state["emerg_lr_fired"], state["consec_mild_regress"] = _maybe_emergency_lr_cut(
        lr_scheduler,
        game_count,
        diag_black_pct,
        state["emerg_lr_fired"],
        state["consec_mild_regress"],
    )
    if diag_strong:
        state["emerg_lr_fired"] = False
        state["consec_mild_regress"] = 0

    state["promotion_pending"] = _maybe_arm_promotion_pending(
        state["promotion_pending"],
        state["diag_strong_history"],
    )
    _maybe_run_pending_promotion(
        state,
        model,
        game_count,
        full_openings,
        opp_model,
        glicko2_table,
    )


def _run_iteration_self_play_and_train(loop_ctx, state):
    model = loop_ctx["model"]
    optimizer = loop_ctx["optimizer"]
    replay = loop_ctx["replay"]
    lr_scheduler = loop_ctx["lr_scheduler"]
    plateau_autotuner = loop_ctx["plateau_autotuner"]
    plateau = loop_ctx["plateau"]
    book_openings = loop_ctx["book_openings"]
    predict_fn = loop_ctx["predict_fn"]
    game_count = loop_ctx["game_count"]
    target = loop_ctx["target"]

    batch_games, asym_frac, noise_frac = _loop_batch_schedule(
        game_count, target, loop_state=state
    )
    full_sims, fast_sims, weak_sims = _current_self_play_budgets(len(replay))
    warmup_active = full_sims < MCTS_SIMS
    if state.get("sims_warmup_active") is None:
        state["sims_warmup_active"] = warmup_active
    elif state["sims_warmup_active"] != warmup_active:
        if warmup_active:
            print(f"  Self-play sims warmup active: {full_sims}/{fast_sims}/{weak_sims}",
                  flush=True)
        else:
            print(
                f"  Self-play sims switched to full budget: "
                f"{full_sims}/{fast_sims}/{weak_sims} "
                f"(replay {len(replay):,}/{MCTS_SIMS_WARMUP_REPLAY:,})",
                flush=True,
            )
        state["sims_warmup_active"] = warmup_active
    game_configs, n_asym, n_vs_best = _build_self_play_game_configs(
        batch_games,
        book_openings,
        state["best_predict_fn"],
        asym_frac,
        full_sims,
        fast_sims,
        weak_sims,
    )
    results, sp_time = _run_self_play_batch(
        predict_fn, game_configs, state["best_predict_fn"], noise_frac
    )
    new_positions, move_counts, n_resigned, game_count = _ingest_self_play_results(
        results, replay, game_count, target
    )
    train_stats = _run_training_updates(
        model, optimizer, replay, new_positions
    )
    avg_moves, kept_n, n_decisive, decisive_rate = _self_play_summary_stats(
        results, move_counts
    )
    _print_training_batch_status(
        game_count,
        target,
        n_asym,
        batch_games,
        asym_frac,
        full_sims,
        fast_sims,
        weak_sims,
        avg_moves,
        n_decisive,
        kept_n,
        n_resigned,
        n_vs_best,
        train_stats,
        len(replay),
        sp_time,
        lr_scheduler,
    )
    _apply_train_feedback_updates(
        plateau,
        lr_scheduler,
        train_stats,
        decisive_rate,
        len(replay),
        game_count,
        state,
        plateau_autotuner,
    )
    return game_count, (game_count - len(move_counts))


def _run_iteration_scheduled_tasks(loop_ctx, state, game_count, prev_game_count):
    model = loop_ctx["model"]
    optimizer = loop_ctx["optimizer"]
    weights_file = loop_ctx["weights_file"]
    replay = loop_ctx["replay"]
    lr_scheduler = loop_ctx["lr_scheduler"]
    full_openings = loop_ctx["full_openings"]
    glicko2_table = loop_ctx["glicko2_table"]
    opp_model = loop_ctx["opp_model"]

    _maybe_save_checkpoint(
        state,
        model,
        optimizer,
        replay,
        lr_scheduler,
        weights_file,
        glicko2_table,
        game_count,
        prev_game_count,
    )
    if INLINE_EVAL_PROMOTION_ENABLED:
        _maybe_run_eval_tick(
            state,
            model,
            game_count,
            full_openings,
            opp_model,
            lr_scheduler,
            glicko2_table,
            prev_game_count,
        )


def _run_training_iteration(loop_ctx, state):
    game_count, prev_game_count = _run_iteration_self_play_and_train(
        loop_ctx, state
    )
    _run_iteration_scheduled_tasks(
        loop_ctx,
        state,
        game_count,
        prev_game_count,
    )
    loop_ctx["game_count"] = game_count


def _run_training_loop(ctx):
    state = {
        "best_state": ctx["best_state"],
        "best_model": ctx["best_model"],
        "best_predict_fn": ctx["best_predict_fn"],
        "diag_strong_history": ctx["diag_strong_history"],
        "promotion_pending": ctx["promotion_pending"],
        "emerg_lr_fired": ctx["emerg_lr_fired"],
        "consec_mild_regress": ctx["consec_mild_regress"],
        "autotune_asym_boost_until": ctx.get("autotune_asym_boost_until", 0),
        "autotune_noise_boost_until": ctx.get("autotune_noise_boost_until", 0),
        "autotune_asym_boost_value": ctx.get(
            "autotune_asym_boost_value", PLATEAU_AUTOTUNE_STALEMATE_ASYM_BOOST
        ),
        "autotune_noise_boost_value": ctx.get(
            "autotune_noise_boost_value", PLATEAU_AUTOTUNE_STALEMATE_NOISE_BOOST
        ),
        "sims_warmup_active": ctx.get("sims_warmup_active"),
    }

    try:
        while ctx["game_count"] < ctx["target"]:
            _run_training_iteration(ctx, state)

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted — saving training state …", flush=True)
        interrupted = True
    else:
        interrupted = False

    ctx.update({
        "best_state": state["best_state"],
        "best_model": state["best_model"],
        "best_predict_fn": state["best_predict_fn"],
        "game_count": ctx["game_count"],
        "diag_strong_history": state["diag_strong_history"],
        "promotion_pending": state["promotion_pending"],
        "emerg_lr_fired": state["emerg_lr_fired"],
        "consec_mild_regress": state["consec_mild_regress"],
        "autotune_asym_boost_until": state["autotune_asym_boost_until"],
        "autotune_noise_boost_until": state["autotune_noise_boost_until"],
        "autotune_asym_boost_value": state["autotune_asym_boost_value"],
        "autotune_noise_boost_value": state["autotune_noise_boost_value"],
        "sims_warmup_active": state.get("sims_warmup_active"),
    })
    return interrupted


def _finalize_training_run(ctx, interrupted):
    model = ctx["model"]
    optimizer = ctx["optimizer"]
    weights_file = ctx["weights_file"]
    replay = ctx["replay"]
    lr_scheduler = ctx["lr_scheduler"]
    best_state = ctx["best_state"]
    glicko2_table = ctx["glicko2_table"]
    game_count = ctx["game_count"]
    diag_strong_history = ctx["diag_strong_history"]
    promotion_pending = ctx["promotion_pending"]
    t_start = ctx["t_start"]

    # ── Cleanup & final save ────────────────────────────────────────────
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "interrupted" if interrupted else "final"
    model.save_weights(f"weights/gomoku_{ts}_{tag}.weights.h5")
    model.save_weights(weights_file)
    save_glicko2_ratings(glicko2_table)

    _save_training_state(
        game_count, optimizer, replay, lr_scheduler,
        eval_state={
            "diag_strong_history": list(diag_strong_history),
            "promotion_pending": bool(promotion_pending),
        },
    )

    if interrupted:
        print(f"\n✓ State saved at game {game_count}.  Resume with: python train.py")
    else:
        print(f"\n✓ Training complete — {game_count} games.  Weights → {weights_file}")

    if best_state.get("path"):
        best_entry = get_glicko2_entry(
            glicko2_table, best_state.get("path"), create=False)
        rating_str = format_glicko2_entry(best_entry) if best_entry else "unrated"
        print(f"  Best checkpoint: g{best_state['game_count']:05d} "
              f"({rating_str})  → {BEST_WEIGHTS_FILE}")

    elapsed = time.time() - t_start
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    print(f"  Total time: {h}h {m}m {s}s")


def main():
    ctx = _setup_training_run()
    interrupted = _run_training_loop(ctx)
    _finalize_training_run(ctx, interrupted)


if __name__ == "__main__":
    main()
