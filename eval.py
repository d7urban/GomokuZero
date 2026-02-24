#!/usr/bin/env python3
"""
Gomoku — Evaluate checkpoints against older baselines.

Runs deterministic matches (no noise, temperature=0) between two
checkpoints on GPU and reports win rates, per-color stats, average game
length, and Glicko-2 rating delta.

Usage:
    # Auto-detect latest checkpoint, eval vs t-1 and t-5:
    python eval.py

    # Evaluate a specific checkpoint:
    python eval.py --checkpoint weights/gomoku_best.weights.h5

    # Custom opening count:
    python eval.py --openings 200

    # Calibrate sim tiers on one checkpoint:
    python eval.py --checkpoint weights/gomoku_best.weights.h5 \
        --calibrate-sims --sim-levels 100,400,1600

    # Run a best-of-best tournament over all weights in a folder:
    python eval.py --tournament-dir botb-weights
"""

import argparse
import glob
import math
import os
import pickle
import re
import time

import numpy as np

# Suppress TF logging before any TF import (gomoku.py imports TF at module level).
# setdefault lets train.py's explicit settings take precedence when importing eval.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

from gomoku import (
    BOARD_SIZE, PLAYER1, PLAYER2, NUM_INPUT_PLANES,
    GomokuGame, create_model, make_predict_fn,
    mcts_search_batched, mcts_policy,
)

# ── Defaults ────────────────────────────────────────────────────────────────
EVAL_SIMS         = 100    # MCTS simulations per move (same for both sides)
EVAL_BATCH_SIZE   = 8      # MCTS batch size
EVAL_GAMES        = 200    # games per matchup (100 openings × 2 color swaps)
EVAL_OPENING_PLIES = 6     # random moves per opening position
EVAL_OPENING_SEED  = 42    # fixed seed — same openings across all evals

# Tournament defaults (best-of-best over multiple weight files)
TOURNEY_RR_SIMS = 50
TOURNEY_RR_OPENINGS = 8              # 16 games per pairing per round
TOURNEY_RR_MAX_ROUNDS = 8
TOURNEY_RR_CONTENDER_PROB = 0.90     # top-2 must beat all others at this p
TOURNEY_RR_MIN_GAMES = 40

# Heads-up final: strong sims for decision quality, then cheap 50-sim
# squeeze batches to tighten RD if needed.
TOURNEY_H2H_PHASES = ((200, 6), (400, 4), (50, 8))   # (sims, batches)
TOURNEY_H2H_OPENINGS_PER_BATCH = 20          # 40 games per batch
TOURNEY_H2H_CONFIDENCE_PROB = 0.975          # ~2 sigma winner confidence
TOURNEY_H2H_MIN_GAMES = 120
TOURNEY_H2H_RATING_PERIOD_GAMES = 200        # aggregate before Glicko update
TOURNEY_H2H_TARGET_RD = 19.5                 # keep running until RD is below this


# ── Opening book generation ─────────────────────────────────────────────────
OPENING_BOOK_FILE = "weights/eval_openings.pkl"
GLICKO2_RATINGS_FILE = "weights/glicko2_ratings.pkl"


def _parse_sim_levels(spec):
    """Parse comma-separated sim levels like '100,400,1600'."""
    try:
        vals = [int(x.strip()) for x in spec.split(",") if x.strip()]
    except Exception as e:
        raise ValueError(f"invalid sim list '{spec}': {e}") from e
    if len(vals) < 2:
        raise ValueError("need at least two sim levels")
    if any(v <= 0 for v in vals):
        raise ValueError("sim levels must be positive integers")
    vals = sorted(set(vals))
    if len(vals) < 2:
        raise ValueError("need at least two distinct sim levels")
    return vals

def generate_openings(n_openings, n_plies=EVAL_OPENING_PLIES,
                      seed=EVAL_OPENING_SEED, max_center_gap=2.0):
    """Generate a fixed set of balanced opening move sequences.

    Uses a fixed RNG seed so every eval run (and every checkpoint
    comparison) starts from the exact same positions.

    Rejects openings where the average center distance between Black
    and White stones differs by more than `max_center_gap`, ensuring
    neither side gets a systematic positional advantage.

    Returns list of [(row, col), ...] move sequences.
    """
    rng = np.random.RandomState(seed)
    center = BOARD_SIZE // 2
    openings = []
    attempts = 0
    max_attempts = n_openings * 50
    while len(openings) < n_openings and attempts < max_attempts:
        attempts += 1
        game = GomokuGame()
        moves = []
        ok = True
        for _ in range(n_plies):
            valid = game.get_valid_moves()
            if not valid:
                ok = False
                break
            idx = rng.randint(len(valid))
            r, c = valid[idx]
            reward, done = game.make_move(r, c)
            moves.append((r, c))
            if done:
                ok = False
                break
        if not ok or len(moves) != n_plies:
            continue

        # Balance check: compare avg Manhattan distance to center
        black_moves = [moves[j] for j in range(0, len(moves), 2)]
        white_moves = [moves[j] for j in range(1, len(moves), 2)]
        b_dist = np.mean([abs(r - center) + abs(c - center)
                          for r, c in black_moves])
        w_dist = np.mean([abs(r - center) + abs(c - center)
                          for r, c in white_moves])
        if abs(b_dist - w_dist) > max_center_gap:
            continue

        openings.append(moves)

    if len(openings) < n_openings:
        print(f"  Warning: only generated {len(openings)}/{n_openings} "
              f"balanced openings in {max_attempts} attempts")
    return openings


def load_or_create_openings(n_openings, n_plies=EVAL_OPENING_PLIES,
                            seed=EVAL_OPENING_SEED,
                            path=OPENING_BOOK_FILE):
    """Load opening book from disk, or generate and save it.

    The first call generates and persists the book.  All subsequent
    calls (including from train.py inline eval) load the same file,
    guaranteeing identical openings across runs even if game logic
    or RNG internals change.

    If more openings are requested than the file contains, the file
    is regenerated from scratch with balanced openings.

    Validates that existing openings pass the balance check; if not,
    regenerates the entire book.
    """
    import pickle
    center = BOARD_SIZE // 2
    max_gap = 2.0

    def _is_balanced(openings_list):
        """Check if opening set has reasonable Black/White balance."""
        b_dists, w_dists = [], []
        for moves in openings_list:
            black = [moves[j] for j in range(0, len(moves), 2)]
            white = [moves[j] for j in range(1, len(moves), 2)]
            b_dists.append(np.mean([abs(r-center)+abs(c-center)
                                    for r, c in black]))
            w_dists.append(np.mean([abs(r-center)+abs(c-center)
                                    for r, c in white]))
        avg_gap = abs(np.mean(b_dists) - np.mean(w_dists))
        return avg_gap < 1.0  # overall book should be very balanced

    existing = []
    if os.path.exists(path):
        with open(path, "rb") as f:
            existing = pickle.load(f)
        if len(existing) >= n_openings and _is_balanced(existing[:n_openings]):
            print(f"  Loaded opening book: {len(existing)} balanced positions from {path}")
            return existing
        if existing and not _is_balanced(existing[:min(len(existing), n_openings)]):
            print(f"  Opening book is unbalanced — regenerating …")
            existing = []  # force full regeneration

    # Generate fresh balanced openings
    new_seed = seed + len(existing)
    need = n_openings - len(existing)
    new_openings = generate_openings(need, n_plies=n_plies, seed=new_seed,
                                     max_center_gap=max_gap)
    combined = existing + new_openings

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(combined, f)

    action = "Extended" if existing else "Generated"
    print(f"  {action} opening book: {len(combined)} balanced positions "
          f"({n_plies} plies) → {path}")
    return combined


# ── Checkpoint discovery ────────────────────────────────────────────────────
def find_checkpoints(weights_dir="weights"):
    """Find all timestamped checkpoints and return sorted by game count.

    Returns list of (game_count, filepath) sorted ascending.
    Excludes 'latest', 'final', 'interrupted', and temp files.
    """
    pattern = os.path.join(weights_dir, "gomoku_*_g?????.weights.h5")
    files = glob.glob(pattern)
    checkpoints = []
    for f in files:
        m = re.search(r"_g(\d{5})\.weights\.h5$", f)
        if m:
            game_count = int(m.group(1))
            checkpoints.append((game_count, f))
    checkpoints.sort()
    return checkpoints


def find_weight_files(weights_dir):
    """Find model weight files in a directory for tournament mode.

    Prefers .weights.h5 but also allows .h5/.keras for convenience.
    Returns normalized absolute/relative paths sorted by filename.
    """
    patterns = ("*.weights.h5", "*.h5", "*.keras")
    found = []
    seen = set()
    for pat in patterns:
        for fp in glob.glob(os.path.join(weights_dir, pat)):
            if not os.path.isfile(fp):
                continue
            key = os.path.normpath(fp)
            if key in seen:
                continue
            seen.add(key)
            found.append(key)
    found.sort(key=lambda p: (os.path.basename(p).lower(), p.lower()))
    return found


def _validate_weight_files_for_model(model, weight_paths):
    """Split candidate files into loadable and incompatible for this model."""
    valid = []
    skipped = []
    for path in weight_paths:
        try:
            model.load_weights(path)
            valid.append(path)
        except Exception as e:
            reason = str(e).splitlines()[0] if str(e) else type(e).__name__
            skipped.append((path, reason))
    return valid, skipped


def select_opponents(checkpoints, current_idx):
    """Select short-horizon (t-1) and long-horizon (t-5) opponents.

    Returns list of (label, filepath) pairs. May return 0-2 opponents
    depending on how many checkpoints exist.
    """
    opponents = []

    # t-1: one checkpoint back (~200 games ago)
    if current_idx >= 1:
        gc, fp = checkpoints[current_idx - 1]
        opponents.append((f"t-1 (g{gc})", fp))

    # t-5: five checkpoints back (~1000 games ago)
    if current_idx >= 5:
        gc, fp = checkpoints[current_idx - 5]
        opponents.append((f"t-5 (g{gc})", fp))
    elif current_idx >= 3:
        # Early training fallback: use t-3 (~600 games ago)
        gc, fp = checkpoints[current_idx - 3]
        opponents.append((f"t-3 (g{gc})", fp))

    return opponents


# ── Glicko-2 rating calculation ────────────────────────────────────────────
GLICKO2_RATING0 = 1500.0
GLICKO2_RD0 = 350.0
GLICKO2_VOL0 = 0.06
GLICKO2_TAU = 0.5
GLICKO2_EPSILON = 1e-6
GLICKO2_SCALE = 173.7178


def _normalize_rating_key(checkpoint_path):
    return os.path.normpath(checkpoint_path)


def _new_glicko2_entry(checkpoint_path):
    return {
        "checkpoint_path": _normalize_rating_key(checkpoint_path),
        "rating": float(GLICKO2_RATING0),
        "rd": float(GLICKO2_RD0),
        "vol": float(GLICKO2_VOL0),
        "games": 0,
        "periods": 0,
        "updated_unix": int(time.time()),
    }


def load_glicko2_ratings(path=GLICKO2_RATINGS_FILE):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                table = pickle.load(f)
            if isinstance(table, dict):
                return table
        except Exception:
            pass
    return {}


def save_glicko2_ratings(table, path=GLICKO2_RATINGS_FILE):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(table, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_glicko2_entry(table, checkpoint_path, create=True):
    if not checkpoint_path:
        return None
    key = _normalize_rating_key(checkpoint_path)
    entry = table.get(key)
    if entry is None and create:
        entry = _new_glicko2_entry(key)
        table[key] = entry
    return entry


def format_glicko2_entry(entry):
    if not entry:
        return "unrated"
    rating = float(entry.get("rating", GLICKO2_RATING0))
    rd = float(entry.get("rd", GLICKO2_RD0))
    vol = float(entry.get("vol", GLICKO2_VOL0))
    return f"R {rating:.0f} ±{2.0 * rd:.0f} (RD {rd:.0f}, v {vol:.3f})"


def _glicko2_to_internal(rating, rd):
    return (rating - GLICKO2_RATING0) / GLICKO2_SCALE, rd / GLICKO2_SCALE


def _glicko2_from_internal(mu, phi):
    return GLICKO2_SCALE * mu + GLICKO2_RATING0, GLICKO2_SCALE * phi


def _sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _glicko2_g(phi):
    return 1.0 / math.sqrt(1.0 + (3.0 * phi * phi) / (math.pi * math.pi))


def _glicko2_expectation(mu, mu_j, phi_j):
    return _sigmoid(_glicko2_g(phi_j) * (mu - mu_j))


def _glicko2_f(x, delta, phi, v, a, tau):
    ex = math.exp(x)
    num = ex * (delta * delta - phi * phi - v - ex)
    den = 2.0 * (phi * phi + v + ex) ** 2
    return (num / den) - ((x - a) / (tau * tau))


def glicko2_update_player(rating, rd, vol, matches,
                          tau=GLICKO2_TAU, epsilon=GLICKO2_EPSILON):
    """Update one player from a list of (opp_rating, opp_rd, score)."""
    if not matches:
        return float(rating), float(rd), float(vol)

    mu, phi = _glicko2_to_internal(rating, rd)
    a = math.log(vol * vol)

    v_inv = 0.0
    delta_sum = 0.0
    for opp_rating, opp_rd, score in matches:
        mu_j, phi_j = _glicko2_to_internal(opp_rating, opp_rd)
        g_phi_j = _glicko2_g(phi_j)
        expected = _glicko2_expectation(mu, mu_j, phi_j)
        v_inv += (g_phi_j * g_phi_j) * expected * (1.0 - expected)
        delta_sum += g_phi_j * (score - expected)

    if v_inv <= 0.0:
        return float(rating), float(rd), float(vol)

    v = 1.0 / v_inv
    delta = v * delta_sum

    A = a
    if delta * delta > phi * phi + v:
        B = math.log(delta * delta - phi * phi - v)
    else:
        k = 1
        while _glicko2_f(a - k * tau, delta, phi, v, a, tau) < 0.0:
            k += 1
            if k > 1000:
                break
        B = a - k * tau

    fA = _glicko2_f(A, delta, phi, v, a, tau)
    fB = _glicko2_f(B, delta, phi, v, a, tau)
    while abs(B - A) > epsilon:
        denom = fB - fA
        if abs(denom) < 1e-12:
            C = 0.5 * (A + B)
        else:
            C = A + (A - B) * fA / denom
        fC = _glicko2_f(C, delta, phi, v, a, tau)
        if fC * fB < 0.0:
            A = B
            fA = fB
        else:
            fA /= 2.0
        B = C
        fB = fC

    vol_prime = math.exp(A / 2.0)
    phi_star = math.sqrt(phi * phi + vol_prime * vol_prime)
    phi_prime = 1.0 / math.sqrt((1.0 / (phi_star * phi_star)) + (1.0 / v))
    mu_prime = mu + (phi_prime * phi_prime) * delta_sum
    rating_prime, rd_prime = _glicko2_from_internal(mu_prime, phi_prime)
    return float(rating_prime), float(rd_prime), float(vol_prime)


def glicko2_rating_delta(wins, losses, draws,
                         rating=GLICKO2_RATING0, rd=GLICKO2_RD0,
                         vol=GLICKO2_VOL0, opp_rating=GLICKO2_RATING0,
                         opp_rd=GLICKO2_RD0, tau=GLICKO2_TAU,
                         epsilon=GLICKO2_EPSILON):
    """Compute candidate rating change for one match period via Glicko-2."""
    outcomes = [1.0] * wins + [0.0] * losses + [0.5] * draws
    matches = [(opp_rating, opp_rd, s) for s in outcomes]
    rating_prime, _, _ = glicko2_update_player(
        rating, rd, vol, matches, tau=tau, epsilon=epsilon
    )
    return rating_prime - rating


def apply_match_glicko2_update(table, candidate_path, opponent_path,
                               wins, losses, draws):
    """Apply one match period update for both sides using stored ratings."""
    if not candidate_path or not opponent_path:
        return None
    cand_key = _normalize_rating_key(candidate_path)
    opp_key = _normalize_rating_key(opponent_path)
    if cand_key == opp_key:
        return None

    cand = get_glicko2_entry(table, cand_key, create=True)
    opp = get_glicko2_entry(table, opp_key, create=True)
    if cand is None or opp is None:
        return None

    total = int(wins + losses + draws)
    if total <= 0:
        return None

    cand_matches = (
        [(opp["rating"], opp["rd"], 1.0)] * int(wins) +
        [(opp["rating"], opp["rd"], 0.0)] * int(losses) +
        [(opp["rating"], opp["rd"], 0.5)] * int(draws)
    )
    opp_matches = (
        [(cand["rating"], cand["rd"], 0.0)] * int(wins) +
        [(cand["rating"], cand["rd"], 1.0)] * int(losses) +
        [(cand["rating"], cand["rd"], 0.5)] * int(draws)
    )

    cand_old = (cand["rating"], cand["rd"], cand["vol"])
    opp_old = (opp["rating"], opp["rd"], opp["vol"])
    cand_new = glicko2_update_player(*cand_old, cand_matches)
    opp_new = glicko2_update_player(*opp_old, opp_matches)

    now = int(time.time())
    cand["rating"], cand["rd"], cand["vol"] = cand_new
    cand["games"] = int(cand.get("games", 0)) + total
    cand["periods"] = int(cand.get("periods", 0)) + 1
    cand["updated_unix"] = now

    opp["rating"], opp["rd"], opp["vol"] = opp_new
    opp["games"] = int(opp.get("games", 0)) + total
    opp["periods"] = int(opp.get("periods", 0)) + 1
    opp["updated_unix"] = now

    return {
        "candidate": {
            "key": cand_key,
            "old": cand_old,
            "new": cand_new,
            "delta": cand_new[0] - cand_old[0],
            "entry": cand,
        },
        "opponent": {
            "key": opp_key,
            "old": opp_old,
            "new": opp_new,
            "delta": opp_new[0] - opp_old[0],
            "entry": opp,
        },
    }


# ── Result tallying ────────────────────────────────────────────────────────
def _tally_results(results, label, elapsed):
    """Tally match results and print summary.  Shared by both runners."""
    wins = losses = draws = 0
    wins_as_black = losses_as_black = draws_as_black = 0
    wins_as_white = losses_as_white = draws_as_white = 0
    move_totals = []
    black_win_moves = []    # moves in games candidate won as Black
    white_loss_moves = []   # moves in games candidate lost as White (survival)
    white_quick_losses = 0  # White losses in < 40 moves
    n_skipped = 0

    QUICK_LOSS_THRESHOLD = 40

    for candidate_won, n_moves, candidate_color in results:
        if candidate_won == "SKIP":
            n_skipped += 1
            continue
        move_totals.append(n_moves)
        as_black = (candidate_color == PLAYER1)

        if candidate_won is None:
            draws += 1
            if as_black:
                draws_as_black += 1
            else:
                draws_as_white += 1
        elif candidate_won:
            wins += 1
            if as_black:
                wins_as_black += 1
                black_win_moves.append(n_moves)
            else:
                wins_as_white += 1
        else:
            losses += 1
            if as_black:
                losses_as_black += 1
            else:
                losses_as_white += 1
                white_loss_moves.append(n_moves)
                if n_moves < QUICK_LOSS_THRESHOLD:
                    white_quick_losses += 1

    total = wins + losses + draws
    g2_delta = glicko2_rating_delta(wins, losses, draws)
    avg_moves = float(np.mean(move_totals)) if move_totals else 0.0
    win_pct = 100.0 * wins / max(1, total)

    black_total = wins_as_black + losses_as_black + draws_as_black
    white_total = wins_as_white + losses_as_white + draws_as_white
    black_pct = 100.0 * wins_as_black / max(1, black_total)
    white_pct = 100.0 * wins_as_white / max(1, white_total)
    avg_black_win_moves = float(np.mean(black_win_moves)) if black_win_moves else 0.0
    avg_white_loss_moves = float(np.mean(white_loss_moves)) if white_loss_moves else 0.0
    white_quick_loss_rate = (100.0 * white_quick_losses
                             / max(1, losses_as_white))

    print(f"  vs {label}:  "
          f"{wins}W {losses}L {draws}D  "
          f"({win_pct:.1f}%)  "
          f"Glicko-2 Δ{g2_delta:+.0f}  "
          f"Avg {avg_moves:.0f} moves  "
          f"[{elapsed:.0f}s]")
    print(f"    As Black: {wins_as_black}W {losses_as_black}L {draws_as_black}D "
          f"({black_pct:.1f}%)"
          f"{'  avg win in %.0f moves' % avg_black_win_moves if black_win_moves else ''}"
          f"  |  As White: {wins_as_white}W {losses_as_white}L {draws_as_white}D "
          f"({white_pct:.1f}%)"
          f"{'  avg loss in %.0f moves' % avg_white_loss_moves if white_loss_moves else ''}"
          f"{'  (%.0f%% quick)' % white_quick_loss_rate if losses_as_white else ''}")
    if n_skipped:
        print(f"    ⚠ {n_skipped} game(s) skipped (invalid opening)")

    return {
        "label": label,
        "wins": wins, "losses": losses, "draws": draws,
        "win_pct": win_pct, "glicko2_delta": g2_delta,
        "avg_moves": avg_moves, "elapsed": elapsed,
        "black_win_pct": black_pct, "white_win_pct": white_pct,
        "black_wins": wins_as_black, "black_losses": losses_as_black,
        "black_draws": draws_as_black,
        "white_wins": wins_as_white, "white_losses": losses_as_white,
        "white_draws": draws_as_white,
        "avg_black_win_moves": avg_black_win_moves,
        "avg_white_loss_moves": avg_white_loss_moves,
        "white_quick_loss_rate": white_quick_loss_rate,
    }


# ── Sequential match runner (GPU) ──────────────────────────────────────────
def _play_eval_game_seq(candidate_fn, opponent_fn, candidate_color,
                        sims, batch_size, opening, opp_sims=None):
    """Play one deterministic eval game using compiled predict functions."""
    game = GomokuGame()

    for r, c in opening:
        reward, done = game.make_move(r, c)
        if done:
            return "SKIP", 0, candidate_color

    done = False
    move_num = len(game.move_history)
    reward = 0

    while not done:
        if game.current_player == candidate_color:
            fn = candidate_fn
            move_sims = sims
        else:
            fn = opponent_fn
            move_sims = opp_sims if opp_sims is not None else sims

        root = mcts_search_batched(
            game, fn,
            num_simulations=move_sims,
            batch_size=batch_size,
            c_puct=1.5,
            add_noise=False,
        )
        pi = mcts_policy(root, temperature=0.0)
        idx = int(np.argmax(pi))
        row, col = divmod(idx, BOARD_SIZE)

        # HARD TRIPWIRE: crash on illegal moves
        assert game.board[row, col] == 0, (
            f"ILLEGAL MOVE ({row},{col}) at move {move_num}, "
            f"board[{row},{col}]={game.board[row, col]}, "
            f"stones={int(np.count_nonzero(game.board))}"
        )

        reward, done = game.make_move(row, col)
        move_num += 1

    if reward == 0:
        return None, move_num, candidate_color
    # reward == 1  => current player made 5-in-a-row
    # reward == -1 => current player made an illegal move, so they lose
    if reward == 1:
        winner = game.current_player
    elif reward == -1:
        winner = -game.current_player
    else:
        raise RuntimeError(f"Unexpected terminal reward: {reward}")
    candidate_won = (winner == candidate_color)
    return candidate_won, move_num, candidate_color


def run_match_sequential(candidate_model, opponent_model, label, openings,
                         sims=EVAL_SIMS, batch_size=8, opp_sims=None):
    """Run a full evaluation match sequentially on GPU.

    Takes pre-loaded model objects (no file I/O, no multiprocessing).
    Each opening is played twice (color swap).  Deterministic play.
    Compiles models via @tf.function for faster inference.

    sims: MCTS sims for candidate side.
    opp_sims: if set, uses a different MCTS sims budget for opponent side.
    """
    # Compile both models once (no-op if already compiled)
    cand_fn = make_predict_fn(candidate_model)
    opp_fn = make_predict_fn(opponent_model)

    # Warmup traces
    _dummy = np.zeros((1, BOARD_SIZE, BOARD_SIZE, NUM_INPUT_PLANES), dtype=np.float32)
    cand_fn(_dummy); opp_fn(_dummy)

    games = []
    for opening in openings:
        games.append((PLAYER1, opening))
        games.append((PLAYER2, opening))

    n_games = len(games)
    t0 = time.time()
    results = []
    for i, (cand_color, opening) in enumerate(games, 1):
        res = _play_eval_game_seq(
            cand_fn, opp_fn, cand_color,
            sims, batch_size, opening, opp_sims=opp_sims,
        )
        results.append(res)
        print(f"  Eval vs {label}: {i}/{n_games} games", end="\r", flush=True)
    elapsed = time.time() - t0
    print(" " * 60, end="\r", flush=True)
    return _tally_results(results, label, elapsed)


# ── Tournament mode ──────────────────────────────────────────────────────────
def _normal_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def glicko2_pairwise_win_prob(entry_a, entry_b):
    """Approximate P(A > B) using rating diff and combined RD."""
    ra = float(entry_a.get("rating", GLICKO2_RATING0))
    rb = float(entry_b.get("rating", GLICKO2_RATING0))
    rda = max(1e-9, float(entry_a.get("rd", GLICKO2_RD0)))
    rdb = max(1e-9, float(entry_b.get("rd", GLICKO2_RD0)))
    sigma = math.sqrt(rda * rda + rdb * rdb)
    z = (ra - rb) / sigma
    return min(1.0, max(0.0, _normal_cdf(z)))


def _take_openings(openings, cursor, count):
    """Take `count` openings from a deterministic cyclic cursor."""
    if count <= 0 or not openings:
        return [], cursor
    n = len(openings)
    out = []
    for k in range(count):
        out.append(openings[(cursor + k) % n])
    return out, cursor + count


def _tourney_rows(ratings_table, weight_paths):
    rows = []
    for path in weight_paths:
        entry = get_glicko2_entry(ratings_table, path, create=True)
        rows.append((path, entry))
    rows.sort(
        key=lambda x: (
            -float(x[1].get("rating", GLICKO2_RATING0)),
            float(x[1].get("rd", GLICKO2_RD0)),
            os.path.basename(x[0]).lower(),
        )
    )
    return rows


def _print_tourney_standings(ratings_table, weight_paths):
    rows = _tourney_rows(ratings_table, weight_paths)
    print("  Standings:")
    for i, (path, entry) in enumerate(rows, 1):
        print(f"    {i:2d}. {os.path.basename(path):30s}  "
              f"{format_glicko2_entry(entry)}  "
              f"games {int(entry.get('games', 0))}")
    return rows


def _top_two_contenders_confident(rows, confidence_prob, min_games):
    if len(rows) < 2:
        return False

    top_entries = [rows[0][1], rows[1][1]]
    if any(int(e.get("games", 0)) < min_games for e in top_entries):
        return False

    rest = rows[2:]
    if not rest:
        return True

    for _, contender_entry in rows[:2]:
        for _, other_entry in rest:
            p = glicko2_pairwise_win_prob(contender_entry, other_entry)
            if p < confidence_prob:
                return False
    return True


def _run_round_robin_to_find_contenders(
    model_a, model_b, weight_paths, openings,
    rr_sims=TOURNEY_RR_SIMS,
    rr_openings=TOURNEY_RR_OPENINGS,
    max_rounds=TOURNEY_RR_MAX_ROUNDS,
    contender_prob=TOURNEY_RR_CONTENDER_PROB,
    min_games=TOURNEY_RR_MIN_GAMES,
):
    """Run repeated RR rounds until top-2 contenders are clear or capped."""
    ratings_table = {}
    for path in weight_paths:
        get_glicko2_entry(ratings_table, path, create=True)

    pairs = []
    for i in range(len(weight_paths)):
        for j in range(i + 1, len(weight_paths)):
            pairs.append((weight_paths[i], weight_paths[j]))

    cursor = 0
    confident = False
    rounds_played = 0

    for round_idx in range(max_rounds):
        rounds_played = round_idx + 1
        round_openings, cursor = _take_openings(openings, cursor, rr_openings)
        if not round_openings:
            break

        print(f"\nRound-robin round {rounds_played}/{max_rounds} "
              f"@ {rr_sims} sims, {len(round_openings) * 2} games per pairing")
        for left_path, right_path in pairs:
            left_name = os.path.basename(left_path)
            right_name = os.path.basename(right_path)
            label = f"{left_name} vs {right_name} [RR {rounds_played}]"

            model_a.load_weights(left_path)
            model_b.load_weights(right_path)
            result = run_match_sequential(
                model_a, model_b, label,
                openings=round_openings,
                sims=rr_sims,
                batch_size=EVAL_BATCH_SIZE,
            )
            apply_match_glicko2_update(
                ratings_table,
                left_path,
                right_path,
                wins=result["wins"],
                losses=result["losses"],
                draws=result["draws"],
            )
            print()

        rows = _print_tourney_standings(ratings_table, weight_paths)
        confident = _top_two_contenders_confident(
            rows,
            confidence_prob=contender_prob,
            min_games=min_games,
        )
        if confident:
            print("  Top-2 contenders are now confident enough to advance.")
            break

    rows = _tourney_rows(ratings_table, weight_paths)
    contenders = [rows[0][0], rows[1][0]]
    return {
        "contenders": contenders,
        "rows": rows,
        "ratings_table": ratings_table,
        "rounds_played": rounds_played,
        "confident": confident,
        "opening_cursor": cursor,
    }


def _run_heads_up_until_confident(
    model_a, model_b, left_path, right_path, openings,
    opening_cursor=0,
    phases=TOURNEY_H2H_PHASES,
    openings_per_batch=TOURNEY_H2H_OPENINGS_PER_BATCH,
    confidence_prob=TOURNEY_H2H_CONFIDENCE_PROB,
    min_games=TOURNEY_H2H_MIN_GAMES,
    rating_period_games=TOURNEY_H2H_RATING_PERIOD_GAMES,
    rd_target=TOURNEY_H2H_TARGET_RD,
):
    """Run staged heads-up batches until confidence and RD targets are met."""
    ratings_table = {}
    get_glicko2_entry(ratings_table, left_path, create=True)
    get_glicko2_entry(ratings_table, right_path, create=True)

    left_name = os.path.basename(left_path)
    right_name = os.path.basename(right_path)
    model_a.load_weights(left_path)
    model_b.load_weights(right_path)

    batch_idx = 0
    confident = False
    winner = None
    conf = 0.5
    cursor = opening_cursor
    min_seen_games = 0
    max_rd = float("inf")
    pending_wins = pending_losses = pending_draws = 0
    pending_games = 0

    def _flush_pending():
        nonlocal pending_wins, pending_losses, pending_draws, pending_games
        nonlocal winner, conf, confident, min_seen_games, max_rd
        if pending_games <= 0:
            return
        apply_match_glicko2_update(
            ratings_table,
            left_path,
            right_path,
            wins=pending_wins,
            losses=pending_losses,
            draws=pending_draws,
        )
        pending_wins = pending_losses = pending_draws = 0
        pending_games = 0

        left_entry = get_glicko2_entry(ratings_table, left_path, create=False)
        right_entry = get_glicko2_entry(ratings_table, right_path, create=False)
        p_left = glicko2_pairwise_win_prob(left_entry, right_entry)
        conf = max(p_left, 1.0 - p_left)
        winner = left_path if p_left >= 0.5 else right_path
        min_seen_games = min(
            int(left_entry.get("games", 0)),
            int(right_entry.get("games", 0)),
        )
        max_rd = max(
            float(left_entry.get("rd", GLICKO2_RD0)),
            float(right_entry.get("rd", GLICKO2_RD0)),
        )

        print(f"    Final rating: {left_name} -> {format_glicko2_entry(left_entry)}")
        print(f"    Final rating: {right_name} -> {format_glicko2_entry(right_entry)}")
        print(f"    Current leader: {os.path.basename(winner)}  "
              f"(confidence {100.0 * conf:.1f}%, games {min_seen_games}, "
              f"max RD {max_rd:.1f})")
        print()

        if (conf >= confidence_prob and
                min_seen_games >= min_games and
                max_rd <= rd_target):
            confident = True

    for sims, n_batches in phases:
        for _ in range(n_batches):
            batch_idx += 1
            batch_openings, cursor = _take_openings(openings, cursor, openings_per_batch)
            if not batch_openings:
                break

            label = f"{left_name} vs {right_name} [Final {batch_idx}, {sims} sims]"
            result = run_match_sequential(
                model_a, model_b, label,
                openings=batch_openings,
                sims=sims,
                batch_size=EVAL_BATCH_SIZE,
            )
            pending_wins += int(result["wins"])
            pending_losses += int(result["losses"])
            pending_draws += int(result["draws"])
            pending_games += int(result["wins"] + result["losses"] + result["draws"])

            if pending_games >= rating_period_games:
                _flush_pending()
                if confident:
                    break
        if confident:
            break

    _flush_pending()

    left_entry = get_glicko2_entry(ratings_table, left_path, create=False)
    right_entry = get_glicko2_entry(ratings_table, right_path, create=False)
    if winner is None:
        winner = left_path if left_entry["rating"] >= right_entry["rating"] else right_path
        p_left = glicko2_pairwise_win_prob(left_entry, right_entry)
        conf = max(p_left, 1.0 - p_left)

    return {
        "winner": winner,
        "confident": confident,
        "confidence": conf,
        "left_entry": left_entry,
        "right_entry": right_entry,
        "batches_played": batch_idx,
        "max_rd": max(
            float(left_entry.get("rd", GLICKO2_RD0)),
            float(right_entry.get("rd", GLICKO2_RD0)),
        ),
        "opening_cursor": cursor,
        "ratings_table": ratings_table,
    }


def run_tournament(args):
    """Best-of-best tournament mode over all weights in a folder."""
    discovered = find_weight_files(args.tournament_dir)
    if len(discovered) < 2:
        print(f"Need at least two weights in {args.tournament_dir}; found {len(discovered)}")
        return

    print(f"Tournament folder: {args.tournament_dir}")
    print(f"Discovered {len(discovered)} candidate files:")
    for i, p in enumerate(discovered, 1):
        print(f"  {i:2d}. {os.path.basename(p)}")
    print()

    model_probe = create_model()
    weight_paths, skipped = _validate_weight_files_for_model(model_probe, discovered)
    if skipped:
        print("Skipping incompatible files:")
        for path, reason in skipped:
            print(f"  - {os.path.basename(path)}: {reason}")
        print()

    if len(weight_paths) < 2:
        print("Need at least two compatible weight files after validation.")
        return

    if len(weight_paths) != len(discovered):
        print(f"Using {len(weight_paths)} compatible weight files:")
        for i, p in enumerate(weight_paths, 1):
            print(f"  {i:2d}. {os.path.basename(p)}")
        print()

    rr_needed = 0
    if len(weight_paths) > 2:
        rr_needed = TOURNEY_RR_MAX_ROUNDS * TOURNEY_RR_OPENINGS
    h2h_needed = TOURNEY_H2H_OPENINGS_PER_BATCH * sum(b for _, b in TOURNEY_H2H_PHASES)
    needed_openings = max(1, rr_needed + h2h_needed)
    openings = load_or_create_openings(
        needed_openings,
        n_plies=args.plies,
        seed=args.seed,
    )
    if not openings:
        print("Failed to build opening book for tournament.")
        return

    print(f"Tournament openings pool: {len(openings)} positions")
    print("Ratings: transient (not writing to persistent glicko2 file)")
    print()

    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU: {gpus[0].name}")
    else:
        print("No GPU detected — eval will be slow.")

    model_a = model_probe
    model_b = create_model()

    opening_cursor = 0
    if len(weight_paths) == 2:
        contenders = weight_paths
        print("\nTwo-player tournament: skipping round robin.")
    else:
        rr = _run_round_robin_to_find_contenders(
            model_a=model_a,
            model_b=model_b,
            weight_paths=weight_paths,
            openings=openings,
            rr_sims=TOURNEY_RR_SIMS,
            rr_openings=TOURNEY_RR_OPENINGS,
            max_rounds=TOURNEY_RR_MAX_ROUNDS,
            contender_prob=TOURNEY_RR_CONTENDER_PROB,
            min_games=TOURNEY_RR_MIN_GAMES,
        )
        contenders = rr["contenders"]
        opening_cursor = rr["opening_cursor"]

        print("\nRound-robin result:")
        print(f"  Rounds played: {rr['rounds_played']}")
        print(f"  Confident top-2: {'yes' if rr['confident'] else 'no (using best-rated top-2)'}")
        print(f"  Contenders: {os.path.basename(contenders[0])} vs "
              f"{os.path.basename(contenders[1])}")

    print("\nHeads-up final configuration:")
    for sims, n_batches in TOURNEY_H2H_PHASES:
        n_games = n_batches * TOURNEY_H2H_OPENINGS_PER_BATCH * 2
        print(f"  {sims} sims: up to {n_games} games")
    print(f"  Glicko period size: {TOURNEY_H2H_RATING_PERIOD_GAMES} games")
    print(f"  Confidence target: {100.0 * TOURNEY_H2H_CONFIDENCE_PROB:.1f}%")
    print(f"  Minimum games before crowning: {TOURNEY_H2H_MIN_GAMES}")
    print(f"  RD target: <= {TOURNEY_H2H_TARGET_RD:.1f}")
    print()

    final = _run_heads_up_until_confident(
        model_a=model_a,
        model_b=model_b,
        left_path=contenders[0],
        right_path=contenders[1],
        openings=openings,
        opening_cursor=opening_cursor,
        phases=TOURNEY_H2H_PHASES,
        openings_per_batch=TOURNEY_H2H_OPENINGS_PER_BATCH,
        confidence_prob=TOURNEY_H2H_CONFIDENCE_PROB,
        min_games=TOURNEY_H2H_MIN_GAMES,
        rating_period_games=TOURNEY_H2H_RATING_PERIOD_GAMES,
        rd_target=TOURNEY_H2H_TARGET_RD,
    )

    winner = final["winner"]
    runner_up = contenders[1] if winner == contenders[0] else contenders[0]
    print("=" * 60)
    print("Tournament final result")
    print("=" * 60)
    print(f"  Champion: {os.path.basename(winner)}")
    print(f"  Runner-up: {os.path.basename(runner_up)}")
    print(f"  Confidence: {100.0 * final['confidence']:.1f}% "
          f"({'confident' if final['confident'] else 'not fully confident at cap'})")
    print(f"  Max RD: {final['max_rd']:.1f} "
          f"({'target met' if final['max_rd'] <= TOURNEY_H2H_TARGET_RD else 'above target'})")
    print(f"  Batches played: {final['batches_played']}")
    print(f"  {os.path.basename(contenders[0])}: {format_glicko2_entry(final['left_entry'])}")
    print(f"  {os.path.basename(contenders[1])}: {format_glicko2_entry(final['right_entry'])}")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    parser = argparse.ArgumentParser(
        description="Evaluate Gomoku checkpoints against older baselines.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to candidate checkpoint (default: latest)")
    parser.add_argument("--openings", type=int, default=EVAL_GAMES // 2,
                        help=f"Number of opening positions "
                             f"(default: {EVAL_GAMES // 2}, "
                             f"→ {EVAL_GAMES} games with color swap)")
    parser.add_argument("--sims", type=int, default=EVAL_SIMS,
                        help=f"MCTS sims per move (default: {EVAL_SIMS})")
    parser.add_argument("--seed", type=int, default=EVAL_OPENING_SEED,
                        help=f"Opening RNG seed (default: {EVAL_OPENING_SEED})")
    parser.add_argument("--plies", type=int, default=EVAL_OPENING_PLIES,
                        help=f"Random plies per opening (default: {EVAL_OPENING_PLIES})")
    parser.add_argument("--weights-dir", type=str, default="weights",
                        help="Directory containing checkpoints")
    parser.add_argument("--tournament-dir", type=str, default=None,
                        help="Run best-of-best tournament over all weight files in directory")
    parser.add_argument("--calibrate-sims", action="store_true",
                        help="Calibrate strength gaps between sim budgets "
                             "on one checkpoint")
    parser.add_argument("--sim-levels", type=str, default="100,400,1600",
                        help="Comma-separated sims for calibration mode "
                             "(default: 100,400,1600)")
    parser.add_argument("--no-rating-update", action="store_true",
                        help="Do not update persistent Glicko-2 ratings")
    args = parser.parse_args()

    if args.tournament_dir:
        if args.calibrate_sims:
            parser.error("--calibrate-sims cannot be combined with --tournament-dir")
        run_tournament(args)
        return

    checkpoints = find_checkpoints(args.weights_dir)
    if not checkpoints:
        print("No checkpoints found in", args.weights_dir)
        return

    print(f"Found {len(checkpoints)} checkpoints "
          f"(g{checkpoints[0][0]:05d} – g{checkpoints[-1][0]:05d})")

    # Determine candidate
    candidate_key = None
    if args.checkpoint:
        candidate_path = args.checkpoint
        candidate_idx = None
        for i, (gc, fp) in enumerate(checkpoints):
            if fp == candidate_path:
                candidate_idx = i
                break

        if candidate_idx is None:
            # Not a timestamped checkpoint — try to resolve via best_checkpoint.pkl
            best_state_file = os.path.join(args.weights_dir, "best_checkpoint.pkl")
            resolved_gc = None
            if os.path.exists(best_state_file):
                import pickle
                try:
                    with open(best_state_file, "rb") as f:
                        best_state = pickle.load(f)
                    resolved_gc = best_state.get("game_count")
                except Exception:
                    pass

            if resolved_gc is not None:
                # Find the checkpoint closest to the resolved game count
                for i, (gc, fp) in enumerate(checkpoints):
                    if abs(gc - resolved_gc) <= 10:
                        candidate_idx = i
                        break
                if candidate_idx is not None:
                    print(f"Candidate: {candidate_path} (best @ g{resolved_gc:05d})")
                    candidate_key = checkpoints[candidate_idx][1]
                else:
                    candidate_idx = len(checkpoints) - 1
                    print(f"Candidate: {candidate_path} "
                          f"(best @ g{resolved_gc:05d}, using latest for opponents)")
                    candidate_key = os.path.normpath(candidate_path)
            else:
                candidate_idx = len(checkpoints) - 1
                print(f"Candidate: {candidate_path} (using latest for opponent selection)")
                candidate_key = os.path.normpath(candidate_path)
        else:
            candidate_gc = checkpoints[candidate_idx][0]
            print(f"Candidate: g{candidate_gc:05d}  ({candidate_path})")
            candidate_key = checkpoints[candidate_idx][1]
    else:
        candidate_idx = len(checkpoints) - 1
        candidate_path = checkpoints[candidate_idx][1]
        candidate_gc = checkpoints[candidate_idx][0]
        print(f"Candidate: g{candidate_gc:05d}  ({candidate_path})")
        candidate_key = checkpoints[candidate_idx][1]

    candidate_gc = checkpoints[candidate_idx][0]
    if candidate_key is None:
        candidate_key = checkpoints[candidate_idx][1]

    # Load or create fixed opening book (same for all matchups and runs)
    openings = load_or_create_openings(args.openings, n_plies=args.plies,
                                       seed=args.seed)
    # Use requested number (may be fewer than saved if book is smaller)
    openings = openings[:args.openings]
    n_games = len(openings) * 2

    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU: {gpus[0].name}")
    else:
        print("No GPU detected — eval will be slow.")

    # Load candidate model once (reused for all matchups)
    candidate_model = create_model()
    candidate_model.load_weights(candidate_path)

    if args.calibrate_sims:
        try:
            sim_levels = _parse_sim_levels(args.sim_levels)
        except ValueError as e:
            parser.error(str(e))

        print("Mode: sims calibration (single checkpoint, deterministic)")
        print("Rating updates: disabled in calibration mode")
        print(f"Sims levels: {', '.join(str(v) for v in sim_levels)}")
        print(f"Openings: {len(openings)} positions × 2 colors = {n_games} games per matchup")
        print()

        cal_rows = []
        g2_by_pair = {}
        for i in range(len(sim_levels)):
            for j in range(i + 1, len(sim_levels)):
                low = sim_levels[i]
                high = sim_levels[j]
                label = f"{high} sims vs {low} sims"
                result = run_match_sequential(
                    candidate_model, candidate_model, label,
                    openings=openings,
                    sims=high,
                    opp_sims=low,
                    batch_size=EVAL_BATCH_SIZE,
                )
                cal_rows.append((high, low, result))
                g2_by_pair[(low, high)] = result["glicko2_delta"]
                print()

        print("=" * 60)
        print(f"Sims calibration summary for g{candidate_gc:05d}")
        print("=" * 60)
        for high, low, r in cal_rows:
            print(f"  {high:4d} vs {low:4d} sims  "
                  f"{r['win_pct']:5.1f}%  Glicko-2 Δ{r['glicko2_delta']:+4.0f}")

        if len(sim_levels) >= 3:
            print("\nAdjacent tier gaps:")
            for low, high in zip(sim_levels[:-1], sim_levels[1:]):
                g2 = g2_by_pair.get((low, high))
                if g2 is None:
                    continue
                print(f"  {low:4d} → {high:4d}: Glicko-2 Δ{g2:+.0f}")
        return

    opponents = select_opponents(checkpoints, candidate_idx)
    if not opponents:
        print("Not enough checkpoints for comparison yet.")
        return

    print(f"Opponents: {', '.join(lbl for lbl, _ in opponents)}")
    print(f"Openings: {len(openings)} positions × 2 colors = {n_games} games")
    print(f"MCTS: {args.sims} sims, deterministic (no noise, temp=0)")
    print(f"Ratings: {'off' if args.no_rating_update else 'on'} "
          f"({GLICKO2_RATINGS_FILE})")
    print()

    # Reusable opponent model (swap weights per matchup)
    opp_model = create_model()
    ratings_table = load_glicko2_ratings()
    ratings_changed = False

    all_results = []
    for label, opp_path in opponents:
        opp_model.load_weights(opp_path)
        result = run_match_sequential(
            candidate_model, opp_model, label,
            openings=openings,
            sims=args.sims,
            batch_size=EVAL_BATCH_SIZE,
        )
        if not args.no_rating_update:
            upd = apply_match_glicko2_update(
                ratings_table,
                candidate_key,
                opp_path,
                wins=result["wins"],
                losses=result["losses"],
                draws=result["draws"],
            )
            if upd:
                ratings_changed = True
                cand_entry = upd["candidate"]["entry"]
                opp_entry = upd["opponent"]["entry"]
                print(f"    Rating update: cand {format_glicko2_entry(cand_entry)}  "
                      f"|  opp {format_glicko2_entry(opp_entry)}")
        all_results.append(result)
        print()

    if ratings_changed:
        save_glicko2_ratings(ratings_table)
        cand_entry = get_glicko2_entry(ratings_table, candidate_key, create=False)
        if cand_entry:
            print(f"Saved ratings → {GLICKO2_RATINGS_FILE}  "
                  f"(candidate now {format_glicko2_entry(cand_entry)})")
            print()

    # Summary
    print("=" * 60)
    print(f"Eval summary for g{candidate_gc:05d}")
    print("=" * 60)
    for r in all_results:
        status = "✓ IMPROVING" if r["win_pct"] > 55 else \
                 "✗ REGRESSED" if r["win_pct"] < 45 else \
                 "~ STABLE"
        print(f"  vs {r['label']:20s}  {r['win_pct']:5.1f}%  "
              f"Glicko-2 Δ{r['glicko2_delta']:+4.0f}  "
              f"{status}")

    # Quick verdict
    if all(r["win_pct"] > 55 for r in all_results):
        print("\n→ Clear improvement across all baselines.")
    elif any(r["win_pct"] < 45 for r in all_results):
        print("\n→ Possible regression detected. Investigate before continuing.")
    else:
        print("\n→ Stable / marginal improvement.")


if __name__ == "__main__":
    main()
