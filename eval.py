#!/usr/bin/env python3
"""
Gomoku — Evaluate checkpoints against older baselines.

Runs deterministic matches (no noise, temperature=0) between two
checkpoints on GPU and reports win rates, per-color stats, average game
length, and Elo delta.

Usage:
    # Auto-detect latest checkpoint, eval vs t-1 and t-5:
    python eval.py

    # Evaluate a specific checkpoint:
    python eval.py --checkpoint weights/gomoku_best.weights.h5

    # Custom opening count:
    python eval.py --openings 200
"""

import argparse
import glob
import math
import os
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


# ── Opening book generation ─────────────────────────────────────────────────
OPENING_BOOK_FILE = "weights/eval_openings.pkl"

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


# ── Elo calculation ─────────────────────────────────────────────────────────
def elo_delta(wins, losses, draws):
    """Compute Elo delta from match results.

    Uses the standard formula: Elo_diff = 400 * log10(score / (1 - score))
    where score = (wins + 0.5 * draws) / total.
    Clamps to avoid ±infinity at 0% or 100%.
    """
    total = wins + losses + draws
    if total == 0:
        return 0.0
    score = (wins + 0.5 * draws) / total
    score = max(0.005, min(0.995, score))   # clamp
    return 400.0 * math.log10(score / (1.0 - score))


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
    elo = elo_delta(wins, losses, draws)
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
          f"Elo Δ{elo:+.0f}  "
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
        "win_pct": win_pct, "elo_delta": elo,
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
                        sims, batch_size, opening):
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
        else:
            fn = opponent_fn

        root = mcts_search_batched(
            game, fn,
            num_simulations=sims,
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
                         sims=EVAL_SIMS, batch_size=8):
    """Run a full evaluation match sequentially on GPU.

    Takes pre-loaded model objects (no file I/O, no multiprocessing).
    Each opening is played twice (color swap).  Deterministic play.
    Compiles models via @tf.function for faster inference.
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
            sims, batch_size, opening,
        )
        results.append(res)
        print(f"  Eval vs {label}: {i}/{n_games} games", end="\r", flush=True)
    elapsed = time.time() - t0
    print(" " * 60, end="\r", flush=True)
    return _tally_results(results, label, elapsed)


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
    args = parser.parse_args()

    checkpoints = find_checkpoints(args.weights_dir)
    if not checkpoints:
        print("No checkpoints found in", args.weights_dir)
        return

    print(f"Found {len(checkpoints)} checkpoints "
          f"(g{checkpoints[0][0]:05d} – g{checkpoints[-1][0]:05d})")

    # Determine candidate
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
                else:
                    candidate_idx = len(checkpoints) - 1
                    print(f"Candidate: {candidate_path} "
                          f"(best @ g{resolved_gc:05d}, using latest for opponents)")
            else:
                candidate_idx = len(checkpoints) - 1
                print(f"Candidate: {candidate_path} (using latest for opponent selection)")
        else:
            candidate_gc = checkpoints[candidate_idx][0]
            print(f"Candidate: g{candidate_gc:05d}  ({candidate_path})")
    else:
        candidate_idx = len(checkpoints) - 1
        candidate_path = checkpoints[candidate_idx][1]
        candidate_gc = checkpoints[candidate_idx][0]
        print(f"Candidate: g{candidate_gc:05d}  ({candidate_path})")

    opponents = select_opponents(checkpoints, candidate_idx)
    candidate_gc = checkpoints[candidate_idx][0]
    if not opponents:
        print("Not enough checkpoints for comparison yet.")
        return

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

    print(f"Opponents: {', '.join(lbl for lbl, _ in opponents)}")
    print(f"Openings: {len(openings)} positions × 2 colors = {n_games} games")
    print(f"MCTS: {args.sims} sims, deterministic (no noise, temp=0)")
    print()

    # Load candidate model once (reused for all matchups)
    candidate_model = create_model()
    candidate_model.load_weights(candidate_path)

    # Reusable opponent model (swap weights per matchup)
    opp_model = create_model()

    all_results = []
    for label, opp_path in opponents:
        opp_model.load_weights(opp_path)
        result = run_match_sequential(
            candidate_model, opp_model, label,
            openings=openings,
            sims=args.sims,
            batch_size=EVAL_BATCH_SIZE,
        )
        all_results.append(result)
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
              f"Elo Δ{r['elo_delta']:+4.0f}  "
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
