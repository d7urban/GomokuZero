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

    # Run Swiss+McMahon tournament over all weights in a folder:
    python eval.py --tournament-dir botb-weights
"""

import argparse
import glob
import os
import pickle
import re
import time

import numpy as np

# Set before any TF import (gomoku.py imports TF at module level).
# setdefault lets train.py's explicit settings take precedence when importing eval.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

from gomoku import (
    BOARD_SIZE, PLAYER1, PLAYER2, NUM_INPUT_PLANES,
    GomokuGame, create_model, make_predict_fn,
    mcts_search_batched, mcts_policy,
)
from entrypoint_shared import MODEL_ARCH_CANDIDATES
from book_openings import (
    EVAL_OPENING_PLIES,
    EVAL_OPENING_SEED,
    load_or_create_openings,
)
from ratings_glicko2 import (
    GLICKO2_RATINGS_FILE,
    apply_match_glicko2_update,
    format_glicko2_entry,
    get_glicko2_entry,
    glicko2_rating_delta,
    load_glicko2_ratings,
    save_glicko2_ratings,
)

# ── Defaults ────────────────────────────────────────────────────────────────
EVAL_SIMS         = 100    # MCTS simulations per move (same for both sides)
EVAL_BATCH_SIZE   = 8      # MCTS batch size
EVAL_GAMES        = 200    # games per matchup (100 openings × 2 color swaps)


def _create_model_for_arch(arch):
    num_res_blocks, num_filters = arch
    return create_model(
        num_res_blocks=num_res_blocks,
        num_filters=num_filters,
    )


def _load_model_with_arch_fallback(weight_path):
    errors = []
    for arch in MODEL_ARCH_CANDIDATES:
        model = _create_model_for_arch(arch)
        try:
            model.load_weights(weight_path)
            return model, arch
        except Exception as e:
            reason = str(e).splitlines()[0] if str(e) else type(e).__name__
            errors.append(f"{arch[0]}x{arch[1]}: {reason}")

    details = "; ".join(errors) if errors else "no loader attempts"
    raise ValueError(
        f"Unable to load weights '{weight_path}'. Tried architectures: {details}"
    )


def _format_arch(arch):
    return f"{arch[0]}x{arch[1]}"


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


# ── Result tallying ────────────────────────────────────────────────────────
def _new_tally_state():
    return {
        "wins": 0, "losses": 0, "draws": 0,
        "wins_as_black": 0, "losses_as_black": 0, "draws_as_black": 0,
        "wins_as_white": 0, "losses_as_white": 0, "draws_as_white": 0,
        "move_totals": [],
        "black_win_moves": [],
        "white_loss_moves": [],
        "white_quick_losses": 0,
        "n_skipped": 0,
    }


def _accumulate_tally(state, candidate_won, n_moves, candidate_color):
    quick_loss_threshold = 40
    if candidate_won == "SKIP":
        state["n_skipped"] += 1
        return

    state["move_totals"].append(n_moves)
    as_black = candidate_color == PLAYER1

    if candidate_won is None:
        state["draws"] += 1
        if as_black:
            state["draws_as_black"] += 1
        else:
            state["draws_as_white"] += 1
        return

    if candidate_won:
        state["wins"] += 1
        if as_black:
            state["wins_as_black"] += 1
            state["black_win_moves"].append(n_moves)
        else:
            state["wins_as_white"] += 1
        return

    state["losses"] += 1
    if as_black:
        state["losses_as_black"] += 1
    else:
        state["losses_as_white"] += 1
        state["white_loss_moves"].append(n_moves)
        if n_moves < quick_loss_threshold:
            state["white_quick_losses"] += 1


def _finalize_tally(state, label, elapsed):
    wins = state["wins"]
    losses = state["losses"]
    draws = state["draws"]
    wins_as_black = state["wins_as_black"]
    losses_as_black = state["losses_as_black"]
    draws_as_black = state["draws_as_black"]
    wins_as_white = state["wins_as_white"]
    losses_as_white = state["losses_as_white"]
    draws_as_white = state["draws_as_white"]
    black_win_moves = state["black_win_moves"]
    white_loss_moves = state["white_loss_moves"]

    total = wins + losses + draws
    g2_delta = glicko2_rating_delta(wins, losses, draws)
    avg_moves = float(np.mean(state["move_totals"])) if state["move_totals"] else 0.0
    win_pct = 100.0 * wins / max(1, total)

    black_total = wins_as_black + losses_as_black + draws_as_black
    white_total = wins_as_white + losses_as_white + draws_as_white
    black_pct = 100.0 * wins_as_black / max(1, black_total)
    white_pct = 100.0 * wins_as_white / max(1, white_total)
    avg_black_win_moves = float(np.mean(black_win_moves)) if black_win_moves else 0.0
    avg_white_loss_moves = float(np.mean(white_loss_moves)) if white_loss_moves else 0.0
    white_quick_loss_rate = (100.0 * state["white_quick_losses"]
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
    if state["n_skipped"]:
        print(f"    ⚠ {state['n_skipped']} game(s) skipped (invalid opening)")

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


def _tally_results(results, label, elapsed):
    """Tally match results and print summary.  Shared by both runners."""
    state = _new_tally_state()
    for candidate_won, n_moves, candidate_color in results:
        _accumulate_tally(state, candidate_won, n_moves, candidate_color)
    return _finalize_tally(state, label, elapsed)


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


def _print_eval_gpu_status():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU: {gpus[0].name}")
    else:
        print("No GPU detected — eval will be slow.")


def run_tournament(args):
    """Swiss + McMahon tournament mode over all weights in a folder."""
    from eval_tournament import run_tournament as run_tournament_impl

    run_tournament_impl(
        args=args,
        run_match_fn=run_match_sequential,
        eval_batch_size=EVAL_BATCH_SIZE,
        print_gpu_status_fn=_print_eval_gpu_status,
    )


def _build_eval_parser():
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
    return parser


def _find_checkpoint_index(checkpoints, candidate_path):
    for i, (_, fp) in enumerate(checkpoints):
        if fp == candidate_path:
            return i
    return None


def _resolve_best_state_game_count(weights_dir):
    best_state_file = os.path.join(weights_dir, "best_checkpoint.pkl")
    if not os.path.exists(best_state_file):
        return None
    try:
        with open(best_state_file, "rb") as f:
            best_state = pickle.load(f)
    except Exception:
        return None
    return best_state.get("game_count")


def _resolve_explicit_candidate(args, checkpoints):
    candidate_path = args.checkpoint
    candidate_idx = _find_checkpoint_index(checkpoints, candidate_path)
    candidate_key = None

    if candidate_idx is not None:
        candidate_gc = checkpoints[candidate_idx][0]
        print(f"Candidate: g{candidate_gc:05d}  ({candidate_path})")
        candidate_key = checkpoints[candidate_idx][1]
        return candidate_idx, candidate_path, candidate_key

    resolved_gc = _resolve_best_state_game_count(args.weights_dir)
    if resolved_gc is not None:
        for i, (gc, _) in enumerate(checkpoints):
            if abs(gc - resolved_gc) <= 10:
                candidate_idx = i
                break

        if candidate_idx is not None:
            print(f"Candidate: {candidate_path} (best @ g{resolved_gc:05d})")
            candidate_key = checkpoints[candidate_idx][1]
            return candidate_idx, candidate_path, candidate_key

        candidate_idx = len(checkpoints) - 1
        print(f"Candidate: {candidate_path} "
              f"(best @ g{resolved_gc:05d}, using latest for opponents)")
        candidate_key = os.path.normpath(candidate_path)
        return candidate_idx, candidate_path, candidate_key

    candidate_idx = len(checkpoints) - 1
    print(f"Candidate: {candidate_path} (using latest for opponent selection)")
    candidate_key = os.path.normpath(candidate_path)
    return candidate_idx, candidate_path, candidate_key


def _resolve_candidate(args, checkpoints):
    if args.checkpoint:
        candidate_idx, candidate_path, candidate_key = _resolve_explicit_candidate(
            args, checkpoints
        )
    else:
        candidate_idx = len(checkpoints) - 1
        candidate_path = checkpoints[candidate_idx][1]
        candidate_gc = checkpoints[candidate_idx][0]
        print(f"Candidate: g{candidate_gc:05d}  ({candidate_path})")
        candidate_key = checkpoints[candidate_idx][1]

    candidate_gc = checkpoints[candidate_idx][0]
    if candidate_key is None:
        candidate_key = checkpoints[candidate_idx][1]

    return {
        "idx": candidate_idx,
        "path": candidate_path,
        "gc": candidate_gc,
        "key": candidate_key,
    }


def _load_eval_openings(args):
    # Load or create fixed opening book (same for all matchups and runs)
    openings = load_or_create_openings(
        args.openings,
        n_plies=args.plies,
        seed=args.seed,
    )
    # Use requested number (may be fewer than saved if book is smaller)
    openings = openings[:args.openings]
    n_games = len(openings) * 2
    return openings, n_games


def _collect_calibration_rows(candidate_model, sim_levels, openings):
    cal_rows = []
    g2_by_pair = {}
    for i in range(len(sim_levels)):
        for j in range(i + 1, len(sim_levels)):
            low = sim_levels[i]
            high = sim_levels[j]
            label = f"{high} sims vs {low} sims"
            result = run_match_sequential(
                candidate_model,
                candidate_model,
                label,
                openings=openings,
                sims=high,
                opp_sims=low,
                batch_size=EVAL_BATCH_SIZE,
            )
            cal_rows.append((high, low, result))
            g2_by_pair[(low, high)] = result["glicko2_delta"]
            print()
    return cal_rows, g2_by_pair


def _print_calibration_summary(candidate_gc, sim_levels, cal_rows, g2_by_pair):
    print("=" * 60)
    print(f"Sims calibration summary for g{candidate_gc:05d}")
    print("=" * 60)
    for high, low, result in cal_rows:
        print(f"  {high:4d} vs {low:4d} sims  "
              f"{result['win_pct']:5.1f}%  Glicko-2 Δ{result['glicko2_delta']:+4.0f}")

    if len(sim_levels) < 3:
        return

    print("\nAdjacent tier gaps:")
    for low, high in zip(sim_levels[:-1], sim_levels[1:], strict=False):
        g2 = g2_by_pair.get((low, high))
        if g2 is None:
            continue
        print(f"  {low:4d} → {high:4d}: Glicko-2 Δ{g2:+.0f}")


def _run_sims_calibration(parser, args, candidate_model, candidate_gc, openings, n_games):
    try:
        sim_levels = _parse_sim_levels(args.sim_levels)
    except ValueError as e:
        parser.error(str(e))

    print("Mode: sims calibration (single checkpoint, deterministic)")
    print("Rating updates: disabled in calibration mode")
    print(f"Sims levels: {', '.join(str(v) for v in sim_levels)}")
    print(f"Openings: {len(openings)} positions × 2 colors = {n_games} games per matchup")
    print()

    cal_rows, g2_by_pair = _collect_calibration_rows(candidate_model, sim_levels, openings)
    _print_calibration_summary(candidate_gc, sim_levels, cal_rows, g2_by_pair)


def _print_eval_match_config(opponents, openings, n_games, sims, no_rating_update):
    print(f"Opponents: {', '.join(lbl for lbl, _ in opponents)}")
    print(f"Openings: {len(openings)} positions × 2 colors = {n_games} games")
    print(f"MCTS: {sims} sims, deterministic (no noise, temp=0)")
    print(f"Ratings: {'off' if no_rating_update else 'on'} "
          f"({GLICKO2_RATINGS_FILE})")
    print()


def _run_standard_eval_matches(
    candidate_model,
    opponents,
    openings,
    sims,
    no_rating_update,
    candidate_key,
):
    ratings_table = load_glicko2_ratings()
    ratings_changed = False

    all_results = []
    for label, opp_path in opponents:
        opp_model, opp_arch = _load_model_with_arch_fallback(opp_path)
        print(f"  Opponent net: {_format_arch(opp_arch)}  ({opp_path})")
        result = run_match_sequential(
            candidate_model,
            opp_model,
            label,
            openings=openings,
            sims=sims,
            batch_size=EVAL_BATCH_SIZE,
        )
        if not no_rating_update:
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
    return all_results, ratings_changed, ratings_table


def _save_ratings_if_changed(ratings_changed, ratings_table, candidate_key):
    if not ratings_changed:
        return
    save_glicko2_ratings(ratings_table)
    cand_entry = get_glicko2_entry(ratings_table, candidate_key, create=False)
    if cand_entry:
        print(f"Saved ratings → {GLICKO2_RATINGS_FILE}  "
              f"(candidate now {format_glicko2_entry(cand_entry)})")
        print()


def _print_eval_summary(candidate_gc, all_results):
    print("=" * 60)
    print(f"Eval summary for g{candidate_gc:05d}")
    print("=" * 60)
    for result in all_results:
        status = "✓ IMPROVING" if result["win_pct"] > 55 else \
                 "✗ REGRESSED" if result["win_pct"] < 45 else \
                 "~ STABLE"
        print(f"  vs {result['label']:20s}  {result['win_pct']:5.1f}%  "
              f"Glicko-2 Δ{result['glicko2_delta']:+4.0f}  "
              f"{status}")

    if all(r["win_pct"] > 55 for r in all_results):
        print("\n→ Clear improvement across all baselines.")
    elif any(r["win_pct"] < 45 for r in all_results):
        print("\n→ Possible regression detected. Investigate before continuing.")
    else:
        print("\n→ Stable / marginal improvement.")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    parser = _build_eval_parser()
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

    candidate = _resolve_candidate(args, checkpoints)
    openings, n_games = _load_eval_openings(args)

    _print_eval_gpu_status()
    candidate_model, candidate_arch = _load_model_with_arch_fallback(candidate["path"])
    print(f"Candidate net: {_format_arch(candidate_arch)}")

    if args.calibrate_sims:
        _run_sims_calibration(
            parser,
            args,
            candidate_model,
            candidate["gc"],
            openings,
            n_games,
        )
        return

    opponents = select_opponents(checkpoints, candidate["idx"])
    if not opponents:
        print("Not enough checkpoints for comparison yet.")
        return

    _print_eval_match_config(
        opponents,
        openings,
        n_games,
        args.sims,
        args.no_rating_update,
    )
    all_results, ratings_changed, ratings_table = _run_standard_eval_matches(
        candidate_model,
        opponents,
        openings,
        args.sims,
        args.no_rating_update,
        candidate["key"],
    )
    _save_ratings_if_changed(ratings_changed, ratings_table, candidate["key"])
    _print_eval_summary(candidate["gc"], all_results)


if __name__ == "__main__":
    main()
