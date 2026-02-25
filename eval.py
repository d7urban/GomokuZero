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
from book_openings import (
    EVAL_OPENING_PLIES,
    EVAL_OPENING_SEED,
    load_or_create_openings,
)
from ratings_glicko2 import (
    GLICKO2_RATING0,
    GLICKO2_RD0,
    GLICKO2_RATINGS_FILE,
    apply_match_glicko2_update,
    format_glicko2_entry,
    get_glicko2_entry,
    glicko2_pairwise_win_prob,
    glicko2_rating_delta,
    load_glicko2_ratings,
    save_glicko2_ratings,
)

# ── Defaults ────────────────────────────────────────────────────────────────
EVAL_SIMS         = 100    # MCTS simulations per move (same for both sides)
EVAL_BATCH_SIZE   = 8      # MCTS batch size
EVAL_GAMES        = 200    # games per matchup (100 openings × 2 color swaps)

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


# ── Tournament mode ──────────────────────────────────────────────────────────
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


def _round_robin_pairs(weight_paths):
    pairs = []
    for i in range(len(weight_paths)):
        for j in range(i + 1, len(weight_paths)):
            pairs.append((weight_paths[i], weight_paths[j]))
    return pairs


def _run_round_robin_pairing(model_a, model_b, ratings_table, left_path, right_path,
                             round_openings, rr_sims, round_idx):
    left_name = os.path.basename(left_path)
    right_name = os.path.basename(right_path)
    label = f"{left_name} vs {right_name} [RR {round_idx}]"

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
    pairs = _round_robin_pairs(weight_paths)

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
            _run_round_robin_pairing(
                model_a,
                model_b,
                ratings_table,
                left_path,
                right_path,
                round_openings,
                rr_sims,
                rounds_played,
            )

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


def _new_heads_up_state(opening_cursor):
    return {
        "batch_idx": 0,
        "confident": False,
        "winner": None,
        "confidence": 0.5,
        "cursor": opening_cursor,
        "min_seen_games": 0,
        "max_rd": float("inf"),
        "pending_wins": 0,
        "pending_losses": 0,
        "pending_draws": 0,
        "pending_games": 0,
    }


def _flush_heads_up_pending(
    state,
    ratings_table,
    left_path,
    right_path,
    left_name,
    right_name,
    confidence_prob,
    min_games,
    rd_target,
):
    if state["pending_games"] <= 0:
        return

    apply_match_glicko2_update(
        ratings_table,
        left_path,
        right_path,
        wins=state["pending_wins"],
        losses=state["pending_losses"],
        draws=state["pending_draws"],
    )
    state["pending_wins"] = 0
    state["pending_losses"] = 0
    state["pending_draws"] = 0
    state["pending_games"] = 0

    left_entry = get_glicko2_entry(ratings_table, left_path, create=False)
    right_entry = get_glicko2_entry(ratings_table, right_path, create=False)
    p_left = glicko2_pairwise_win_prob(left_entry, right_entry)
    state["confidence"] = max(p_left, 1.0 - p_left)
    state["winner"] = left_path if p_left >= 0.5 else right_path
    state["min_seen_games"] = min(
        int(left_entry.get("games", 0)),
        int(right_entry.get("games", 0)),
    )
    state["max_rd"] = max(
        float(left_entry.get("rd", GLICKO2_RD0)),
        float(right_entry.get("rd", GLICKO2_RD0)),
    )

    print(f"    Final rating: {left_name} -> {format_glicko2_entry(left_entry)}")
    print(f"    Final rating: {right_name} -> {format_glicko2_entry(right_entry)}")
    print(f"    Current leader: {os.path.basename(state['winner'])}  "
          f"(confidence {100.0 * state['confidence']:.1f}%, "
          f"games {state['min_seen_games']}, max RD {state['max_rd']:.1f})")
    print()

    if (state["confidence"] >= confidence_prob and
            state["min_seen_games"] >= min_games and
            state["max_rd"] <= rd_target):
        state["confident"] = True


def _run_heads_up_batch(
    model_a,
    model_b,
    state,
    openings,
    openings_per_batch,
    left_name,
    right_name,
    sims,
):
    state["batch_idx"] += 1
    batch_openings, state["cursor"] = _take_openings(
        openings, state["cursor"], openings_per_batch
    )
    if not batch_openings:
        return False

    label = f"{left_name} vs {right_name} [Final {state['batch_idx']}, {sims} sims]"
    result = run_match_sequential(
        model_a,
        model_b,
        label,
        openings=batch_openings,
        sims=sims,
        batch_size=EVAL_BATCH_SIZE,
    )
    state["pending_wins"] += int(result["wins"])
    state["pending_losses"] += int(result["losses"])
    state["pending_draws"] += int(result["draws"])
    state["pending_games"] += int(result["wins"] + result["losses"] + result["draws"])
    return True


def _run_heads_up_phases(
    model_a,
    model_b,
    state,
    ratings_table,
    match,
    openings,
    phases,
    openings_per_batch,
    rating_period_games,
):
    for sims, n_batches in phases:
        for _ in range(n_batches):
            ran_batch = _run_heads_up_batch(
                model_a,
                model_b,
                state,
                openings,
                openings_per_batch,
                match["left_name"],
                match["right_name"],
                sims,
            )
            if not ran_batch:
                break

            if state["pending_games"] >= rating_period_games:
                _flush_heads_up_for_match(state, ratings_table, match)
                if state["confident"]:
                    break
        if state["confident"]:
            break


def _build_heads_up_context(
    model_a,
    model_b,
    left_path,
    right_path,
    opening_cursor,
    confidence_prob,
    min_games,
    rd_target,
):
    ratings_table = {}
    get_glicko2_entry(ratings_table, left_path, create=True)
    get_glicko2_entry(ratings_table, right_path, create=True)
    model_a.load_weights(left_path)
    model_b.load_weights(right_path)
    match = {
        "left_path": left_path,
        "right_path": right_path,
        "left_name": os.path.basename(left_path),
        "right_name": os.path.basename(right_path),
        "confidence_prob": confidence_prob,
        "min_games": min_games,
        "rd_target": rd_target,
    }
    return ratings_table, _new_heads_up_state(opening_cursor), match


def _flush_heads_up_for_match(state, ratings_table, match):
    _flush_heads_up_pending(
        state,
        ratings_table,
        match["left_path"],
        match["right_path"],
        match["left_name"],
        match["right_name"],
        match["confidence_prob"],
        match["min_games"],
        match["rd_target"],
    )


def _finalize_heads_up_entries(state, ratings_table, left_path, right_path):
    left_entry = get_glicko2_entry(ratings_table, left_path, create=False)
    right_entry = get_glicko2_entry(ratings_table, right_path, create=False)
    if state["winner"] is None:
        state["winner"] = left_path if left_entry["rating"] >= right_entry["rating"] else right_path
        p_left = glicko2_pairwise_win_prob(left_entry, right_entry)
        state["confidence"] = max(p_left, 1.0 - p_left)
    return left_entry, right_entry


def _build_heads_up_result(state, ratings_table, match):
    left_entry, right_entry = _finalize_heads_up_entries(
        state,
        ratings_table,
        match["left_path"],
        match["right_path"],
    )
    return {
        "winner": state["winner"],
        "confident": state["confident"],
        "confidence": state["confidence"],
        "left_entry": left_entry,
        "right_entry": right_entry,
        "batches_played": state["batch_idx"],
        "max_rd": max(
            float(left_entry.get("rd", GLICKO2_RD0)),
            float(right_entry.get("rd", GLICKO2_RD0)),
        ),
        "opening_cursor": state["cursor"],
        "ratings_table": ratings_table,
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
    ratings_table, state, match = _build_heads_up_context(
        model_a,
        model_b,
        left_path,
        right_path,
        opening_cursor,
        confidence_prob,
        min_games,
        rd_target,
    )
    _run_heads_up_phases(
        model_a,
        model_b,
        state,
        ratings_table,
        match,
        openings,
        phases,
        openings_per_batch,
        rating_period_games,
    )
    _flush_heads_up_for_match(state, ratings_table, match)
    return _build_heads_up_result(state, ratings_table, match)


def _print_tournament_candidates(tournament_dir, discovered):
    print(f"Tournament folder: {tournament_dir}")
    print(f"Discovered {len(discovered)} candidate files:")
    for i, p in enumerate(discovered, 1):
        print(f"  {i:2d}. {os.path.basename(p)}")
    print()


def _resolve_compatible_weight_paths(model_probe, discovered):
    weight_paths, skipped = _validate_weight_files_for_model(model_probe, discovered)
    if skipped:
        print("Skipping incompatible files:")
        for path, reason in skipped:
            print(f"  - {os.path.basename(path)}: {reason}")
        print()

    if len(weight_paths) < 2:
        print("Need at least two compatible weight files after validation.")
        return None

    if len(weight_paths) != len(discovered):
        print(f"Using {len(weight_paths)} compatible weight files:")
        for i, p in enumerate(weight_paths, 1):
            print(f"  {i:2d}. {os.path.basename(p)}")
        print()
    return weight_paths


def _load_tournament_openings(args, weight_paths):
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
        return None

    print(f"Tournament openings pool: {len(openings)} positions")
    print("Ratings: transient (not writing to persistent glicko2 file)")
    print()
    return openings


def _print_eval_gpu_status():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU: {gpus[0].name}")
    else:
        print("No GPU detected — eval will be slow.")


def _resolve_tournament_contenders(model_a, model_b, weight_paths, openings):
    if len(weight_paths) == 2:
        print("\nTwo-player tournament: skipping round robin.")
        return weight_paths, 0

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
    print("\nRound-robin result:")
    print(f"  Rounds played: {rr['rounds_played']}")
    print(f"  Confident top-2: {'yes' if rr['confident'] else 'no (using best-rated top-2)'}")
    print(f"  Contenders: {os.path.basename(contenders[0])} vs "
          f"{os.path.basename(contenders[1])}")
    return contenders, rr["opening_cursor"]


def _print_heads_up_config():
    print("\nHeads-up final configuration:")
    for sims, n_batches in TOURNEY_H2H_PHASES:
        n_games = n_batches * TOURNEY_H2H_OPENINGS_PER_BATCH * 2
        print(f"  {sims} sims: up to {n_games} games")
    print(f"  Glicko period size: {TOURNEY_H2H_RATING_PERIOD_GAMES} games")
    print(f"  Confidence target: {100.0 * TOURNEY_H2H_CONFIDENCE_PROB:.1f}%")
    print(f"  Minimum games before crowning: {TOURNEY_H2H_MIN_GAMES}")
    print(f"  RD target: <= {TOURNEY_H2H_TARGET_RD:.1f}")
    print()


def _print_tournament_final(contenders, final):
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


def run_tournament(args):
    """Best-of-best tournament mode over all weights in a folder."""
    discovered = find_weight_files(args.tournament_dir)
    if len(discovered) < 2:
        print(f"Need at least two weights in {args.tournament_dir}; found {len(discovered)}")
        return

    _print_tournament_candidates(args.tournament_dir, discovered)
    model_probe = create_model()
    weight_paths = _resolve_compatible_weight_paths(model_probe, discovered)
    if not weight_paths:
        return

    openings = _load_tournament_openings(args, weight_paths)
    if not openings:
        return

    _print_eval_gpu_status()
    model_a = model_probe
    model_b = create_model()
    contenders, opening_cursor = _resolve_tournament_contenders(
        model_a, model_b, weight_paths, openings
    )
    _print_heads_up_config()

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
    _print_tournament_final(contenders, final)


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
    for low, high in zip(sim_levels[:-1], sim_levels[1:]):
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
    # Reusable opponent model (swap weights per matchup)
    opp_model = create_model()
    ratings_table = load_glicko2_ratings()
    ratings_changed = False

    all_results = []
    for label, opp_path in opponents:
        opp_model.load_weights(opp_path)
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
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
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
    candidate_model = create_model()
    candidate_model.load_weights(candidate["path"])

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
