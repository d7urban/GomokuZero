#!/usr/bin/env python3
"""
Gomoku tournament mode extracted from eval.py.

Usage:
    python eval_tournament.py --tournament-dir botb-weights
"""

import argparse
import glob
import os

from book_openings import (
    EVAL_OPENING_PLIES,
    EVAL_OPENING_SEED,
    load_or_create_openings,
)
from gomoku import create_model
from ratings_glicko2 import (
    GLICKO2_RATING0,
    GLICKO2_RD0,
    apply_match_glicko2_update,
    format_glicko2_entry,
    get_glicko2_entry,
    glicko2_pairwise_win_prob,
)

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
TOURNEY_H2H_SHARED_GOLD_MARGIN = 0.02        # practical tie band (<=52/48)
TOURNEY_H2H_SHARED_GOLD_MIN_GAMES = 120


def find_weight_files(weights_dir):
    """Find model weight files in a directory for tournament mode."""
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


def _cfg(args, key, default):
    return getattr(args, key, default)


def _build_config(args):
    return {
        "rr_sims": _cfg(args, "rr_sims", TOURNEY_RR_SIMS),
        "rr_openings": _cfg(args, "rr_openings", TOURNEY_RR_OPENINGS),
        "rr_max_rounds": _cfg(args, "rr_max_rounds", TOURNEY_RR_MAX_ROUNDS),
        "rr_contender_prob": _cfg(args, "rr_contender_prob", TOURNEY_RR_CONTENDER_PROB),
        "rr_min_games": _cfg(args, "rr_min_games", TOURNEY_RR_MIN_GAMES),
        "h2h_phases": _cfg(args, "h2h_phases", TOURNEY_H2H_PHASES),
        "h2h_openings_per_batch": _cfg(
            args, "h2h_openings_per_batch", TOURNEY_H2H_OPENINGS_PER_BATCH
        ),
        "h2h_confidence_prob": _cfg(
            args, "h2h_confidence_prob", TOURNEY_H2H_CONFIDENCE_PROB
        ),
        "h2h_min_games": _cfg(args, "h2h_min_games", TOURNEY_H2H_MIN_GAMES),
        "h2h_rating_period_games": _cfg(
            args, "h2h_rating_period_games", TOURNEY_H2H_RATING_PERIOD_GAMES
        ),
        "h2h_target_rd": _cfg(args, "h2h_target_rd", TOURNEY_H2H_TARGET_RD),
        "h2h_shared_gold_margin": _cfg(
            args, "shared_gold_margin", TOURNEY_H2H_SHARED_GOLD_MARGIN
        ),
        "h2h_shared_gold_min_games": _cfg(
            args, "shared_gold_min_games", TOURNEY_H2H_SHARED_GOLD_MIN_GAMES
        ),
    }


def _take_openings(openings, cursor, count):
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


def _run_round_robin_pairing(
    model_a,
    model_b,
    ratings_table,
    left_path,
    right_path,
    round_openings,
    rr_sims,
    round_idx,
    run_match_fn,
    eval_batch_size,
):
    left_name = os.path.basename(left_path)
    right_name = os.path.basename(right_path)
    label = f"{left_name} vs {right_name} [RR {round_idx}]"

    model_a.load_weights(left_path)
    model_b.load_weights(right_path)
    result = run_match_fn(
        model_a,
        model_b,
        label,
        openings=round_openings,
        sims=rr_sims,
        batch_size=eval_batch_size,
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
    model_a,
    model_b,
    weight_paths,
    openings,
    run_match_fn,
    eval_batch_size,
    config,
):
    ratings_table = {}
    for path in weight_paths:
        get_glicko2_entry(ratings_table, path, create=True)
    pairs = _round_robin_pairs(weight_paths)

    cursor = 0
    confident = False
    rounds_played = 0
    rr_sims = config["rr_sims"]
    rr_openings = config["rr_openings"]
    rr_max_rounds = config["rr_max_rounds"]
    rr_contender_prob = config["rr_contender_prob"]
    rr_min_games = config["rr_min_games"]

    for round_idx in range(rr_max_rounds):
        rounds_played = round_idx + 1
        round_openings, cursor = _take_openings(openings, cursor, rr_openings)
        if not round_openings:
            break

        print(f"\nRound-robin round {rounds_played}/{rr_max_rounds} "
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
                run_match_fn,
                eval_batch_size,
            )

        rows = _print_tourney_standings(ratings_table, weight_paths)
        confident = _top_two_contenders_confident(
            rows,
            confidence_prob=rr_contender_prob,
            min_games=rr_min_games,
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
        "shared_gold": False,
        "winner": None,
        "confidence": 0.5,
        "cursor": opening_cursor,
        "min_seen_games": 0,
        "max_rd": float("inf"),
        "pending_wins": 0,
        "pending_losses": 0,
        "pending_draws": 0,
        "pending_games": 0,
        "total_wins": 0,
        "total_losses": 0,
        "total_draws": 0,
        "total_games": 0,
        "total_score_left": 0.0,
        "score_left": 0.5,
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
    shared_gold_margin,
    shared_gold_min_games,
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
    state["total_games"] = (
        int(state["total_wins"])
        + int(state["total_losses"])
        + int(state["total_draws"])
    )
    state["total_score_left"] = (
        float(state["total_wins"])
        + 0.5 * float(state["total_draws"])
    )
    if state["total_games"] > 0:
        state["score_left"] = state["total_score_left"] / float(state["total_games"])
    else:
        state["score_left"] = 0.5

    print(f"    Final rating: {left_name} -> {format_glicko2_entry(left_entry)}")
    print(f"    Final rating: {right_name} -> {format_glicko2_entry(right_entry)}")
    print(f"    Current leader: {os.path.basename(state['winner'])}  "
          f"(confidence {100.0 * state['confidence']:.1f}%, "
          f"games {state['min_seen_games']}, max RD {state['max_rd']:.1f})")
    print(
        f"    Match score: {left_name} {100.0 * state['score_left']:.1f}% "
        f"({state['total_wins']}W {state['total_losses']}L {state['total_draws']}D)"
    )
    print()

    if (state["confidence"] >= confidence_prob and
            state["min_seen_games"] >= min_games and
            state["max_rd"] <= rd_target):
        state["confident"] = True
        state["shared_gold"] = False
        return

    if (state["total_games"] >= shared_gold_min_games and
            abs(state["score_left"] - 0.5) <= shared_gold_margin):
        state["shared_gold"] = True
        state["winner"] = None
        state["confident"] = True
        print(
            "    Shared gold: practical tie detected "
            f"(within ±{100.0 * shared_gold_margin:.1f}% around 50/50)."
        )
        print()


def _run_heads_up_batch(
    model_a,
    model_b,
    state,
    openings,
    openings_per_batch,
    left_name,
    right_name,
    sims,
    shared_gold_margin,
    shared_gold_min_games,
    run_match_fn,
    eval_batch_size,
):
    state["batch_idx"] += 1
    batch_openings, state["cursor"] = _take_openings(
        openings, state["cursor"], openings_per_batch
    )
    if not batch_openings:
        return False

    label = f"{left_name} vs {right_name} [Final {state['batch_idx']}, {sims} sims]"
    result = run_match_fn(
        model_a,
        model_b,
        label,
        openings=batch_openings,
        sims=sims,
        batch_size=eval_batch_size,
    )
    state["pending_wins"] += int(result["wins"])
    state["pending_losses"] += int(result["losses"])
    state["pending_draws"] += int(result["draws"])
    state["pending_games"] += int(result["wins"] + result["losses"] + result["draws"])
    state["total_wins"] += int(result["wins"])
    state["total_losses"] += int(result["losses"])
    state["total_draws"] += int(result["draws"])
    total_games = (
        int(state["total_wins"])
        + int(state["total_losses"])
        + int(state["total_draws"])
    )
    total_score_left = float(state["total_wins"]) + 0.5 * float(state["total_draws"])
    score_left = (total_score_left / float(total_games)) if total_games > 0 else 0.5
    state["total_games"] = total_games
    state["total_score_left"] = total_score_left
    state["score_left"] = score_left

    if (total_games >= shared_gold_min_games and
            abs(score_left - 0.5) <= shared_gold_margin):
        state["shared_gold"] = True
        state["winner"] = None
        state["confident"] = True
        print(
            "    Shared gold: practical tie detected early "
            f"after {total_games} games "
            f"(score {100.0 * score_left:.1f}% within "
            f"±{100.0 * shared_gold_margin:.1f}%)."
        )
        print()
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
    run_match_fn,
    eval_batch_size,
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
                match["shared_gold_margin"],
                match["shared_gold_min_games"],
                run_match_fn,
                eval_batch_size,
            )
            if not ran_batch:
                break
            if state["confident"]:
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
    config,
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
        "confidence_prob": config["h2h_confidence_prob"],
        "min_games": config["h2h_min_games"],
        "rd_target": config["h2h_target_rd"],
        "shared_gold_margin": config["h2h_shared_gold_margin"],
        "shared_gold_min_games": config["h2h_shared_gold_min_games"],
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
        match["shared_gold_margin"],
        match["shared_gold_min_games"],
    )


def _finalize_heads_up_entries(state, ratings_table, left_path, right_path):
    left_entry = get_glicko2_entry(ratings_table, left_path, create=False)
    right_entry = get_glicko2_entry(ratings_table, right_path, create=False)
    if state["winner"] is None and not state["shared_gold"]:
        state["winner"] = (
            left_path if left_entry["rating"] >= right_entry["rating"] else right_path
        )
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
        "shared_gold": state["shared_gold"],
        "confident": state["confident"],
        "confidence": state["confidence"],
        "score_left": state["score_left"],
        "total_games": state["total_games"],
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
    model_a,
    model_b,
    left_path,
    right_path,
    openings,
    opening_cursor,
    config,
    run_match_fn,
    eval_batch_size,
):
    ratings_table, state, match = _build_heads_up_context(
        model_a,
        model_b,
        left_path,
        right_path,
        opening_cursor,
        config,
    )
    _run_heads_up_phases(
        model_a,
        model_b,
        state,
        ratings_table,
        match,
        openings,
        config["h2h_phases"],
        config["h2h_openings_per_batch"],
        config["h2h_rating_period_games"],
        run_match_fn,
        eval_batch_size,
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


def _load_tournament_openings(args, weight_paths, config):
    rr_needed = 0
    if len(weight_paths) > 2:
        rr_needed = config["rr_max_rounds"] * config["rr_openings"]
    h2h_needed = config["h2h_openings_per_batch"] * sum(
        n_batches for _, n_batches in config["h2h_phases"]
    )
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


def _print_gpu_status_local():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU: {gpus[0].name}")
    else:
        print("No GPU detected — eval will be slow.")


def _resolve_tournament_contenders(
    model_a,
    model_b,
    weight_paths,
    openings,
    run_match_fn,
    eval_batch_size,
    config,
):
    if len(weight_paths) == 2:
        print("\nTwo-player tournament: skipping round robin.")
        return weight_paths, 0

    rr = _run_round_robin_to_find_contenders(
        model_a=model_a,
        model_b=model_b,
        weight_paths=weight_paths,
        openings=openings,
        run_match_fn=run_match_fn,
        eval_batch_size=eval_batch_size,
        config=config,
    )

    contenders = rr["contenders"]
    print("\nRound-robin result:")
    print(f"  Rounds played: {rr['rounds_played']}")
    print(f"  Confident top-2: {'yes' if rr['confident'] else 'no (using best-rated top-2)'}")
    print(f"  Contenders: {os.path.basename(contenders[0])} vs "
          f"{os.path.basename(contenders[1])}")
    return contenders, rr["opening_cursor"]


def _print_heads_up_config(config):
    print("\nHeads-up final configuration:")
    for sims, n_batches in config["h2h_phases"]:
        n_games = n_batches * config["h2h_openings_per_batch"] * 2
        print(f"  {sims} sims: up to {n_games} games")
    print(f"  Glicko period size: {config['h2h_rating_period_games']} games")
    print(f"  Confidence target: {100.0 * config['h2h_confidence_prob']:.1f}%")
    print(f"  Minimum games before crowning: {config['h2h_min_games']}")
    print(f"  RD target: <= {config['h2h_target_rd']:.1f}")
    print(f"  Shared-gold band: 50/50 ± {100.0 * config['h2h_shared_gold_margin']:.1f}% "
          f"(min {config['h2h_shared_gold_min_games']} games)")
    print()


def _print_tournament_final(contenders, final, config):
    left_name = os.path.basename(contenders[0])
    right_name = os.path.basename(contenders[1])
    print("=" * 60)
    print("Tournament final result")
    print("=" * 60)
    if final.get("shared_gold"):
        print(f"  Champions (shared gold): {left_name} and {right_name}")
        print(f"  Match score: {left_name} {100.0 * final['score_left']:.1f}% "
              f"over {final['total_games']} games")
    else:
        winner = final["winner"]
        runner_up = contenders[1] if winner == contenders[0] else contenders[0]
        print(f"  Champion: {os.path.basename(winner)}")
        print(f"  Runner-up: {os.path.basename(runner_up)}")
        print(f"  Confidence: {100.0 * final['confidence']:.1f}% "
              f"({'confident' if final['confident'] else 'not fully confident at cap'})")
    print(f"  Max RD: {final['max_rd']:.1f} "
          f"({'target met' if final['max_rd'] <= config['h2h_target_rd'] else 'above target'})")
    print(f"  Batches played: {final['batches_played']}")
    print(f"  {left_name}: {format_glicko2_entry(final['left_entry'])}")
    print(f"  {right_name}: {format_glicko2_entry(final['right_entry'])}")


def run_tournament(args, run_match_fn, eval_batch_size, print_gpu_status_fn=None):
    """Run best-of-best tournament over all weights in args.tournament_dir."""
    config = _build_config(args)
    discovered = find_weight_files(args.tournament_dir)
    if len(discovered) < 2:
        print(f"Need at least two weights in {args.tournament_dir}; found {len(discovered)}")
        return

    _print_tournament_candidates(args.tournament_dir, discovered)
    model_probe = create_model()
    weight_paths = _resolve_compatible_weight_paths(model_probe, discovered)
    if not weight_paths:
        return

    openings = _load_tournament_openings(args, weight_paths, config)
    if not openings:
        return

    if print_gpu_status_fn is None:
        _print_gpu_status_local()
    else:
        print_gpu_status_fn()

    model_a = model_probe
    model_b = create_model()
    contenders, opening_cursor = _resolve_tournament_contenders(
        model_a,
        model_b,
        weight_paths,
        openings,
        run_match_fn,
        eval_batch_size,
        config,
    )
    _print_heads_up_config(config)

    final = _run_heads_up_until_confident(
        model_a=model_a,
        model_b=model_b,
        left_path=contenders[0],
        right_path=contenders[1],
        openings=openings,
        opening_cursor=opening_cursor,
        config=config,
        run_match_fn=run_match_fn,
        eval_batch_size=eval_batch_size,
    )
    _print_tournament_final(contenders, final, config)


def _build_tournament_parser():
    parser = argparse.ArgumentParser(
        description="Run best-of-best tournament over all weight files in a directory."
    )
    parser.add_argument("--tournament-dir", type=str, required=True,
                        help="Directory with checkpoint weights")
    parser.add_argument("--seed", type=int, default=EVAL_OPENING_SEED,
                        help=f"Opening RNG seed (default: {EVAL_OPENING_SEED})")
    parser.add_argument("--plies", type=int, default=EVAL_OPENING_PLIES,
                        help=f"Random plies per opening (default: {EVAL_OPENING_PLIES})")
    parser.add_argument("--shared-gold-margin", type=float,
                        default=TOURNEY_H2H_SHARED_GOLD_MARGIN,
                        help=f"Practical tie band around 50/50 "
                             f"(default: {TOURNEY_H2H_SHARED_GOLD_MARGIN:.2f})")
    parser.add_argument("--shared-gold-min-games", type=int,
                        default=TOURNEY_H2H_SHARED_GOLD_MIN_GAMES,
                        help=f"Minimum games before shared-gold tie "
                             f"(default: {TOURNEY_H2H_SHARED_GOLD_MIN_GAMES})")
    return parser


def main():
    parser = _build_tournament_parser()
    args = parser.parse_args()
    from eval import EVAL_BATCH_SIZE, run_match_sequential
    run_tournament(
        args=args,
        run_match_fn=run_match_sequential,
        eval_batch_size=EVAL_BATCH_SIZE,
        print_gpu_status_fn=None,
    )


if __name__ == "__main__":
    main()
