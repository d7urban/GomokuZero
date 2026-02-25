#!/usr/bin/env python3
"""
Gomoku tournament runner.

Pipeline:
1. Swiss seeding over all checkpoints (scales to large fields).
2. McMahon final stage over a bar-selected top group.
3. Champion (or shared gold) is promoted to best checkpoint outputs.

Usage:
    python eval_tournament.py --tournament-dir botb-weights
"""

import argparse
import glob
import os
import pickle
import re
import shutil

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
    load_glicko2_ratings,
    save_glicko2_ratings,
)

# Swiss seeding stage (all players)
TOURNEY_SWISS_SIMS = 50
TOURNEY_SWISS_OPENINGS = 4   # 8 games per pairing per round
TOURNEY_SWISS_ROUNDS = 6

# McMahon final stage (top bar-selected players)
TOURNEY_MCM_SIMS = 100
TOURNEY_MCM_OPENINGS = 8     # 16 games per pairing per round
TOURNEY_MCM_ROUNDS = 5
TOURNEY_MCM_BAR_GAP = 80.0   # include players within this rating gap from leader
TOURNEY_MCM_MIN_PLAYERS = 8
TOURNEY_MCM_MAX_PLAYERS = 24

# Practical tie policy for champion decision
TOURNEY_SHARED_GOLD_MARGIN = 0.02
TOURNEY_SHARED_GOLD_MIN_GAMES = 120


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


def _default_tournament_output_paths(tournament_dir):
    base_dir = os.path.normpath(str(tournament_dir)) if tournament_dir else "weights"
    return {
        "ratings_file": os.path.join(base_dir, "glicko2_ratings.pkl"),
        "best_weights_file": os.path.join(base_dir, "gomoku_best.weights.h5"),
        "best_state_file": os.path.join(base_dir, "best_checkpoint.pkl"),
    }


def _build_config(args):
    defaults = _default_tournament_output_paths(getattr(args, "tournament_dir", None))
    ratings_file = getattr(args, "ratings_file", None) or defaults["ratings_file"]
    best_weights_file = (
        getattr(args, "best_weights_file", None) or defaults["best_weights_file"]
    )
    best_state_file = (
        getattr(args, "best_state_file", None) or defaults["best_state_file"]
    )
    return {
        "swiss_sims": _cfg(args, "swiss_sims", TOURNEY_SWISS_SIMS),
        "swiss_openings": _cfg(args, "swiss_openings", TOURNEY_SWISS_OPENINGS),
        "swiss_rounds": _cfg(args, "swiss_rounds", TOURNEY_SWISS_ROUNDS),
        "mcmahon_sims": _cfg(args, "mcmahon_sims", TOURNEY_MCM_SIMS),
        "mcmahon_openings": _cfg(args, "mcmahon_openings", TOURNEY_MCM_OPENINGS),
        "mcmahon_rounds": _cfg(args, "mcmahon_rounds", TOURNEY_MCM_ROUNDS),
        "mcmahon_bar_gap": float(_cfg(args, "mcmahon_bar_gap", TOURNEY_MCM_BAR_GAP)),
        "mcmahon_min_players": int(
            _cfg(args, "mcmahon_min_players", TOURNEY_MCM_MIN_PLAYERS)
        ),
        "mcmahon_max_players": int(
            _cfg(args, "mcmahon_max_players", TOURNEY_MCM_MAX_PLAYERS)
        ),
        "shared_gold_margin": _cfg(args, "shared_gold_margin", TOURNEY_SHARED_GOLD_MARGIN),
        "shared_gold_min_games": _cfg(
            args, "shared_gold_min_games", TOURNEY_SHARED_GOLD_MIN_GAMES
        ),
        "persist_ratings": bool(_cfg(args, "persist_ratings", True)),
        "ratings_file": ratings_file,
        "promote_winner": bool(_cfg(args, "promote_winner", True)),
        "best_weights_file": best_weights_file,
        "best_state_file": best_state_file,
    }


def _load_ratings_table(config):
    if not config["persist_ratings"]:
        print("Ratings: transient (not writing to persistent glicko2 file)")
        print()
        return {}

    ratings_file = config["ratings_file"]
    table = load_glicko2_ratings(ratings_file)
    print(f"Ratings: persistent ({ratings_file})")
    print()
    return table


def _save_ratings_table_if_enabled(config, ratings_table):
    if not config["persist_ratings"]:
        return
    save_glicko2_ratings(ratings_table, config["ratings_file"])
    print(f"Saved ratings -> {config['ratings_file']}")


def _checkpoint_game_count(path):
    m = re.search(r"_g(\d{5})\.weights\.h5$", os.path.basename(path))
    if not m:
        return 0
    return int(m.group(1))


def _pick_promoted_winner(final):
    winner = final.get("winner")
    if winner:
        return winner, "McMahon champion"

    shared = final.get("shared_gold_paths") or []
    if shared:
        best = max(shared, key=lambda row: row["rating"])
        return best["path"], "shared-gold tie break by rating"

    rows = final.get("rows") or []
    if rows:
        return rows[0]["path"], "top McMahon score"
    return None, "no champion resolved"


def _write_best_checkpoint(config, promoted_path, ratings_table):
    best_weights_file = config["best_weights_file"]
    best_state_file = config["best_state_file"]
    weights_dir = os.path.dirname(best_weights_file)
    state_dir = os.path.dirname(best_state_file)
    if weights_dir:
        os.makedirs(weights_dir, exist_ok=True)
    if state_dir:
        os.makedirs(state_dir, exist_ok=True)

    shutil.copy2(promoted_path, best_weights_file)
    promoted_entry = get_glicko2_entry(ratings_table, promoted_path, create=False)
    best_state = {
        "path": promoted_path,
        "game_count": _checkpoint_game_count(promoted_path),
        "glicko2_vs_long": (
            promoted_entry.get("rating") if promoted_entry is not None else None
        ),
    }
    with open(best_state_file, "wb") as f:
        pickle.dump(best_state, f)
    return best_state


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


def _print_tourney_standings(ratings_table, weight_paths, limit=None):
    rows = _tourney_rows(ratings_table, weight_paths)
    shown = rows if limit is None else rows[: max(1, int(limit))]
    print("  Rating ladder:")
    for i, (path, entry) in enumerate(shown, 1):
        print(f"    {i:2d}. {os.path.basename(path):30s}  "
              f"{format_glicko2_entry(entry)}  "
              f"games {int(entry.get('games', 0))}")
    if limit is not None and len(rows) > len(shown):
        print(f"    ... +{len(rows) - len(shown)} more")
    return rows


def _new_stage_stats(paths):
    stats = {}
    for path in paths:
        stats[path] = {
            "score": 0.0,     # match points (W=1, D=0.5, L=0)
            "gp": 0.0,        # game points (wins + 0.5*draws)
            "games": 0,
            "rounds": 0,
            "byes": 0,
            "opponents": [],
        }
    return stats


def _stats_entry(stats, path):
    return stats[path]


def _stage_sort_key(path, stats, ratings_table):
    row = _stats_entry(stats, path)
    entry = get_glicko2_entry(ratings_table, path, create=True)
    score = float(row["score"])
    gp = float(row["gp"])
    games = max(1, int(row["games"]))
    gp_rate = gp / float(games)
    return (
        -score,
        -gp_rate,
        -float(entry.get("rating", GLICKO2_RATING0)),
        float(entry.get("rd", GLICKO2_RD0)),
        os.path.basename(path).lower(),
    )


def _ordered_players(paths, stats, ratings_table):
    return sorted(paths, key=lambda p: _stage_sort_key(p, stats, ratings_table))


def _select_bye_player(ordered_paths, stats):
    # Lowest seeded players get byes first; avoid multiple byes when possible.
    candidate = None
    candidate_key = None
    for path in reversed(ordered_paths):
        row = _stats_entry(stats, path)
        key = (
            int(row["byes"]),
            float(row["score"]),
            int(row["games"]),
            os.path.basename(path).lower(),
        )
        if candidate is None or key < candidate_key:
            candidate = path
            candidate_key = key
    return candidate


def _find_non_repeat_partner_index(left_path, candidates, played_pairs):
    for i, right_path in enumerate(candidates):
        pair_key = tuple(sorted((left_path, right_path)))
        if pair_key not in played_pairs:
            return i
    return None


def _pair_round_players(ordered_paths, stats, played_pairs):
    work = list(ordered_paths)
    bye_path = None
    if len(work) % 2 == 1:
        bye_path = _select_bye_player(work, stats)
        work.remove(bye_path)

    pairs = []
    while len(work) >= 2:
        left_path = work.pop(0)
        partner_idx = _find_non_repeat_partner_index(left_path, work, played_pairs)
        if partner_idx is None:
            partner_idx = 0
        right_path = work.pop(partner_idx)
        pairs.append((left_path, right_path))
    return pairs, bye_path


def _award_bye(stats, path):
    row = _stats_entry(stats, path)
    row["score"] += 1.0
    row["rounds"] += 1
    row["byes"] += 1


def _result_to_points(result):
    wins = int(result.get("wins", 0))
    losses = int(result.get("losses", 0))
    draws = int(result.get("draws", 0))
    total_games = wins + losses + draws
    left_gp = float(wins) + 0.5 * float(draws)
    right_gp = float(losses) + 0.5 * float(draws)

    if left_gp > right_gp:
        left_mp, right_mp = 1.0, 0.0
    elif right_gp > left_gp:
        left_mp, right_mp = 0.0, 1.0
    else:
        left_mp, right_mp = 0.5, 0.5

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "total_games": total_games,
        "left_gp": left_gp,
        "right_gp": right_gp,
        "left_mp": left_mp,
        "right_mp": right_mp,
    }


def _apply_stage_result(stats, left_path, right_path, result):
    scored = _result_to_points(result)
    left_row = _stats_entry(stats, left_path)
    right_row = _stats_entry(stats, right_path)

    left_row["score"] += scored["left_mp"]
    right_row["score"] += scored["right_mp"]
    left_row["gp"] += scored["left_gp"]
    right_row["gp"] += scored["right_gp"]
    left_row["games"] += scored["total_games"]
    right_row["games"] += scored["total_games"]
    left_row["rounds"] += 1
    right_row["rounds"] += 1
    left_row["opponents"].append(right_path)
    right_row["opponents"].append(left_path)

    return scored


def _stage_rows(stage_paths, stats, ratings_table):
    rows = []
    for path in stage_paths:
        st = _stats_entry(stats, path)
        entry = get_glicko2_entry(ratings_table, path, create=True)
        games = max(1, int(st["games"]))
        rows.append({
            "path": path,
            "score": float(st["score"]),
            "gp": float(st["gp"]),
            "gp_rate": float(st["gp"]) / float(games),
            "games": int(st["games"]),
            "rounds": int(st["rounds"]),
            "byes": int(st["byes"]),
            "rating": float(entry.get("rating", GLICKO2_RATING0)),
            "rd": float(entry.get("rd", GLICKO2_RD0)),
            "entry": entry,
        })
    rows.sort(
        key=lambda r: (
            -r["score"],
            -r["gp_rate"],
            -r["rating"],
            r["rd"],
            os.path.basename(r["path"]).lower(),
        )
    )
    return rows


def _print_stage_leaderboard(stage_name, stage_paths, stats, ratings_table, limit=12):
    rows = _stage_rows(stage_paths, stats, ratings_table)
    shown = rows[: max(1, int(limit))]
    print(f"\n{stage_name} leaderboard:")
    for i, row in enumerate(shown, 1):
        print(
            f"  {i:2d}. {os.path.basename(row['path']):30s}  "
            f"score {row['score']:.1f}  "
            f"gp {row['gp_rate']:.3f}  "
            f"{format_glicko2_entry(row['entry'])}"
        )
    if len(rows) > len(shown):
        print(f"  ... +{len(rows) - len(shown)} more")
    return rows


def _run_stage_pairing(
    model_a,
    model_b,
    stage_tag,
    round_idx,
    left_path,
    right_path,
    round_openings,
    sims,
    eval_batch_size,
    run_match_fn,
    ratings_table,
    stage_stats,
    played_pairs,
):
    left_name = os.path.basename(left_path)
    right_name = os.path.basename(right_path)
    label = f"{left_name} vs {right_name} [{stage_tag} {round_idx}]"

    model_a.load_weights(left_path)
    model_b.load_weights(right_path)
    result = run_match_fn(
        model_a,
        model_b,
        label,
        openings=round_openings,
        sims=sims,
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
    _apply_stage_result(stage_stats, left_path, right_path, result)
    played_pairs.add(tuple(sorted((left_path, right_path))))
    print()


def _run_pairing_stage(
    model_a,
    model_b,
    stage_name,
    stage_tag,
    stage_paths,
    openings,
    opening_cursor,
    rounds,
    sims,
    openings_per_round,
    run_match_fn,
    eval_batch_size,
    ratings_table,
    initial_stats=None,
    seed_played_pairs=None,
):
    stats = initial_stats if initial_stats is not None else _new_stage_stats(stage_paths)
    played_pairs = set(seed_played_pairs or [])
    cursor = int(opening_cursor)
    rounds_played = 0

    for round_idx in range(1, max(0, int(rounds)) + 1):
        round_openings, cursor = _take_openings(openings, cursor, openings_per_round)
        if not round_openings:
            break
        rounds_played = round_idx

        ordered = _ordered_players(stage_paths, stats, ratings_table)
        pairs, bye_path = _pair_round_players(ordered, stats, played_pairs)
        print(f"\n{stage_name} round {round_idx}/{rounds} "
              f"@ {sims} sims, {len(round_openings) * 2} games/pairing")
        print(f"  Pairings: {len(pairs)}")
        if bye_path:
            _award_bye(stats, bye_path)
            print(f"  Bye: {os.path.basename(bye_path)} (+1.0 match point)")

        for left_path, right_path in pairs:
            _run_stage_pairing(
                model_a=model_a,
                model_b=model_b,
                stage_tag=stage_tag,
                round_idx=round_idx,
                left_path=left_path,
                right_path=right_path,
                round_openings=round_openings,
                sims=sims,
                eval_batch_size=eval_batch_size,
                run_match_fn=run_match_fn,
                ratings_table=ratings_table,
                stage_stats=stats,
                played_pairs=played_pairs,
            )

        _print_stage_leaderboard(stage_name, stage_paths, stats, ratings_table)

    return {
        "stats": stats,
        "played_pairs": played_pairs,
        "rounds_played": rounds_played,
        "opening_cursor": cursor,
        "ratings_table": ratings_table,
    }


def _select_mcmahon_finalists(weight_paths, ratings_table, config):
    rows = _tourney_rows(ratings_table, weight_paths)
    total = len(rows)
    if total == 0:
        return [], None, rows

    bar_gap = float(config["mcmahon_bar_gap"])
    min_players = max(2, int(config["mcmahon_min_players"]))
    max_players = max(min_players, int(config["mcmahon_max_players"]))
    min_players = min(min_players, total)
    max_players = min(max_players, total)

    top_rating = float(rows[0][1].get("rating", GLICKO2_RATING0))
    finalists = [
        path for path, entry in rows
        if float(entry.get("rating", GLICKO2_RATING0)) >= (top_rating - bar_gap)
    ]
    if len(finalists) < min_players:
        finalists = [path for path, _ in rows[:min_players]]
    if len(finalists) > max_players:
        finalists = finalists[:max_players]

    bar_index = min(len(finalists), total) - 1
    bar_rating = float(rows[bar_index][1].get("rating", GLICKO2_RATING0))
    return finalists, bar_rating, rows


def _resolve_mcmahon_result(rows, ratings_table, config):
    if not rows:
        return {
            "winner": None,
            "shared_gold": False,
            "shared_gold_paths": [],
            "rows": [],
            "top_two_prob": None,
            "ratings_table": ratings_table,
        }

    winner = rows[0]["path"]
    shared_gold = False
    shared_gold_paths = []
    top_two_prob = None

    if len(rows) >= 2 and abs(rows[0]["score"] - rows[1]["score"]) < 1e-12:
        first = rows[0]
        second = rows[1]
        p_first = glicko2_pairwise_win_prob(first["entry"], second["entry"])
        top_two_prob = p_first
        min_games = min(
            int(first["entry"].get("games", 0)),
            int(second["entry"].get("games", 0)),
        )
        if (
            min_games >= int(config["shared_gold_min_games"])
            and abs(p_first - 0.5) <= float(config["shared_gold_margin"])
        ):
            shared_gold = True
            winner = None
            shared_gold_paths = [first, second]

    return {
        "winner": winner,
        "shared_gold": shared_gold,
        "shared_gold_paths": shared_gold_paths,
        "rows": rows,
        "top_two_prob": top_two_prob,
        "ratings_table": ratings_table,
    }


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


def _load_tournament_openings(args, config):
    needed_openings = max(
        1,
        int(config["swiss_rounds"]) * int(config["swiss_openings"])
        + int(config["mcmahon_rounds"]) * int(config["mcmahon_openings"]),
    )
    openings = load_or_create_openings(
        needed_openings,
        n_plies=args.plies,
        seed=args.seed,
    )
    if not openings:
        print("Failed to build opening book for tournament.")
        return None

    print(f"Tournament openings pool: {len(openings)} positions")
    print()
    return openings


def _print_gpu_status_local():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU: {gpus[0].name}")
    else:
        print("No GPU detected — eval will be slow.")


def _print_tournament_config(config):
    print("Tournament format:")
    print(f"  Swiss seeding:  {config['swiss_rounds']} rounds  "
          f"@ {config['swiss_sims']} sims, {config['swiss_openings'] * 2} games/pairing")
    print(f"  McMahon final:  {config['mcmahon_rounds']} rounds  "
          f"@ {config['mcmahon_sims']} sims, {config['mcmahon_openings'] * 2} games/pairing")
    print(f"  McMahon bar gap: {config['mcmahon_bar_gap']:.1f} rating points")
    print(f"  McMahon field:   min {config['mcmahon_min_players']} / "
          f"max {config['mcmahon_max_players']} players")
    print(f"  Shared gold:     50/50 ± {100.0 * config['shared_gold_margin']:.1f}% "
          f"(min {config['shared_gold_min_games']} games)")
    print()


def _print_finalists(finalists, bar_rating, ratings_table):
    print("\nMcMahon finalists:")
    print(f"  Bar rating: {bar_rating:.1f}")
    for i, path in enumerate(finalists, 1):
        entry = get_glicko2_entry(ratings_table, path, create=True)
        print(f"  {i:2d}. {os.path.basename(path):30s}  {format_glicko2_entry(entry)}")
    print()


def _print_tournament_final(final):
    rows = final["rows"]
    print("=" * 60)
    print("Tournament final result")
    print("=" * 60)
    if final["shared_gold"]:
        left = final["shared_gold_paths"][0]
        right = final["shared_gold_paths"][1]
        print(f"  Champions (shared gold): {os.path.basename(left['path'])} "
              f"and {os.path.basename(right['path'])}")
        if final["top_two_prob"] is not None:
            print(f"  Top-2 model win-prob split: "
                  f"{100.0 * final['top_two_prob']:.1f}% / "
                  f"{100.0 * (1.0 - final['top_two_prob']):.1f}%")
    else:
        winner = final["winner"]
        print(f"  Champion: {os.path.basename(winner)}")

    print("\n  Final McMahon standings:")
    shown = rows[: min(10, len(rows))]
    for i, row in enumerate(shown, 1):
        print(f"    {i:2d}. {os.path.basename(row['path']):30s}  "
              f"score {row['score']:.1f}  gp {row['gp_rate']:.3f}  "
              f"{format_glicko2_entry(row['entry'])}")
    if len(rows) > len(shown):
        print(f"    ... +{len(rows) - len(shown)} more")


def run_tournament(args, run_match_fn, eval_batch_size, print_gpu_status_fn=None):
    """Run Swiss + McMahon tournament over all weights in args.tournament_dir."""
    config = _build_config(args)
    discovered = find_weight_files(args.tournament_dir)
    if len(discovered) < 2:
        print(f"Need at least two weights in {args.tournament_dir}; found {len(discovered)}")
        return None

    _print_tournament_candidates(args.tournament_dir, discovered)
    model_probe = create_model()
    weight_paths = _resolve_compatible_weight_paths(model_probe, discovered)
    if not weight_paths:
        return None

    ratings_table = _load_ratings_table(config)
    openings = _load_tournament_openings(args, config)
    if not openings:
        return None

    _print_tournament_config(config)
    if print_gpu_status_fn is None:
        _print_gpu_status_local()
    else:
        print_gpu_status_fn()

    model_a = model_probe
    model_b = create_model()
    for path in weight_paths:
        get_glicko2_entry(ratings_table, path, create=True)

    swiss = _run_pairing_stage(
        model_a=model_a,
        model_b=model_b,
        stage_name="Swiss",
        stage_tag="Swiss",
        stage_paths=weight_paths,
        openings=openings,
        opening_cursor=0,
        rounds=config["swiss_rounds"],
        sims=config["swiss_sims"],
        openings_per_round=config["swiss_openings"],
        run_match_fn=run_match_fn,
        eval_batch_size=eval_batch_size,
        ratings_table=ratings_table,
    )
    ratings_table = swiss["ratings_table"]
    _print_tourney_standings(ratings_table, weight_paths, limit=15)

    finalists, bar_rating, _ = _select_mcmahon_finalists(weight_paths, ratings_table, config)
    if len(finalists) < 2:
        print("McMahon finalist selection produced fewer than two players.")
        return None
    _print_finalists(finalists, bar_rating, ratings_table)

    finalist_set = set(finalists)
    seeded_pairs = {
        pair for pair in swiss["played_pairs"]
        if pair[0] in finalist_set and pair[1] in finalist_set
    }
    mcmahon = _run_pairing_stage(
        model_a=model_a,
        model_b=model_b,
        stage_name="McMahon",
        stage_tag="McM",
        stage_paths=finalists,
        openings=openings,
        opening_cursor=swiss["opening_cursor"],
        rounds=config["mcmahon_rounds"],
        sims=config["mcmahon_sims"],
        openings_per_round=config["mcmahon_openings"],
        run_match_fn=run_match_fn,
        eval_batch_size=eval_batch_size,
        ratings_table=ratings_table,
        seed_played_pairs=seeded_pairs,
    )

    mc_rows = _stage_rows(finalists, mcmahon["stats"], ratings_table)
    final = _resolve_mcmahon_result(mc_rows, ratings_table, config)
    final["bar_rating"] = bar_rating
    final["finalists"] = finalists
    final["swiss_rounds_played"] = swiss["rounds_played"]
    final["mcmahon_rounds_played"] = mcmahon["rounds_played"]
    _print_tournament_final(final)
    _save_ratings_table_if_enabled(config, ratings_table)

    promoted_path = None
    best_state = None
    if config["promote_winner"]:
        promoted_path, reason = _pick_promoted_winner(final)
        if promoted_path:
            best_state = _write_best_checkpoint(config, promoted_path, ratings_table)
            print(f"Best checkpoint updated ({reason}) -> {promoted_path}")
            print(f"  Best weights: {config['best_weights_file']}")
            print(f"  Best state:   {config['best_state_file']}")
        else:
            print("Best checkpoint not updated (no champion resolved).")

    return {
        "final": final,
        "ratings_table": ratings_table,
        "promoted_path": promoted_path,
        "best_state": best_state,
    }


def _build_tournament_parser():
    parser = argparse.ArgumentParser(
        description="Run Swiss + McMahon tournament over all weight files in a directory."
    )
    parser.add_argument("--tournament-dir", type=str, required=True,
                        help="Directory with checkpoint weights")
    parser.add_argument("--seed", type=int, default=EVAL_OPENING_SEED,
                        help=f"Opening RNG seed (default: {EVAL_OPENING_SEED})")
    parser.add_argument("--plies", type=int, default=EVAL_OPENING_PLIES,
                        help=f"Random plies per opening (default: {EVAL_OPENING_PLIES})")

    parser.add_argument("--swiss-rounds", type=int, default=TOURNEY_SWISS_ROUNDS,
                        help=f"Swiss seeding rounds (default: {TOURNEY_SWISS_ROUNDS})")
    parser.add_argument("--swiss-sims", type=int, default=TOURNEY_SWISS_SIMS,
                        help=f"Swiss sims per move (default: {TOURNEY_SWISS_SIMS})")
    parser.add_argument("--swiss-openings", type=int, default=TOURNEY_SWISS_OPENINGS,
                        help=f"Swiss openings per round "
                             f"(default: {TOURNEY_SWISS_OPENINGS})")

    parser.add_argument("--mcmahon-rounds", type=int, default=TOURNEY_MCM_ROUNDS,
                        help=f"McMahon rounds (default: {TOURNEY_MCM_ROUNDS})")
    parser.add_argument("--mcmahon-sims", type=int, default=TOURNEY_MCM_SIMS,
                        help=f"McMahon sims per move (default: {TOURNEY_MCM_SIMS})")
    parser.add_argument("--mcmahon-openings", type=int, default=TOURNEY_MCM_OPENINGS,
                        help=f"McMahon openings per round "
                             f"(default: {TOURNEY_MCM_OPENINGS})")
    parser.add_argument("--mcmahon-bar-gap", type=float, default=TOURNEY_MCM_BAR_GAP,
                        help=f"Bar gap from leader rating "
                             f"(default: {TOURNEY_MCM_BAR_GAP:.1f})")
    parser.add_argument("--mcmahon-min-players", type=int, default=TOURNEY_MCM_MIN_PLAYERS,
                        help=f"Minimum McMahon finalists "
                             f"(default: {TOURNEY_MCM_MIN_PLAYERS})")
    parser.add_argument("--mcmahon-max-players", type=int, default=TOURNEY_MCM_MAX_PLAYERS,
                        help=f"Maximum McMahon finalists "
                             f"(default: {TOURNEY_MCM_MAX_PLAYERS})")

    parser.add_argument("--shared-gold-margin", type=float,
                        default=TOURNEY_SHARED_GOLD_MARGIN,
                        help=f"Practical tie band around 50/50 "
                             f"(default: {TOURNEY_SHARED_GOLD_MARGIN:.2f})")
    parser.add_argument("--shared-gold-min-games", type=int,
                        default=TOURNEY_SHARED_GOLD_MIN_GAMES,
                        help=f"Minimum games before shared-gold tie "
                             f"(default: {TOURNEY_SHARED_GOLD_MIN_GAMES})")

    parser.add_argument("--persist-ratings", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Load and save persistent Glicko-2 ratings "
                             "(default: enabled)")
    parser.add_argument("--ratings-file", type=str, default=None,
                        help="Persistent ratings file "
                             "(default: <tournament-dir>/glicko2_ratings.pkl)")
    parser.add_argument("--promote-winner", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Update best checkpoint files from tournament winner "
                             "(default: enabled)")
    parser.add_argument("--best-weights-file", type=str,
                        default=None,
                        help="Output path for best weights copy "
                             "(default: <tournament-dir>/gomoku_best.weights.h5)")
    parser.add_argument("--best-state-file", type=str,
                        default=None,
                        help="Output path for persisted best-state metadata "
                             "(default: <tournament-dir>/best_checkpoint.pkl)")
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
