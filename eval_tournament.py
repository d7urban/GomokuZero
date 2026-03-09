#!/usr/bin/env python3
"""
Gomoku tournament runner.

Pipeline:
1. Swiss seeding over all checkpoints (scales to large fields).
2. Final stage:
   - McMahon over a bar-selected top group, or
   - sequential SPRT ladder over Swiss-shortlisted challengers.
3. Champion is promoted to best checkpoint outputs.

Usage:
    python eval_tournament.py --tournament-dir botb-weights
"""

import argparse
import glob
import json
import math
import os
import pickle
import re
import shutil
from datetime import datetime, timezone

from book_openings import (
    EVAL_OPENING_PLIES,
    EVAL_OPENING_SEED,
    load_or_create_openings,
)
from entrypoint_shared import MODEL_ARCH_CANDIDATES
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
TOURNEY_SWISS_BATCH_SIZE = 16

# McMahon final stage (top bar-selected players)
TOURNEY_MCM_SIMS = 160
TOURNEY_MCM_OPENINGS = 8     # 16 games per pairing per round
TOURNEY_MCM_ROUNDS = 5
TOURNEY_MCM_BATCH_SIZE = 32
TOURNEY_MCM_BAR_GAP = 80.0   # include players within this rating gap from leader
TOURNEY_MCM_MIN_PLAYERS = 8
TOURNEY_MCM_MAX_PLAYERS = 24

# Practical tie policy for champion decision
TOURNEY_SHARED_GOLD_MARGIN = 0.02
TOURNEY_SHARED_GOLD_MIN_GAMES = 120
TOURNEY_SUMMARY_DIR = "tournament_summaries"

# Tournament mode
TOURNEY_MODE = "swiss-mcmahon"  # "swiss-mcmahon" or "swiss-sprt"
TOURNEY_CERTAINTY = 0.95

# SPRT challenger gate (hybrid mode: Swiss shortlist -> SPRT ladder)
TOURNEY_SPRT_SCORE0 = 0.50        # null hypothesis expected score
TOURNEY_SPRT_SCORE1 = 0.55        # alternative hypothesis expected score
TOURNEY_SPRT_MIN_GAMES = 32
TOURNEY_SPRT_MAX_GAMES = 320
TOURNEY_SPRT_OPENINGS_STEP = 4    # openings chunk per SPRT update (2x games)
TOURNEY_SPRT_MAX_CHALLENGERS = 12

# Keep canonical live pointers out of tournament pools.
TOURNEY_EXCLUDED_BASENAMES = {
    "gomoku_best.weights.h5",
    "gomoku_weights.weights.h5",
}
TOURNEY_EXCLUDED_BASENAME_PATTERNS = (
    re.compile(r"^gomoku_.*_final\.weights\.h5$"),
    re.compile(r"^gomoku_.*_interrupted\.weights\.h5$"),
)


def _is_excluded_tourney_file(path_or_name):
    base = os.path.basename(str(path_or_name)).lower()
    if base in TOURNEY_EXCLUDED_BASENAMES:
        return True
    return any(pat.fullmatch(base) for pat in TOURNEY_EXCLUDED_BASENAME_PATTERNS)


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
            if _is_excluded_tourney_file(key):
                continue
            if key in seen:
                continue
            seen.add(key)
            found.append(key)
    found.sort(key=lambda p: (os.path.basename(p).lower(), p.lower()))
    return found


def _create_model_for_arch(arch):
    num_res_blocks, num_filters = arch
    return create_model(
        num_res_blocks=num_res_blocks,
        num_filters=num_filters,
    )


def _format_arch(arch):
    return f"{arch[0]}x{arch[1]}"


def _resolve_weight_architectures(weight_paths):
    probe_models = {
        arch: _create_model_for_arch(arch)
        for arch in MODEL_ARCH_CANDIDATES
    }
    resolved = {}
    skipped = []

    for path in weight_paths:
        matched = False
        errors = []
        for arch in MODEL_ARCH_CANDIDATES:
            model = probe_models[arch]
            try:
                model.load_weights(path)
                resolved[path] = arch
                matched = True
                break
            except Exception as e:
                reason = str(e).splitlines()[0] if str(e) else type(e).__name__
                errors.append(f"{_format_arch(arch)}: {reason}")
        if matched:
            continue
        details = "; ".join(errors) if errors else "no loader attempts"
        skipped.append((path, details))

    return resolved, skipped


def _get_side_model(model_pool, arch):
    model = model_pool.get(arch)
    if model is None:
        model = _create_model_for_arch(arch)
        model_pool[arch] = model
    return model


def _resolve_compatible_weight_paths(discovered):
    resolved_arch, skipped = _resolve_weight_architectures(discovered)
    weight_paths = [path for path in discovered if path in resolved_arch]

    if skipped:
        print("Skipping incompatible files:")
        for path, reason in skipped:
            print(f"  - {os.path.basename(path)}: {reason}")
        print()

    if len(weight_paths) < 2:
        print("Need at least two compatible weight files after validation.")
        return None, None

    if len(weight_paths) != len(discovered):
        print(f"Using {len(weight_paths)} compatible weight files:")
        for i, p in enumerate(weight_paths, 1):
            print(f"  {i:2d}. {os.path.basename(p)}  [{_format_arch(resolved_arch[p])}]")
        print()
    else:
        print("Resolved architectures:")
        for i, p in enumerate(weight_paths, 1):
            print(f"  {i:2d}. {os.path.basename(p)}  [{_format_arch(resolved_arch[p])}]")
        print()

    return weight_paths, resolved_arch


def _validate_weight_files_for_model(model, weight_paths):
    # Backward-compatible shim: retained for external imports/tests.
    # Tournament execution now resolves per-file architectures.
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


def _build_config(args, eval_batch_size):
    defaults = _default_tournament_output_paths(getattr(args, "tournament_dir", None))
    ratings_file = getattr(args, "ratings_file", None) or defaults["ratings_file"]
    best_weights_file = (
        getattr(args, "best_weights_file", None) or defaults["best_weights_file"]
    )
    best_state_file = (
        getattr(args, "best_state_file", None) or defaults["best_state_file"]
    )
    swiss_batch_size = max(
        1, int(_cfg(args, "swiss_batch_size", TOURNEY_SWISS_BATCH_SIZE))
    )
    mcmahon_batch_size = max(
        1, int(_cfg(args, "mcmahon_batch_size", TOURNEY_MCM_BATCH_SIZE))
    )
    mode = str(_cfg(args, "mode", TOURNEY_MODE))
    certainty = float(_cfg(args, "certainty", TOURNEY_CERTAINTY))
    sprt_sims_raw = _cfg(args, "sprt_sims", None)
    sprt_batch_size_raw = _cfg(args, "sprt_batch_size", None)
    sprt_sims = int(
        _cfg(args, "mcmahon_sims", TOURNEY_MCM_SIMS)
        if sprt_sims_raw is None else sprt_sims_raw
    )
    sprt_batch_size = int(
        mcmahon_batch_size if sprt_batch_size_raw is None else sprt_batch_size_raw
    )

    # top-N group size overrides
    swiss_top_n_raw = _cfg(args, "swiss_top_n", None)
    swiss_top_n = max(2, int(swiss_top_n_raw)) if swiss_top_n_raw is not None else None

    mcmahon_min_players = int(_cfg(args, "mcmahon_min_players", TOURNEY_MCM_MIN_PLAYERS))
    mcmahon_max_players = int(_cfg(args, "mcmahon_max_players", TOURNEY_MCM_MAX_PLAYERS))
    mcmahon_top_n_raw = _cfg(args, "mcmahon_top_n", None)
    effective_mcmahon_top_n = (
        max(2, int(mcmahon_top_n_raw)) if mcmahon_top_n_raw is not None else swiss_top_n
    )
    if effective_mcmahon_top_n is not None:
        mcmahon_max_players = effective_mcmahon_top_n
        mcmahon_min_players = min(effective_mcmahon_top_n, mcmahon_min_players)

    sprt_top_n_raw = _cfg(args, "sprt_top_n", None)
    sprt_top_n = (
        max(2, int(sprt_top_n_raw)) if sprt_top_n_raw is not None else swiss_top_n
    )

    return {
        "mode": mode,
        "certainty": certainty,
        "swiss_sims": _cfg(args, "swiss_sims", TOURNEY_SWISS_SIMS),
        "swiss_openings": _cfg(args, "swiss_openings", TOURNEY_SWISS_OPENINGS),
        "swiss_rounds": _cfg(args, "swiss_rounds", TOURNEY_SWISS_ROUNDS),
        "swiss_batch_size": swiss_batch_size,
        "swiss_top_n": swiss_top_n,
        "mcmahon_sims": _cfg(args, "mcmahon_sims", TOURNEY_MCM_SIMS),
        "mcmahon_openings": _cfg(args, "mcmahon_openings", TOURNEY_MCM_OPENINGS),
        "mcmahon_rounds": _cfg(args, "mcmahon_rounds", TOURNEY_MCM_ROUNDS),
        "mcmahon_batch_size": mcmahon_batch_size,
        "mcmahon_bar_gap": float(_cfg(args, "mcmahon_bar_gap", TOURNEY_MCM_BAR_GAP)),
        "mcmahon_min_players": mcmahon_min_players,
        "mcmahon_max_players": mcmahon_max_players,
        "sprt_top_n": sprt_top_n,
        "sprt_score0": float(_cfg(args, "sprt_score0", TOURNEY_SPRT_SCORE0)),
        "sprt_score1": float(_cfg(args, "sprt_score1", TOURNEY_SPRT_SCORE1)),
        "sprt_min_games": max(2, int(_cfg(args, "sprt_min_games", TOURNEY_SPRT_MIN_GAMES))),
        "sprt_max_games": max(2, int(_cfg(args, "sprt_max_games", TOURNEY_SPRT_MAX_GAMES))),
        "sprt_openings_step": int(
            max(1, int(_cfg(args, "sprt_openings_step", TOURNEY_SPRT_OPENINGS_STEP)))
        ),
        "sprt_sims": max(1, int(sprt_sims)),
        "sprt_batch_size": max(1, int(sprt_batch_size)),
        "sprt_max_challengers": max(1, int(
            _cfg(args, "sprt_max_challengers", TOURNEY_SPRT_MAX_CHALLENGERS)
        )),
        "sprt_alpha": _cfg(args, "sprt_alpha", None),
        "sprt_beta": _cfg(args, "sprt_beta", None),
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
    m = re.search(r"_g(\d+)\.weights\.h5$", os.path.basename(path))
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
    model_pool_a,
    model_pool_b,
    arch_by_path,
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

    left_arch = arch_by_path[left_path]
    right_arch = arch_by_path[right_path]
    model_a = _get_side_model(model_pool_a, left_arch)
    model_b = _get_side_model(model_pool_b, right_arch)
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
    model_pool_a,
    model_pool_b,
    arch_by_path,
    stage_name,
    stage_tag,
    stage_paths,
    openings,
    opening_cursor,
    rounds,
    sims,
    openings_per_round,
    run_match_fn,
    stage_batch_size,
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
              f"@ {sims} sims, batch {stage_batch_size}, "
              f"{len(round_openings) * 2} games/pairing")
        print(f"  Pairings: {len(pairs)}")
        if bye_path:
            _award_bye(stats, bye_path)
            print(f"  Bye: {os.path.basename(bye_path)} (+1.0 match point)")

        for left_path, right_path in pairs:
            _run_stage_pairing(
                model_pool_a=model_pool_a,
                model_pool_b=model_pool_b,
                arch_by_path=arch_by_path,
                stage_tag=stage_tag,
                round_idx=round_idx,
                left_path=left_path,
                right_path=right_path,
                round_openings=round_openings,
                sims=sims,
                eval_batch_size=stage_batch_size,
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


def _clamp_prob(x, lo=1e-6, hi=1.0 - 1e-6):
    return min(max(float(x), float(lo)), float(hi))


def _resolve_sprt_settings(config, n_tests):
    n_tests = max(1, int(n_tests))
    certainty = _clamp_prob(config["certainty"], lo=0.50, hi=0.999999)
    per_test_error = _clamp_prob(
        1.0 - certainty ** (1.0 / n_tests),
        lo=1e-6,
        hi=0.49,
    )

    alpha = config.get("sprt_alpha", None)
    beta = config.get("sprt_beta", None)
    alpha = per_test_error if alpha is None else _clamp_prob(alpha, lo=1e-6, hi=0.49)
    beta = per_test_error if beta is None else _clamp_prob(beta, lo=1e-6, hi=0.49)

    score0 = _clamp_prob(config["sprt_score0"], lo=1e-4, hi=1.0 - 1e-4)
    score1 = _clamp_prob(config["sprt_score1"], lo=1e-4, hi=1.0 - 1e-4)
    if score1 <= score0:
        raise ValueError(
            f"SPRT requires score1 > score0; got {score1:.4f} <= {score0:.4f}"
        )

    min_games = max(2, int(config["sprt_min_games"]))
    max_games = max(min_games, int(config["sprt_max_games"]))
    openings_step = max(1, int(config["sprt_openings_step"]))

    llr_win = math.log(score1 / score0)
    llr_loss = math.log((1.0 - score1) / (1.0 - score0))
    upper = math.log((1.0 - beta) / alpha)
    lower = math.log(beta / (1.0 - alpha))

    return {
        "certainty": certainty,
        "n_tests": n_tests,
        "alpha": alpha,
        "beta": beta,
        "score0": score0,
        "score1": score1,
        "llr_win": llr_win,
        "llr_loss": llr_loss,
        "upper": upper,
        "lower": lower,
        "min_games": min_games,
        "max_games": max_games,
        "openings_step": openings_step,
        "sims": int(config["sprt_sims"]),
        "batch_size": int(config["sprt_batch_size"]),
    }


def _sprt_llr(wins, losses, draws, score0, score1):
    total = int(wins) + int(losses) + int(draws)
    if total <= 0:
        return 0.0
    score = float(wins) + 0.5 * float(draws)
    return (
        score * math.log(score1 / score0)
        + (float(total) - score) * math.log((1.0 - score1) / (1.0 - score0))
    )


def _run_sprt_duel(
    model_pool_a,
    model_pool_b,
    arch_by_path,
    challenger_path,
    incumbent_path,
    openings,
    opening_cursor,
    run_match_fn,
    ratings_table,
    sprt,
):
    challenger_name = os.path.basename(challenger_path)
    incumbent_name = os.path.basename(incumbent_path)
    print(f"\nSPRT: {challenger_name} vs {incumbent_name}")

    challenger_arch = arch_by_path[challenger_path]
    incumbent_arch = arch_by_path[incumbent_path]
    challenger_model = _get_side_model(model_pool_a, challenger_arch)
    incumbent_model = _get_side_model(model_pool_b, incumbent_arch)
    challenger_model.load_weights(challenger_path)
    incumbent_model.load_weights(incumbent_path)

    wins = 0
    losses = 0
    draws = 0
    cursor = int(opening_cursor)
    decision = "inconclusive"
    rounds = 0

    while (wins + losses + draws) < sprt["max_games"]:
        rounds += 1
        games_left = sprt["max_games"] - (wins + losses + draws)
        openings_now = max(1, min(sprt["openings_step"], int(math.ceil(games_left / 2.0))))
        duel_openings, cursor = _take_openings(openings, cursor, openings_now)
        if not duel_openings:
            break

        label = (
            f"SPRT {challenger_name} vs {incumbent_name} "
            f"[chunk {rounds}]"
        )
        result = run_match_fn(
            challenger_model,
            incumbent_model,
            label,
            openings=duel_openings,
            sims=sprt["sims"],
            batch_size=sprt["batch_size"],
        )

        w = int(result.get("wins", 0))
        l = int(result.get("losses", 0))
        d = int(result.get("draws", 0))
        wins += w
        losses += l
        draws += d

        apply_match_glicko2_update(
            ratings_table,
            challenger_path,
            incumbent_path,
            wins=w,
            losses=l,
            draws=d,
        )

        total = wins + losses + draws
        score = float(wins) + 0.5 * float(draws)
        score_rate = score / max(1.0, float(total))
        llr = _sprt_llr(
            wins,
            losses,
            draws,
            score0=sprt["score0"],
            score1=sprt["score1"],
        )
        print(
            f"  SPRT progress: n={total:3d}  score={score_rate:.3f}  "
            f"LLR={llr:+.3f}  [{sprt['lower']:+.3f}, {sprt['upper']:+.3f}]"
        )

        if total >= sprt["min_games"] and llr >= sprt["upper"]:
            decision = "accept_h1"
            break
        if total >= sprt["min_games"] and llr <= sprt["lower"]:
            decision = "accept_h0"
            break

        remaining = sprt["max_games"] - total
        if remaining > 0:
            llr_max_reachable = llr + float(remaining) * float(sprt["llr_win"])
            llr_min_reachable = llr + float(remaining) * float(sprt["llr_loss"])
            if llr_max_reachable < sprt["upper"] and llr_min_reachable > sprt["lower"]:
                decision = "inconclusive_futility"
                print("  SPRT decision: INCONCLUSIVE (bounds unreachable with remaining games)")
                break

    total = wins + losses + draws
    score = float(wins) + 0.5 * float(draws)
    score_rate = score / max(1.0, float(total))
    llr = _sprt_llr(
        wins,
        losses,
        draws,
        score0=sprt["score0"],
        score1=sprt["score1"],
    )
    if decision == "inconclusive":
        if total >= sprt["min_games"] and llr >= sprt["upper"]:
            decision = "accept_h1"
        elif total >= sprt["min_games"] and llr <= sprt["lower"]:
            decision = "accept_h0"

    if decision == "accept_h1":
        print("  SPRT decision: ACCEPT H1 (challenger is stronger)")
    elif decision == "accept_h0":
        print("  SPRT decision: ACCEPT H0 (challenger not stronger)")
    elif decision == "inconclusive_futility":
        print("  SPRT decision: INCONCLUSIVE (early futility stop)")
    else:
        print("  SPRT decision: INCONCLUSIVE (max games reached)")

    return {
        "challenger_path": challenger_path,
        "incumbent_path": incumbent_path,
        "wins": int(wins),
        "losses": int(losses),
        "draws": int(draws),
        "games": int(total),
        "score_rate": float(score_rate),
        "llr": float(llr),
        "decision": decision,
        "opening_cursor": int(cursor),
    }


def _run_swiss_sprt_stage(
    model_pool_a,
    model_pool_b,
    arch_by_path,
    weight_paths,
    openings,
    config,
    run_match_fn,
    ratings_table,
):
    swiss = _run_pairing_stage(
        model_pool_a=model_pool_a,
        model_pool_b=model_pool_b,
        arch_by_path=arch_by_path,
        stage_name="Swiss",
        stage_tag="Swiss",
        stage_paths=weight_paths,
        openings=openings,
        opening_cursor=0,
        rounds=config["swiss_rounds"],
        sims=config["swiss_sims"],
        openings_per_round=config["swiss_openings"],
        run_match_fn=run_match_fn,
        stage_batch_size=config["swiss_batch_size"],
        ratings_table=ratings_table,
    )
    ratings_table = swiss["ratings_table"]
    _print_tourney_standings(ratings_table, weight_paths, limit=15)

    finalists, bar_rating, _ = _select_mcmahon_finalists(
        weight_paths, ratings_table, config,
        max_players_override=config.get("sprt_top_n"),
    )
    if len(finalists) < 2:
        print("SPRT finalist selection produced fewer than two players.")
        return None, ratings_table
    _print_finalists(finalists, bar_rating, ratings_table)

    swiss_rows = _stage_rows(finalists, swiss["stats"], ratings_table)
    ordered = [row["path"] for row in swiss_rows]

    max_challengers = max(1, int(config["sprt_max_challengers"]))
    selected_count = min(len(ordered), 1 + max_challengers)
    ladder_paths = ordered[:selected_count]
    if len(ladder_paths) < 2:
        winner = ladder_paths[0]
        rows = _stage_rows(finalists, swiss["stats"], ratings_table)
        for i, row in enumerate(rows):
            if row["path"] == winner:
                rows.insert(0, rows.pop(i))
                break
        final = {
            "mode": "swiss-sprt",
            "winner": winner,
            "shared_gold": False,
            "shared_gold_paths": [],
            "rows": rows,
            "top_two_prob": None,
            "bar_rating": bar_rating,
            "finalists": finalists,
            "swiss_rounds_played": swiss["rounds_played"],
            "mcmahon_rounds_played": 0,
            "sprt": {
                "settings": None,
                "runs": [],
                "tested_challengers": 0,
                "selected_paths": ladder_paths,
            },
        }
        return final, ratings_table

    sprt = _resolve_sprt_settings(config, n_tests=len(ladder_paths) - 1)
    print("\nSPRT ladder:")
    print(f"  Certainty target: {100.0 * sprt['certainty']:.1f}%")
    print(f"  Hypotheses: H0 score={sprt['score0']:.3f}, H1 score={sprt['score1']:.3f}")
    print(f"  Error rates: alpha={sprt['alpha']:.4f}, beta={sprt['beta']:.4f}")
    print(f"  Bounds: [{sprt['lower']:+.3f}, {sprt['upper']:+.3f}]")
    print(f"  Budget per duel: min {sprt['min_games']} / max {sprt['max_games']} games "
          f"@ {sprt['sims']} sims, batch {sprt['batch_size']}")

    incumbent = ladder_paths[0]
    cursor = swiss["opening_cursor"]
    sprt_runs = []
    challengers = ladder_paths[1:]
    for idx, challenger in enumerate(challengers, 1):
        print(f"\nSPRT challenge {idx}/{len(challengers)}:")
        duel = _run_sprt_duel(
            model_pool_a=model_pool_a,
            model_pool_b=model_pool_b,
            arch_by_path=arch_by_path,
            challenger_path=challenger,
            incumbent_path=incumbent,
            openings=openings,
            opening_cursor=cursor,
            run_match_fn=run_match_fn,
            ratings_table=ratings_table,
            sprt=sprt,
        )
        cursor = duel["opening_cursor"]
        sprt_runs.append(duel)
        if duel["decision"] == "accept_h1":
            incumbent = challenger
            print(f"  New incumbent: {os.path.basename(incumbent)}")
        else:
            print(f"  Incumbent kept: {os.path.basename(incumbent)}")

    rows = _stage_rows(finalists, swiss["stats"], ratings_table)
    for i, row in enumerate(rows):
        if row["path"] == incumbent:
            rows.insert(0, rows.pop(i))
            break

    final = {
        "mode": "swiss-sprt",
        "winner": incumbent,
        "shared_gold": False,
        "shared_gold_paths": [],
        "rows": rows,
        "top_two_prob": None,
        "bar_rating": bar_rating,
        "finalists": finalists,
        "swiss_rounds_played": swiss["rounds_played"],
        "mcmahon_rounds_played": 0,
        "sprt": {
            "settings": sprt,
            "runs": sprt_runs,
            "tested_challengers": len(challengers),
            "selected_paths": ladder_paths,
        },
    }
    return final, ratings_table


def _select_mcmahon_finalists(weight_paths, ratings_table, config, max_players_override=None):
    rows = _tourney_rows(ratings_table, weight_paths)
    total = len(rows)
    if total == 0:
        return [], None, rows

    bar_gap = float(config["mcmahon_bar_gap"])
    min_players = max(2, int(config["mcmahon_min_players"]))
    if max_players_override is not None:
        max_players = max(min_players, int(max_players_override))
    else:
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


def _load_tournament_openings(args, config):
    if config.get("mode") == "swiss-sprt":
        sprt_duel_openings = int(math.ceil(int(config["sprt_max_games"]) / 2.0))
        needed_openings = max(
            1,
            int(config["swiss_rounds"]) * int(config["swiss_openings"])
            + int(config["sprt_max_challengers"]) * sprt_duel_openings,
        )
    else:
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
    mode = str(config.get("mode", TOURNEY_MODE))
    print("Tournament format:")
    print(f"  Mode: {mode}")
    swiss_top_n = config.get("swiss_top_n")
    swiss_field = f"top {swiss_top_n}" if swiss_top_n is not None else "all"
    print(f"  Swiss seeding:  {config['swiss_rounds']} rounds  "
          f"@ {config['swiss_sims']} sims, batch {config['swiss_batch_size']}, "
          f"{config['swiss_openings'] * 2} games/pairing  [{swiss_field} players]")
    if mode == "swiss-sprt":
        sprt_top_n = config.get("sprt_top_n")
        shortlist_str = f"top {sprt_top_n}" if sprt_top_n is not None else (
            f"bar gap {config['mcmahon_bar_gap']:.1f}, "
            f"min {config['mcmahon_min_players']}, "
            f"max {config['mcmahon_max_players']}"
        )
        print("  SPRT ladder:    challengers from Swiss shortlist")
        print(f"    Certainty target: {100.0 * float(config['certainty']):.1f}%")
        print(f"    Hypotheses: H0 score={float(config['sprt_score0']):.3f}, "
              f"H1 score={float(config['sprt_score1']):.3f}")
        print(f"    Per duel: min {int(config['sprt_min_games'])} / "
              f"max {int(config['sprt_max_games'])} games, "
              f"step {int(config['sprt_openings_step']) * 2} games, "
              f"@ {int(config['sprt_sims'])} sims, "
              f"batch {int(config['sprt_batch_size'])}")
        print(f"    Max challengers: {int(config['sprt_max_challengers'])}")
        print(f"  Swiss shortlist:  {shortlist_str}")
    else:
        mcmahon_top_n = config.get("mcmahon_max_players")
        print(f"  McMahon final:  {config['mcmahon_rounds']} rounds  "
              f"@ {config['mcmahon_sims']} sims, batch {config['mcmahon_batch_size']}, "
              f"{config['mcmahon_openings'] * 2} games/pairing")
        print(f"  McMahon bar gap: {config['mcmahon_bar_gap']:.1f} rating points")
        print(f"  McMahon field:   min {config['mcmahon_min_players']} / "
              f"max {mcmahon_top_n} players")
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
    mode = str(final.get("mode", "swiss-mcmahon"))
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

    if mode == "swiss-sprt":
        sprt = final.get("sprt") or {}
        runs = sprt.get("runs") or []
        print(f"  SPRT duels run: {len(runs)}")
        if runs:
            accepted = sum(1 for r in runs if r.get("decision") == "accept_h1")
            rejected = sum(1 for r in runs if r.get("decision") == "accept_h0")
            inconclusive = len(runs) - accepted - rejected
            print(f"  SPRT decisions: {accepted} challenger accepted, "
                  f"{rejected} rejected, {inconclusive} inconclusive")
        standings_label = "Final Swiss+SPRT standings"
    else:
        standings_label = "Final McMahon standings"

    print(f"\n  {standings_label}:")
    shown = rows[: min(10, len(rows))]
    for i, row in enumerate(shown, 1):
        print(f"    {i:2d}. {os.path.basename(row['path']):30s}  "
              f"score {row['score']:.1f}  gp {row['gp_rate']:.3f}  "
              f"{format_glicko2_entry(row['entry'])}")
    if len(rows) > len(shown):
        print(f"    ... +{len(rows) - len(shown)} more")


def _summarize_standings_rows(rows):
    out = []
    for i, row in enumerate(rows, 1):
        out.append({
            "rank": int(i),
            "path": row["path"],
            "file": os.path.basename(row["path"]),
            "score": float(row["score"]),
            "gp_rate": float(row["gp_rate"]),
            "games": int(row["games"]),
            "rounds": int(row["rounds"]),
            "rating": float(row["rating"]),
            "rd": float(row["rd"]),
        })
    return out


def _arch_histogram(weight_paths, arch_by_path):
    hist = {}
    for path in weight_paths:
        arch = arch_by_path.get(path)
        key = _format_arch(arch) if arch is not None else "unknown"
        hist[key] = int(hist.get(key, 0)) + 1
    return hist


def _format_summary_markdown(summary):
    mode = summary["config"].get("mode", TOURNEY_MODE)
    lines = [
        "# Tournament Summary",
        "",
        f"- UTC: {summary['timestamp_utc']}",
        f"- Tournament dir: {summary['tournament_dir']}",
        "",
        "## Format",
        f"- Mode: {mode}",
        (
            f"- Swiss: {summary['config']['swiss_rounds']} rounds, "
            f"{summary['config']['swiss_sims']} sims, "
            f"batch {summary['config']['swiss_batch_size']}, "
            f"{summary['config']['swiss_openings'] * 2} games/pairing"
        ),
    ]

    if mode == "swiss-sprt":
        lines.extend([
            (
                f"- SPRT: H0 score={summary['config']['sprt_score0']:.3f}, "
                f"H1 score={summary['config']['sprt_score1']:.3f}, "
                f"certainty {100.0 * summary['config']['certainty']:.1f}%"
            ),
            (
                f"- SPRT budget: min {summary['config']['sprt_min_games']} / "
                f"max {summary['config']['sprt_max_games']} games per duel, "
                f"{summary['config']['sprt_sims']} sims, "
                f"batch {summary['config']['sprt_batch_size']}"
            ),
            "",
        ])
    else:
        lines.extend([
            (
                f"- McMahon: {summary['config']['mcmahon_rounds']} rounds, "
                f"{summary['config']['mcmahon_sims']} sims, "
                f"batch {summary['config']['mcmahon_batch_size']}, "
                f"{summary['config']['mcmahon_openings'] * 2} games/pairing"
            ),
            "",
        ])

    lines.extend([
        "## Field",
        f"- Discovered files: {summary['field']['discovered_files']}",
        f"- Compatible files: {summary['field']['compatible_files']}",
        (
            "- Architectures: "
            + ", ".join(
                f"{k}={v}" for k, v in sorted(summary["field"]["architectures"].items())
            )
        ),
        "",
        "## Result",
        f"- Champion: {summary['result']['winner_file'] or '(shared gold/no single winner)'}",
        f"- Shared gold: {'yes' if summary['result']['shared_gold'] else 'no'}",
        f"- Promoted checkpoint: {summary['result']['promoted_file'] or '(none)'}",
        "",
        "## Final Standings",
        "| # | Checkpoint | Score | GP | Rating | RD |",
        "|---:|---|---:|---:|---:|---:|",
    ])

    for row in summary["standings"]:
        lines.append(
            f"| {row['rank']} | {row['file']} | {row['score']:.1f} | "
            f"{row['gp_rate']:.3f} | {row['rating']:.0f} | {row['rd']:.0f} |"
        )
    return "\n".join(lines) + "\n"


def _write_tournament_summary(
    args,
    config,
    final,
    discovered,
    weight_paths,
    arch_by_path,
    promoted_path,
):
    tournament_dir = os.path.normpath(str(args.tournament_dir))
    summary_dir = os.path.join(tournament_dir, TOURNEY_SUMMARY_DIR)
    os.makedirs(summary_dir, exist_ok=True)

    ts_utc = datetime.now(timezone.utc)
    ts_tag = ts_utc.strftime("%Y%m%d_%H%M%SZ")
    standings = _summarize_standings_rows(final.get("rows") or [])
    winner = final.get("winner")
    shared_gold_rows = final.get("shared_gold_paths") or []

    summary = {
        "timestamp_utc": ts_utc.isoformat().replace("+00:00", "Z"),
        "tournament_dir": tournament_dir,
        "config": {
            "mode": str(config["mode"]),
            "certainty": float(config["certainty"]),
            "swiss_rounds": int(config["swiss_rounds"]),
            "swiss_sims": int(config["swiss_sims"]),
            "swiss_openings": int(config["swiss_openings"]),
            "swiss_batch_size": int(config["swiss_batch_size"]),
            "mcmahon_rounds": int(config["mcmahon_rounds"]),
            "mcmahon_sims": int(config["mcmahon_sims"]),
            "mcmahon_openings": int(config["mcmahon_openings"]),
            "mcmahon_batch_size": int(config["mcmahon_batch_size"]),
            "mcmahon_bar_gap": float(config["mcmahon_bar_gap"]),
            "mcmahon_min_players": int(config["mcmahon_min_players"]),
            "mcmahon_max_players": int(config["mcmahon_max_players"]),
            "sprt_score0": float(config["sprt_score0"]),
            "sprt_score1": float(config["sprt_score1"]),
            "sprt_min_games": int(config["sprt_min_games"]),
            "sprt_max_games": int(config["sprt_max_games"]),
            "sprt_sims": int(config["sprt_sims"]),
            "sprt_batch_size": int(config["sprt_batch_size"]),
        },
        "field": {
            "discovered_files": int(len(discovered)),
            "compatible_files": int(len(weight_paths)),
            "architectures": _arch_histogram(weight_paths, arch_by_path),
        },
        "result": {
            "winner_path": winner,
            "winner_file": os.path.basename(winner) if winner else None,
            "shared_gold": bool(final.get("shared_gold", False)),
            "shared_gold_files": [
                os.path.basename(row["path"]) for row in shared_gold_rows
            ],
            "shared_gold_paths": [row["path"] for row in shared_gold_rows],
            "promoted_path": promoted_path,
            "promoted_file": (
                os.path.basename(promoted_path) if promoted_path else None
            ),
            "sprt": final.get("sprt", None),
        },
        "standings": standings,
    }

    json_name = f"tournament_{ts_tag}.json"
    md_name = f"tournament_{ts_tag}.md"
    json_path = os.path.join(summary_dir, json_name)
    md_path = os.path.join(summary_dir, md_name)
    latest_json = os.path.join(summary_dir, "latest.json")
    latest_md = os.path.join(summary_dir, "latest.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_format_summary_markdown(summary))
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(latest_md, "w", encoding="utf-8") as f:
        f.write(_format_summary_markdown(summary))

    print("Saved tournament summary:")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")

    return {
        "json": json_path,
        "markdown": md_path,
        "latest_json": latest_json,
        "latest_markdown": latest_md,
    }


def run_tournament(args, run_match_fn, eval_batch_size, print_gpu_status_fn=None):
    """Run tournament over all weights (Swiss+McMahon or Swiss+SPRT)."""
    config = _build_config(args, eval_batch_size)
    discovered = find_weight_files(args.tournament_dir)
    if len(discovered) < 2:
        print(f"Need at least two weights in {args.tournament_dir}; found {len(discovered)}")
        return None

    _print_tournament_candidates(args.tournament_dir, discovered)
    weight_paths, arch_by_path = _resolve_compatible_weight_paths(discovered)
    if not weight_paths or not arch_by_path:
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

    model_pool_a = {}
    model_pool_b = {}
    for path in weight_paths:
        get_glicko2_entry(ratings_table, path, create=True)

    swiss_top_n = config.get("swiss_top_n")
    if swiss_top_n is not None and len(weight_paths) > swiss_top_n:
        weight_paths = sorted(
            weight_paths,
            key=lambda p: -float(
                get_glicko2_entry(ratings_table, p, create=False).get("rating", GLICKO2_RATING0)
            ),
        )[:int(swiss_top_n)]
        arch_by_path = {p: arch_by_path[p] for p in weight_paths}
        print(f"Swiss field capped to top {swiss_top_n} by Glicko-2 rating.")
        print()

    mode = str(config.get("mode", TOURNEY_MODE))
    if mode == "swiss-sprt":
        final, ratings_table = _run_swiss_sprt_stage(
            model_pool_a=model_pool_a,
            model_pool_b=model_pool_b,
            arch_by_path=arch_by_path,
            weight_paths=weight_paths,
            openings=openings,
            config=config,
            run_match_fn=run_match_fn,
            ratings_table=ratings_table,
        )
        if final is None:
            return None
    else:
        swiss = _run_pairing_stage(
            model_pool_a=model_pool_a,
            model_pool_b=model_pool_b,
            arch_by_path=arch_by_path,
            stage_name="Swiss",
            stage_tag="Swiss",
            stage_paths=weight_paths,
            openings=openings,
            opening_cursor=0,
            rounds=config["swiss_rounds"],
            sims=config["swiss_sims"],
            openings_per_round=config["swiss_openings"],
            run_match_fn=run_match_fn,
            stage_batch_size=config["swiss_batch_size"],
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
            model_pool_a=model_pool_a,
            model_pool_b=model_pool_b,
            arch_by_path=arch_by_path,
            stage_name="McMahon",
            stage_tag="McM",
            stage_paths=finalists,
            openings=openings,
            opening_cursor=swiss["opening_cursor"],
            rounds=config["mcmahon_rounds"],
            sims=config["mcmahon_sims"],
            openings_per_round=config["mcmahon_openings"],
            run_match_fn=run_match_fn,
            stage_batch_size=config["mcmahon_batch_size"],
            ratings_table=ratings_table,
            seed_played_pairs=seeded_pairs,
        )

        mc_rows = _stage_rows(finalists, mcmahon["stats"], ratings_table)
        final = _resolve_mcmahon_result(mc_rows, ratings_table, config)
        final["mode"] = "swiss-mcmahon"
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

    summary_files = None
    try:
        summary_files = _write_tournament_summary(
            args=args,
            config=config,
            final=final,
            discovered=discovered,
            weight_paths=weight_paths,
            arch_by_path=arch_by_path,
            promoted_path=promoted_path,
        )
    except Exception as e:
        print(f"Warning: failed to write tournament summary: {e}")

    return {
        "final": final,
        "ratings_table": ratings_table,
        "promoted_path": promoted_path,
        "best_state": best_state,
        "summary_files": summary_files,
    }


def _build_tournament_parser():
    parser = argparse.ArgumentParser(
        description="Run Swiss+McMahon or Swiss+SPRT tournament over weight files."
    )
    parser.add_argument("--tournament-dir", type=str, required=True,
                        help="Directory with checkpoint weights")
    parser.add_argument(
        "--mode",
        type=str,
        choices=("swiss-mcmahon", "swiss-sprt"),
        default=TOURNEY_MODE,
        help="Tournament mode (default: swiss-mcmahon)",
    )
    parser.add_argument(
        "--certainty",
        type=float,
        default=TOURNEY_CERTAINTY,
        help="Target confidence for strongest-net selection in SPRT mode "
             f"(default: {TOURNEY_CERTAINTY:.2f})",
    )
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
    parser.add_argument("--swiss-batch-size", type=int, default=TOURNEY_SWISS_BATCH_SIZE,
                        help=f"Swiss MCTS batch size (default: {TOURNEY_SWISS_BATCH_SIZE})")
    parser.add_argument("--swiss-top-n", type=int, default=None,
                        help="Limit Swiss field to top N players by Glicko-2 rating "
                             "(default: all)")

    parser.add_argument("--mcmahon-rounds", type=int, default=TOURNEY_MCM_ROUNDS,
                        help=f"McMahon rounds (default: {TOURNEY_MCM_ROUNDS})")
    parser.add_argument("--mcmahon-sims", type=int, default=TOURNEY_MCM_SIMS,
                        help=f"McMahon sims per move (default: {TOURNEY_MCM_SIMS})")
    parser.add_argument("--mcmahon-openings", type=int, default=TOURNEY_MCM_OPENINGS,
                        help=f"McMahon openings per round "
                             f"(default: {TOURNEY_MCM_OPENINGS})")
    parser.add_argument("--mcmahon-batch-size", type=int, default=TOURNEY_MCM_BATCH_SIZE,
                        help=f"McMahon MCTS batch size (default: {TOURNEY_MCM_BATCH_SIZE})")
    parser.add_argument("--mcmahon-bar-gap", type=float, default=TOURNEY_MCM_BAR_GAP,
                        help=f"Bar gap from leader rating "
                             f"(default: {TOURNEY_MCM_BAR_GAP:.1f})")
    parser.add_argument("--mcmahon-min-players", type=int, default=TOURNEY_MCM_MIN_PLAYERS,
                        help=f"Minimum McMahon finalists "
                             f"(default: {TOURNEY_MCM_MIN_PLAYERS})")
    parser.add_argument("--mcmahon-max-players", type=int, default=TOURNEY_MCM_MAX_PLAYERS,
                        help=f"Maximum McMahon finalists "
                             f"(default: {TOURNEY_MCM_MAX_PLAYERS})")
    parser.add_argument("--mcmahon-top-n", type=int, default=None,
                        help="Cap McMahon finalists to top N "
                             "(overrides --mcmahon-max-players; default: all)")
    parser.add_argument("--sprt-score0", type=float, default=TOURNEY_SPRT_SCORE0,
                        help=f"SPRT H0 expected score (default: {TOURNEY_SPRT_SCORE0:.2f})")
    parser.add_argument("--sprt-score1", type=float, default=TOURNEY_SPRT_SCORE1,
                        help=f"SPRT H1 expected score (default: {TOURNEY_SPRT_SCORE1:.2f})")
    parser.add_argument("--sprt-min-games", type=int, default=TOURNEY_SPRT_MIN_GAMES,
                        help=f"SPRT minimum games per duel (default: {TOURNEY_SPRT_MIN_GAMES})")
    parser.add_argument("--sprt-max-games", type=int, default=TOURNEY_SPRT_MAX_GAMES,
                        help=f"SPRT maximum games per duel (default: {TOURNEY_SPRT_MAX_GAMES})")
    parser.add_argument(
        "--sprt-openings-step",
        type=int,
        default=TOURNEY_SPRT_OPENINGS_STEP,
        help=f"SPRT openings chunk per update (default: {TOURNEY_SPRT_OPENINGS_STEP})",
    )
    parser.add_argument("--sprt-sims", type=int, default=None,
                        help="SPRT sims per move (default: mcmahon-sims)")
    parser.add_argument("--sprt-batch-size", type=int, default=None,
                        help="SPRT MCTS batch size (default: mcmahon-batch-size)")
    parser.add_argument("--sprt-max-challengers", type=int,
                        default=TOURNEY_SPRT_MAX_CHALLENGERS,
                        help="Max Swiss challengers tested in SPRT ladder "
                             f"(default: {TOURNEY_SPRT_MAX_CHALLENGERS})")
    parser.add_argument("--sprt-top-n", type=int, default=None,
                        help="Cap SPRT finalist shortlist to top N players "
                             "(default: all; in swiss-sprt mode)")
    parser.add_argument("--sprt-alpha", type=float, default=None,
                        help="Override SPRT Type-I error (default: derived from --certainty)")
    parser.add_argument("--sprt-beta", type=float, default=None,
                        help="Override SPRT Type-II error (default: derived from --certainty)")

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
