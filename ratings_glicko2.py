#!/usr/bin/env python3
"""
Glicko-2 rating math and persistence helpers.

Shared by eval.py and train.py.
"""

import argparse
import math
import os
import pickle
import time
from datetime import datetime

GLICKO2_RATINGS_FILE = "weights/glicko2_ratings.pkl"

GLICKO2_RATING0 = 1500.0
GLICKO2_RD0 = 350.0
GLICKO2_VOL0 = 0.06
GLICKO2_TAU = 0.5
GLICKO2_EPSILON = 1e-6
GLICKO2_SCALE = 173.7178

# Defensive sanity bounds. Values outside these ranges are treated as
# corrupted persistence artifacts and reset to defaults.
GLICKO2_SANITY_RATING_ABS_MAX = 5000.0
GLICKO2_SANITY_RD_MIN = 1.0
GLICKO2_SANITY_RD_MAX = 1000.0
GLICKO2_SANITY_VOL_MIN = 1e-6
GLICKO2_SANITY_VOL_MAX = 1.0


def _normalize_rating_key(checkpoint_path):
    if not checkpoint_path:
        return None
    if isinstance(checkpoint_path, str) and checkpoint_path.startswith(("ino:", "path:")):
        return checkpoint_path
    path = os.path.normpath(str(checkpoint_path))
    try:
        st = os.stat(path)
    except OSError:
        return None
    return f"ino:{int(st.st_dev)}:{int(st.st_ino)}"


def _path_fallback_key(checkpoint_path):
    return f"path:{os.path.normpath(str(checkpoint_path))}"


def _new_glicko2_entry(checkpoint_path, rating_key):
    return {
        "checkpoint_path": os.path.normpath(str(checkpoint_path)),
        "rating_key": rating_key,
        "rating": float(GLICKO2_RATING0),
        "rd": float(GLICKO2_RD0),
        "vol": float(GLICKO2_VOL0),
        "games": 0,
        "periods": 0,
        "updated_unix": int(time.time()),
    }


def _sanitize_rating_rd(rating, rd):
    changed = False
    try:
        rating = float(rating)
    except Exception:
        rating = float(GLICKO2_RATING0)
        changed = True
    try:
        rd = float(rd)
    except Exception:
        rd = float(GLICKO2_RD0)
        changed = True

    if (not math.isfinite(rating)
            or abs(rating) > GLICKO2_SANITY_RATING_ABS_MAX):
        rating = float(GLICKO2_RATING0)
        changed = True
    if (not math.isfinite(rd)
            or rd < GLICKO2_SANITY_RD_MIN
            or rd > GLICKO2_SANITY_RD_MAX):
        rd = float(GLICKO2_RD0)
        changed = True
    return rating, rd, changed


def _sanitize_vol(vol):
    changed = False
    try:
        vol = float(vol)
    except Exception:
        vol = float(GLICKO2_VOL0)
        changed = True
    if (not math.isfinite(vol)
            or vol < GLICKO2_SANITY_VOL_MIN
            or vol > GLICKO2_SANITY_VOL_MAX):
        vol = float(GLICKO2_VOL0)
        changed = True
    return vol, changed


def _sanitize_glicko2_triplet(rating, rd, vol):
    rating, rd, changed_a = _sanitize_rating_rd(rating, rd)
    vol, changed_b = _sanitize_vol(vol)
    return rating, rd, vol, (changed_a or changed_b)


def _sanitize_glicko2_entry(entry):
    if not isinstance(entry, dict):
        return False
    rating, rd, vol, changed = _sanitize_glicko2_triplet(
        entry.get("rating", GLICKO2_RATING0),
        entry.get("rd", GLICKO2_RD0),
        entry.get("vol", GLICKO2_VOL0),
    )
    if changed:
        entry["rating"] = rating
        entry["rd"] = rd
        entry["vol"] = vol
    return changed


def _merge_glicko2_entries(left, right):
    """Merge duplicate entries that resolve to the same rating key."""
    l_periods = int(left.get("periods", 0))
    r_periods = int(right.get("periods", 0))
    if r_periods > l_periods:
        left, right = right, left
    merged = dict(left)
    merged["games"] = max(int(left.get("games", 0)), int(right.get("games", 0)))
    merged["periods"] = max(l_periods, r_periods)
    merged["updated_unix"] = max(
        int(left.get("updated_unix", 0)),
        int(right.get("updated_unix", 0)),
    )
    return merged


def _migrate_rating_table_keys(table):
    migrated = {}
    changed = False
    for old_key, entry in table.items():
        if not isinstance(entry, dict):
            changed = True
            continue

        checkpoint_path = entry.get("checkpoint_path")
        if checkpoint_path:
            checkpoint_path = os.path.normpath(str(checkpoint_path))
        else:
            checkpoint_path = None

        new_key = None
        if isinstance(old_key, str) and old_key.startswith("ino:"):
            new_key = old_key
        elif checkpoint_path:
            new_key = _normalize_rating_key(checkpoint_path)
        elif isinstance(old_key, str) and old_key.startswith("path:"):
            new_key = _normalize_rating_key(old_key[5:])
        elif isinstance(old_key, str):
            new_key = _normalize_rating_key(old_key)

        if new_key is None:
            if checkpoint_path:
                new_key = _path_fallback_key(checkpoint_path)
            elif isinstance(old_key, str):
                new_key = old_key if old_key.startswith("path:") else f"path:{old_key}"
            else:
                new_key = str(old_key)

        row = dict(entry)
        if checkpoint_path:
            row["checkpoint_path"] = checkpoint_path
        row["rating_key"] = new_key
        if _sanitize_glicko2_entry(row):
            changed = True

        if new_key in migrated:
            migrated[new_key] = _merge_glicko2_entries(migrated[new_key], row)
            changed = True
        else:
            migrated[new_key] = row
        if new_key != old_key:
            changed = True

    return migrated, changed


def load_glicko2_ratings(path=GLICKO2_RATINGS_FILE):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                table = pickle.load(f)
            if isinstance(table, dict):
                migrated, _ = _migrate_rating_table_keys(table)
                return migrated
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
    direct_key = (
        checkpoint_path
        if isinstance(checkpoint_path, str) and checkpoint_path.startswith(("ino:", "path:"))
        else None
    )
    key = _normalize_rating_key(checkpoint_path) if direct_key is None else direct_key
    entry = table.get(key)
    if entry is None and direct_key is None:
        path_norm = os.path.normpath(str(checkpoint_path))
        legacy_key = path_norm
        if legacy_key in table:
            entry = table[legacy_key]
            if key is not None:
                table.pop(legacy_key, None)
                table[key] = entry
            else:
                key = legacy_key
        elif _path_fallback_key(path_norm) in table:
            old_key = _path_fallback_key(path_norm)
            entry = table[old_key]
            if key is not None:
                table.pop(old_key, None)
                table[key] = entry
            else:
                key = old_key

    if entry is None and direct_key is None:
        base = os.path.basename(os.path.normpath(str(checkpoint_path)))
        matches = []
        for k, v in table.items():
            if not isinstance(v, dict):
                continue
            cp = v.get("checkpoint_path")
            if cp and os.path.basename(str(cp)) == base:
                matches.append(k)
        if len(matches) == 1:
            old_key = matches[0]
            entry = table[old_key]
            if key is not None and old_key != key:
                table[key] = table.pop(old_key)
                entry = table[key]

    if entry is None and create:
        if key is None:
            key = _path_fallback_key(checkpoint_path)
        entry = _new_glicko2_entry(checkpoint_path, key)
        table[key] = entry
    if entry is not None and direct_key is None:
        entry["checkpoint_path"] = os.path.normpath(str(checkpoint_path))
    if entry is not None and key is not None:
        entry["rating_key"] = key
    if entry is not None:
        _sanitize_glicko2_entry(entry)
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
    # Numerically stable evaluation for extreme x:
    # as x -> +inf, ex dominates and the fraction term -> -0.5.
    # as x -> -inf, ex -> 0 and the fraction term -> 0.
    if x > 50.0:
        frac = -0.5
    elif x < -50.0:
        frac = 0.0
    else:
        ex = math.exp(x)
        phi2 = phi * phi
        base = phi2 + v + ex
        frac = (ex * (delta * delta - phi2 - v - ex)) / (2.0 * base * base)
    return frac - ((x - a) / (tau * tau))


def glicko2_update_player(rating, rd, vol, matches,
                          tau=GLICKO2_TAU, epsilon=GLICKO2_EPSILON):
    """Update one player from a list of (opp_rating, opp_rd, score)."""
    rating, rd, vol, _ = _sanitize_glicko2_triplet(rating, rd, vol)
    if not matches:
        return float(rating), float(rd), float(vol)

    mu, phi = _glicko2_to_internal(rating, rd)
    a = math.log(vol * vol)

    v_inv = 0.0
    delta_sum = 0.0
    for opp_rating, opp_rd, score in matches:
        opp_rating, opp_rd, _ = _sanitize_rating_rd(opp_rating, opp_rd)
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


def _match_update_payload(cand_key, cand_old, cand_new, cand_entry,
                          opp_key, opp_old, opp_new, opp_entry):
    return {
        "candidate": {
            "key": cand_key,
            "old": cand_old,
            "new": cand_new,
            "delta": cand_new[0] - cand_old[0],
            "entry": cand_entry,
        },
        "opponent": {
            "key": opp_key,
            "old": opp_old,
            "new": opp_new,
            "delta": opp_new[0] - opp_old[0],
            "entry": opp_entry,
        },
    }


def apply_match_glicko2_update(table, candidate_path, opponent_path,
                               wins, losses, draws):
    """Apply one match period update for both sides using stored ratings."""
    if not candidate_path or not opponent_path:
        return None
    cand = get_glicko2_entry(table, candidate_path, create=True)
    opp = get_glicko2_entry(table, opponent_path, create=True)
    if cand is None or opp is None:
        return None
    _sanitize_glicko2_entry(cand)
    _sanitize_glicko2_entry(opp)
    cand_key = cand.get("rating_key") or _normalize_rating_key(candidate_path)
    opp_key = opp.get("rating_key") or _normalize_rating_key(opponent_path)
    if not cand_key:
        cand_key = _path_fallback_key(candidate_path)
        cand["rating_key"] = cand_key
        table[cand_key] = cand
    if not opp_key:
        opp_key = _path_fallback_key(opponent_path)
        opp["rating_key"] = opp_key
        table[opp_key] = opp
    if cand_key == opp_key:
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

    return _match_update_payload(
        cand_key, cand_old, cand_new, cand,
        opp_key, opp_old, opp_new, opp,
    )


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


def _rating_row_from_entry(key, entry, full_path=False):
    checkpoint_path = entry.get("checkpoint_path")
    if checkpoint_path:
        display_path = os.path.normpath(str(checkpoint_path))
    elif isinstance(key, str) and key.startswith("path:"):
        display_path = os.path.normpath(key[5:])
    else:
        display_path = str(key)

    rating = float(entry.get("rating", GLICKO2_RATING0))
    rd = float(entry.get("rd", GLICKO2_RD0))
    games = int(entry.get("games", 0))
    periods = int(entry.get("periods", 0))
    updated_unix = int(entry.get("updated_unix", 0))

    return {
        "key": str(key),
        "path": display_path,
        "name": display_path if full_path else os.path.basename(display_path),
        "rating": rating,
        "rd": rd,
        "ci95": 2.0 * rd,
        "games": games,
        "periods": periods,
        "updated_unix": updated_unix,
    }


def _sorted_rating_rows(table, sort_by="rating", full_path=False):
    rows = []
    for key, entry in table.items():
        if not isinstance(entry, dict):
            continue
        rows.append(_rating_row_from_entry(key, entry, full_path=full_path))

    if sort_by == "rd":
        rows.sort(key=lambda r: (
            r["rd"],
            -r["rating"],
            -r["games"],
            r["name"].lower(),
        ))
        return rows
    if sort_by == "games":
        rows.sort(key=lambda r: (
            -r["games"],
            -r["rating"],
            r["rd"],
            r["name"].lower(),
        ))
        return rows
    if sort_by == "updated":
        rows.sort(key=lambda r: (
            -r["updated_unix"],
            -r["rating"],
            r["rd"],
            r["name"].lower(),
        ))
        return rows

    rows.sort(key=lambda r: (
        -r["rating"],
        r["rd"],
        -r["games"],
        r["name"].lower(),
    ))
    return rows


def _format_updated(updated_unix):
    if updated_unix <= 0:
        return "-"
    return datetime.fromtimestamp(updated_unix).strftime("%Y-%m-%d %H:%M")


def _print_ratings_table(rows, limit=None, show_key=False):
    if not rows:
        print("No rating entries found.")
        return

    if limit is not None and limit > 0:
        shown = rows[:limit]
    else:
        shown = rows

    def _fmt_float(x):
        ax = abs(float(x))
        if ax >= 1e6 or (ax > 0 and ax < 1e-3):
            return f"{x:.3e}"
        return f"{x:.1f}"

    for row in shown:
        row["rating_s"] = _fmt_float(row["rating"])
        row["rd_s"] = _fmt_float(row["rd"])
        row["ci95_s"] = _fmt_float(row["ci95"])

    name_w = max(20, min(52, max(len(r["name"]) for r in shown)))
    rating_w = max(7, max(len(r["rating_s"]) for r in shown))
    rd_w = max(6, max(len(r["rd_s"]) for r in shown))
    ci95_w = max(6, max(len(r["ci95_s"]) for r in shown))
    header = (
        f"{'#':>3}  {'checkpoint':<{name_w}}  {'R':>{rating_w}}  {'RD':>{rd_w}}  "
        f"{'95%':>{ci95_w}}  {'games':>7}  {'periods':>7}  {'updated':<16}"
    )
    if show_key:
        header += "  key"
    print(header)
    print("-" * len(header))
    for idx, row in enumerate(shown, 1):
        line = (
            f"{idx:>3d}  {row['name']:<{name_w}}  "
            f"{row['rating_s']:>{rating_w}}  {row['rd_s']:>{rd_w}}  "
            f"{row['ci95_s']:>{ci95_w}}  "
            f"{row['games']:>7d}  {row['periods']:>7d}  "
            f"{_format_updated(row['updated_unix']):<16}"
        )
        if show_key:
            line += f"  {row['key']}"
        print(line)
    if len(shown) < len(rows):
        print(f"... showing {len(shown)} of {len(rows)} entries")


def _build_ratings_printer_parser():
    parser = argparse.ArgumentParser(
        description="Pretty-print a Glicko-2 ratings table."
    )
    parser.add_argument(
        "ratings_file",
        nargs="?",
        default=GLICKO2_RATINGS_FILE,
        help=f"Ratings pickle file (default: {GLICKO2_RATINGS_FILE})",
    )
    parser.add_argument(
        "--sort",
        choices=("rating", "rd", "games", "updated"),
        default="rating",
        help="Sort order (default: rating)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max rows to print (<=0 means all, default: 50)",
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=0,
        help="Hide entries with fewer games (default: 0)",
    )
    parser.add_argument(
        "--full-path",
        action="store_true",
        help="Show full checkpoint paths instead of basenames",
    )
    parser.add_argument(
        "--show-key",
        action="store_true",
        help="Show internal rating key column",
    )
    return parser


def _run_ratings_printer_cli():
    parser = _build_ratings_printer_parser()
    args = parser.parse_args()

    ratings_file = os.path.normpath(args.ratings_file)
    if not os.path.exists(ratings_file):
        print(f"Ratings file not found: {ratings_file}")
        return 1

    table = load_glicko2_ratings(ratings_file)
    rows = _sorted_rating_rows(table, sort_by=args.sort, full_path=args.full_path)
    min_games = max(0, int(args.min_games))
    if min_games > 0:
        rows = [r for r in rows if r["games"] >= min_games]

    print(f"Ratings file: {ratings_file}")
    print(f"Entries: {len(rows)} (filtered by min-games >= {min_games})")
    print()
    limit = int(args.limit)
    _print_ratings_table(rows, limit=limit if limit > 0 else None, show_key=args.show_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_ratings_printer_cli())
