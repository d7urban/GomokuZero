#!/usr/bin/env python3
"""
Glicko-2 rating math and persistence helpers.

Shared by eval.py and train.py.
"""

import math
import os
import pickle
import time

GLICKO2_RATINGS_FILE = "weights/glicko2_ratings.pkl"

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
