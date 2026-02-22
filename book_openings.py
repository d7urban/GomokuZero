#!/usr/bin/env python3
"""
Gomoku opening book for training.

Contains plausible 4-6 move openings derived from:
  - crazy-sensei.com 15×15 opening book (root-level values)
  - Standard freestyle Gomoku opening theory
  - Symmetric variations via D4 board symmetries

Each opening is a list of (row, col) tuples, 0-indexed, on a 15×15 board.
Move 1 is Black (PLAYER1), move 2 is White, etc.

Usage:
    from book_openings import get_book_openings
    openings = get_book_openings()  # list of [(r,c), (r,c), ...]
"""

import numpy as np

# Center = (7, 7).  Good first moves per crazy-sensei (value > 0.8):
# H8=(7,7), G7=(8,6), H7=(7,6), G6=(8,5), H6=(7,5),
# F6=(9,5), G5=(8,4), F5=(9,4), H5=(7,4), E5=(10,4)

# ── Hand-crafted 4-6 ply openings ─────────────────────────────────────────
# Format: Black, White, Black, White [, Black, White]
# Selected for: near-center, both sides have reasonable development,
# White is not immediately lost.

_BASE_OPENINGS = [
    # === Direct contact (adjacent stones) ===
    # Center horizontal development
    [(7,7), (7,8), (7,6), (8,7)],
    [(7,7), (7,8), (8,7), (6,8)],
    [(7,7), (7,6), (8,8), (6,6)],
    [(7,7), (8,7), (7,8), (8,8)],
    [(7,7), (8,8), (6,6), (8,6)],

    # Diagonal development
    [(7,7), (8,8), (6,6), (9,9)],
    [(7,7), (8,8), (7,6), (7,9)],
    [(7,7), (6,6), (8,8), (5,5)],
    [(7,7), (6,8), (8,6), (5,9)],
    [(7,7), (8,6), (6,8), (9,5)],

    # L-shape patterns
    [(7,7), (8,7), (7,8), (6,6)],
    [(7,7), (7,8), (8,7), (9,8)],
    [(7,7), (6,7), (8,7), (6,8)],
    [(7,7), (7,6), (7,8), (8,6)],

    # === Indirect / gap openings ===
    # One-gap horizontal
    [(7,7), (7,9), (7,5), (8,7)],
    [(7,7), (7,5), (7,9), (6,7)],
    [(7,7), (7,9), (8,8), (6,6)],

    # One-gap diagonal
    [(7,7), (5,5), (9,9), (6,6)],
    [(7,7), (9,9), (5,5), (8,8)],
    [(7,7), (9,5), (5,9), (8,6)],
    [(7,7), (5,9), (9,5), (6,8)],

    # Knight-move responses
    [(7,7), (9,8), (7,8), (5,6)],
    [(7,7), (5,6), (8,7), (9,8)],
    [(7,7), (8,5), (6,9), (7,8)],
    [(7,7), (6,9), (8,5), (7,6)],

    # === Off-center starts (per crazy-sensei top moves) ===
    # G7=(8,6) start
    [(8,6), (7,7), (8,7), (8,8)],
    [(8,6), (7,7), (9,7), (6,5)],
    [(8,6), (8,7), (7,6), (6,7)],

    # G6=(8,5) start
    [(8,5), (7,7), (8,6), (8,7)],
    [(8,5), (7,6), (9,6), (6,5)],
    [(8,5), (7,6), (8,6), (7,7)],

    # H7=(7,6) start
    [(7,6), (7,7), (8,6), (6,7)],
    [(7,6), (8,7), (6,5), (7,7)],
    [(7,6), (8,6), (7,7), (6,7)],

    # F5=(9,4) start (slightly off-center)
    [(9,4), (8,5), (8,4), (7,5)],
    [(9,4), (8,5), (9,5), (7,4)],

    # === Deeper 6-ply openings ===
    [(7,7), (7,8), (7,6), (8,7), (6,7), (8,8)],
    [(7,7), (8,8), (6,6), (8,6), (6,8), (7,8)],
    [(7,7), (7,8), (8,7), (6,8), (8,8), (6,6)],
    [(7,7), (8,7), (7,8), (8,8), (6,7), (6,8)],
    [(7,7), (6,6), (8,8), (5,5), (9,9), (7,8)],
    [(7,7), (8,8), (7,6), (7,9), (8,7), (6,8)],
    [(7,7), (7,6), (8,8), (6,6), (7,8), (6,7)],
    [(7,7), (8,6), (6,8), (9,5), (7,8), (8,7)],

    # === Joseki-like patterns (standard exchanges) ===
    # Tiger's mouth
    [(7,7), (8,7), (6,8), (8,8), (7,9), (6,6)],
    # Cross pattern
    [(7,7), (6,7), (8,7), (7,6), (7,8), (6,8)],
    # Arrow pattern
    [(7,7), (7,8), (6,7), (5,7), (8,7), (7,6)],
]


def _apply_symmetry(opening, k, board_size=15):
    """Apply D4 symmetry transform k ∈ [0,7] to an opening sequence."""
    center = (board_size - 1) / 2.0
    result = []
    rot = k % 4
    flip = k >= 4

    for r, c in opening:
        # Translate to center-relative
        dr, dc = r - center, c - center
        if flip:
            dc = -dc
        for _ in range(rot):
            dr, dc = -dc, dr
        result.append((int(round(dr + center)), int(round(dc + center))))
    return result


def _is_valid(opening, board_size=15):
    """Check all moves are in bounds and no duplicates."""
    seen = set()
    for r, c in opening:
        if r < 0 or r >= board_size or c < 0 or c >= board_size:
            return False
        if (r, c) in seen:
            return False
        seen.add((r, c))
    return True


def get_book_openings(board_size=15, include_symmetries=True):
    """Return list of book openings, optionally expanded with D4 symmetries.

    With symmetries (default): ~350+ unique openings from ~45 base patterns.
    Without: ~45 base openings only.
    """
    openings = []
    seen = set()

    transforms = range(8) if include_symmetries else range(1)

    for base in _BASE_OPENINGS:
        for k in transforms:
            opening = _apply_symmetry(base, k, board_size)
            if not _is_valid(opening, board_size):
                continue
            key = tuple(tuple(m) for m in opening)
            if key not in seen:
                seen.add(key)
                openings.append(opening)

    return openings


if __name__ == "__main__":
    openings = get_book_openings()
    print(f"Total book openings: {len(openings)}")
    print(f"Base patterns: {len(_BASE_OPENINGS)}")
    print(f"Unique after D4 expansion: {len(openings)}")

    # Show some examples
    for i, op in enumerate(openings[:10]):
        coords = ", ".join(f"({r},{c})" for r, c in op)
        print(f"  {i+1}: [{coords}]")

    # Verify all valid
    bad = sum(1 for o in openings if not _is_valid(o))
    print(f"Invalid: {bad}")

    # Distribution of lengths
    from collections import Counter
    lens = Counter(len(o) for o in openings)
    print(f"Length distribution: {dict(sorted(lens.items()))}")
