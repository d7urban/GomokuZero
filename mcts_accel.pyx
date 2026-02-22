# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Cython-accelerated MCTS hot-path functions.

Drop-in replacements for the pure-Python versions in gomoku.py.
Build:  python setup_accel.py build_ext --inplace
"""
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, exp, INFINITY

np.import_array()

# Capability flag consumed by gomoku.py to detect fixed select_child semantics.
SELECT_CHILD_PARENT_VIEW = 1

# ── _select_child (lazy expansion) ────────────────────────────────────────
def select_child(node, double c_puct):
    """Pick child with highest UCB score.  Creates child lazily on first visit.

    Iterates over node._moves / node._priors (compact arrays set at expansion).
    Only the winning child gets a MCTSNode allocated if it doesn't exist yet.
    """
    cdef double sqrt_n = sqrt(<double>node.visit_count) if node.visit_count > 0 else 0.0
    cdef double best_score = -INFINITY
    cdef double ucb, q, prior_val
    cdef int vc, best_idx, i, n_moves

    moves = node._moves
    cdef np.ndarray[np.float32_t, ndim=1] priors = node._priors
    children = node.children
    n_moves = len(moves)
    best_idx = -1
    best_child = None

    for i in range(n_moves):
        action = moves[i]
        child = children.get(action)
        if child is not None:
            vc = child.visit_count
            if vc > 0:
                # child.q_value is from child-player perspective; flip sign so
                # parent selects moves maximizing parent value.
                q = -(child.value_sum / <double>vc)
            else:
                q = 0.0
        else:
            vc = 0
            q = 0.0
        prior_val = <double>priors[i]
        ucb = q + c_puct * prior_val * sqrt_n / (1.0 + <double>vc)
        if ucb > best_score:
            best_score = ucb
            best_idx = i
            best_child = child

    action = moves[best_idx]
    if best_child is None:
        from gomoku import MCTSNode
        best_child = MCTSNode(prior=<float>priors[best_idx])
        children[action] = best_child

    return action, best_child


# ── get_candidate_moves ────────────────────────────────────────────────────
def get_candidate_moves(np.ndarray[np.int8_t, ndim=2] board,
                        int distance=2, int density_threshold=8):
    """Return candidate empty squares near occupied stones."""
    cdef int size = board.shape[0]
    cdef int r, c, nr, nc, r_lo, r_hi, c_lo, c_hi
    cdef int n_occupied = 0
    cdef np.int8_t EMPTY = 0

    # Count occupied squares
    for r in range(size):
        for c in range(size):
            if board[r, c] != EMPTY:
                n_occupied += 1

    # Sparse board: return all empties
    if n_occupied == 0 or n_occupied < density_threshold:
        result = []
        for r in range(size):
            for c in range(size):
                if board[r, c] == EMPTY:
                    result.append((r, c))
        return result

    # Dense board: only nearby empties
    # Use a boolean grid to avoid set() overhead
    cdef np.ndarray[np.uint8_t, ndim=2] nearby = np.zeros((size, size), dtype=np.uint8)

    for r in range(size):
        for c in range(size):
            if board[r, c] != EMPTY:
                r_lo = r - distance if r - distance > 0 else 0
                r_hi = r + distance + 1 if r + distance + 1 < size else size
                c_lo = c - distance if c - distance > 0 else 0
                c_hi = c + distance + 1 if c + distance + 1 < size else size
                for nr in range(r_lo, r_hi):
                    for nc in range(c_lo, c_hi):
                        if board[nr, nc] == EMPTY:
                            nearby[nr, nc] = 1

    result = []
    for r in range(size):
        for c in range(size):
            if nearby[r, c]:
                result.append((r, c))
    return result


# ── masked_softmax ─────────────────────────────────────────────────────────
def masked_softmax(np.ndarray[np.float32_t, ndim=1] logits,
                   list moves, int board_size):
    """Masked softmax over legal moves.  Returns full probability vector."""
    cdef int n = logits.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] out = np.full(n, -1e9, dtype=np.float64)
    cdef int r, c, idx
    cdef double max_val, s, val

    # Unmask legal moves
    for (r, c) in moves:
        idx = r * board_size + c
        out[idx] = <double>logits[idx]

    # Stable softmax
    max_val = -INFINITY
    for idx in range(n):
        if out[idx] > max_val:
            max_val = out[idx]

    s = 0.0
    for idx in range(n):
        val = out[idx] - max_val
        if val > -500.0:  # avoid underflow
            val = exp(val)
        else:
            val = 0.0
        out[idx] = val
        s += val

    cdef np.ndarray[np.float32_t, ndim=1] result = np.zeros(n, dtype=np.float32)
    cdef double uniform

    if s > 0.0:
        for idx in range(n):
            result[idx] = <float>(out[idx] / s)
    else:
        # Fallback: uniform over legal
        uniform = 1.0 / <double>len(moves)
        for (r, c) in moves:
            result[r * board_size + c] = <float>uniform

    return result


# ── expand_from_output (lazy) ─────────────────────────────────────────────
def expand_from_output(node, list moves,
                       np.ndarray[np.float32_t, ndim=1] logits,
                       float value, int board_size):
    """Expand a leaf: compute priors, store for lazy child creation.

    Does NOT create child MCTSNode objects.  Children are created on
    demand by select_child when UCB first selects them.
    """
    cdef np.ndarray[np.float32_t, ndim=1] probs = masked_softmax(logits, moves, board_size)
    cdef int r, c, n_moves, i
    n_moves = len(moves)

    cdef np.ndarray[np.intp_t, ndim=1] indices = np.empty(n_moves, dtype=np.intp)
    for i in range(n_moves):
        r, c = moves[i]
        indices[i] = r * board_size + c

    node._moves = moves
    node._priors = probs[indices]
    return value


# ── compute_threat_planes ─────────────────────────────────────────────────
def compute_threat_planes(np.ndarray[np.int8_t, ndim=2] my,
                          np.ndarray[np.int8_t, ndim=2] opp,
                          np.ndarray[np.float32_t, ndim=3] out,
                          int size):
    """Fill threat planes (channels 2-5) by scanning all line-of-5 windows.

    Channels: 2=my fours, 3=opp fours, 4=my threes, 5=opp threes.
    ~3μs on 15×15.
    """
    cdef int n = size - 4
    cdef int r, c, k, mc, oc

    # Horizontal
    for r in range(size):
        for c in range(n):
            mc = my[r,c]+my[r,c+1]+my[r,c+2]+my[r,c+3]+my[r,c+4]
            oc = opp[r,c]+opp[r,c+1]+opp[r,c+2]+opp[r,c+3]+opp[r,c+4]
            if mc == 4 and oc == 0:
                for k in range(5): out[r, c+k, 2] = 1.0
            elif mc == 3 and oc == 0:
                for k in range(5): out[r, c+k, 4] = 1.0
            if oc == 4 and mc == 0:
                for k in range(5): out[r, c+k, 3] = 1.0
            elif oc == 3 and mc == 0:
                for k in range(5): out[r, c+k, 5] = 1.0
    # Vertical
    for r in range(n):
        for c in range(size):
            mc = my[r,c]+my[r+1,c]+my[r+2,c]+my[r+3,c]+my[r+4,c]
            oc = opp[r,c]+opp[r+1,c]+opp[r+2,c]+opp[r+3,c]+opp[r+4,c]
            if mc == 4 and oc == 0:
                for k in range(5): out[r+k, c, 2] = 1.0
            elif mc == 3 and oc == 0:
                for k in range(5): out[r+k, c, 4] = 1.0
            if oc == 4 and mc == 0:
                for k in range(5): out[r+k, c, 3] = 1.0
            elif oc == 3 and mc == 0:
                for k in range(5): out[r+k, c, 5] = 1.0
    # Diagonal and anti-diagonal
    for r in range(n):
        for c in range(n):
            mc = my[r,c]+my[r+1,c+1]+my[r+2,c+2]+my[r+3,c+3]+my[r+4,c+4]
            oc = opp[r,c]+opp[r+1,c+1]+opp[r+2,c+2]+opp[r+3,c+3]+opp[r+4,c+4]
            if mc == 4 and oc == 0:
                for k in range(5): out[r+k, c+k, 2] = 1.0
            elif mc == 3 and oc == 0:
                for k in range(5): out[r+k, c+k, 4] = 1.0
            if oc == 4 and mc == 0:
                for k in range(5): out[r+k, c+k, 3] = 1.0
            elif oc == 3 and mc == 0:
                for k in range(5): out[r+k, c+k, 5] = 1.0
            # Anti-diagonal: (r, c+4) to (r+4, c)
            mc = my[r,c+4]+my[r+1,c+3]+my[r+2,c+2]+my[r+3,c+1]+my[r+4,c]
            oc = opp[r,c+4]+opp[r+1,c+3]+opp[r+2,c+2]+opp[r+3,c+1]+opp[r+4,c]
            if mc == 4 and oc == 0:
                for k in range(5): out[r+k, c+4-k, 2] = 1.0
            elif mc == 3 and oc == 0:
                for k in range(5): out[r+k, c+4-k, 4] = 1.0
            if oc == 4 and mc == 0:
                for k in range(5): out[r+k, c+4-k, 3] = 1.0
            elif oc == 3 and mc == 0:
                for k in range(5): out[r+k, c+4-k, 5] = 1.0
