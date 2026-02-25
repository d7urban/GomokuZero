#!/usr/bin/env python3
"""
Gomoku pipeline sanity checks.
Run before committing to a long training run to catch wiring bugs early.

Usage:  python sanity_check.py
"""

import numpy as np
import os, sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf
from tensorflow import keras

tf.get_logger().setLevel("ERROR")

from gomoku import (
    BOARD_SIZE, PLAYER1, PLAYER2, EMPTY, NUM_INPUT_PLANES,
    GomokuGame, create_model, encode_state,
    mcts_search, mcts_search_batched, mcts_policy, get_candidate_moves,
    mcts_begin, mcts_expand_root, mcts_select_leaves, mcts_process_results,
)

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}  {detail}")
        failed += 1


# ── 0. Optimizer availability ─────────────────────────────────────────────
print("\n[0] Optimizer availability")

_adamw_cls = None
_adamw_source = None

# Try locations in order of preference
try:
    _adamw_cls = keras.optimizers.AdamW
    _adamw_source = "keras.optimizers.AdamW"
except AttributeError:
    pass

if _adamw_cls is None:
    try:
        _adamw_cls = keras.optimizers.experimental.AdamW
        _adamw_source = "keras.optimizers.experimental.AdamW"
    except AttributeError:
        pass

check(
    f"AdamW available (TF {tf.__version__})",
    _adamw_cls is not None,
    "AdamW not found — train.py requires it. Upgrade TF or install tf-nightly.",
)
if _adamw_cls is not None:
    # Verify it can be instantiated
    try:
        _test_opt = _adamw_cls(learning_rate=1e-3, weight_decay=1e-4)
        check(f"AdamW instantiates OK (via {_adamw_source})", True)
    except Exception as e:
        check(f"AdamW instantiates OK (via {_adamw_source})", False, str(e))


# ── 1. Encoding symmetry ───────────────────────────────────────────────────
print("\n[1] Encoding symmetry (P1-to-move vs P2-to-move)")

g1 = GomokuGame()
g1.make_move(7, 7)   # P1 plays centre
g1.make_move(7, 8)   # P2 plays next to it
# Now P1 to move.  Board: (7,7)=+1, (7,8)=-1

g2 = GomokuGame()
g2.board[7, 7] = PLAYER2
g2.board[7, 8] = PLAYER1
g2.current_player = PLAYER2
# Mirror: same pattern but colours swapped, P2 to move

enc1 = encode_state(g1)
enc2 = encode_state(g2)

check(
    "Swapped-colour positions produce identical encodings",
    np.allclose(enc1, enc2),
    f"max diff = {np.max(np.abs(enc1 - enc2)):.6f}",
)

check(
    f"Encoding has {NUM_INPUT_PLANES} planes (no colour leak)",
    enc1.shape == (BOARD_SIZE, BOARD_SIZE, NUM_INPUT_PLANES),
    f"got shape {enc1.shape}",
)


# ── 2. Value sign symmetry (untrained net — should be near zero) ──────────
print("\n[2] Value sign check (untrained network)")

model = create_model()

g = GomokuGame()
g.make_move(7, 7)
g.make_move(3, 3)
# P1 to move

enc_p1 = encode_state(g)
logits_p1, val_p1 = model(enc_p1[np.newaxis], training=False)
val_p1 = float(val_p1.numpy().ravel()[0])

# Swap perspective: make it P2's view of the same board
g_swap = g.copy()
g_swap.current_player *= -1
enc_p2 = encode_state(g_swap)
logits_p2, val_p2 = model(enc_p2[np.newaxis], training=False)
val_p2 = float(val_p2.numpy().ravel()[0])

check(
    "Untrained value is near zero for both sides",
    abs(val_p1) < 0.5 and abs(val_p2) < 0.5,
    f"v(P1)={val_p1:+.4f}, v(P2)={val_p2:+.4f}",
)

# Encoding planes should swap when perspective swaps
check(
    "Swapping current_player swaps the two planes",
    np.allclose(enc_p1[:, :, 0], enc_p2[:, :, 1])
    and np.allclose(enc_p1[:, :, 1], enc_p2[:, :, 0]),
    "planes did not swap correctly",
)


# ── 3. Policy legality & NaN check ────────────────────────────────────────
print("\n[3] Policy legality and NaN checks")

# Test on a few random positions
rng = np.random.RandomState(42)
for trial in range(5):
    g = GomokuGame()
    n_moves = rng.randint(0, 30)
    for _ in range(n_moves):
        moves = g.get_valid_moves()
        if not moves:
            break
        r, c = moves[rng.randint(len(moves))]
        reward, done = g.make_move(r, c)
        if done:
            break

    if not g.get_valid_moves():
        continue

    root = mcts_search(g, model, num_simulations=50, add_noise=False)
    pi = mcts_policy(root, temperature=1.0)

    valid = set(g.get_valid_moves())
    has_nan = np.any(np.isnan(pi))
    mass_on_illegal = 0.0
    for idx in range(len(pi)):
        r, c = divmod(idx, BOARD_SIZE)
        if pi[idx] > 0 and (r, c) not in valid:
            mass_on_illegal += pi[idx]

    check(
        f"Trial {trial}: no NaN in policy",
        not has_nan,
        f"NaN count: {np.sum(np.isnan(pi))}",
    )
    check(
        f"Trial {trial}: zero mass on illegal moves",
        mass_on_illegal < 1e-8,
        f"illegal mass = {mass_on_illegal:.6f}",
    )
    check(
        f"Trial {trial}: policy sums to ~1",
        abs(pi.sum() - 1.0) < 1e-5,
        f"sum = {pi.sum():.6f}",
    )


# ── 3b. Tactical MCTS sanity (must take immediate win) ────────────────────
print("\n[3b] MCTS tactical sanity (immediate win)")

def _zero_predict(batch):
    b = batch.shape[0]
    logits = np.zeros((b, BOARD_SIZE * BOARD_SIZE), dtype=np.float32)
    values = np.zeros((b, 1), dtype=np.float32)
    return logits, values

g_win = GomokuGame()
forced_seq = [
    (7, 7), (7, 6),
    (7, 8), (6, 6),
    (7, 9), (6, 7),
    (7, 10), (6, 8),
]
for r, c in forced_seq:
    reward, done = g_win.make_move(r, c)
    if done:
        break

check(
    "Setup: non-terminal position with P1 to move",
    (not done) and g_win.current_player == PLAYER1,
    f"done={done}, current_player={g_win.current_player}",
)

g_tmp = g_win.copy()
reward_tmp, done_tmp = g_tmp.make_move(7, 11)
check(
    "Setup: (7,11) is an immediate winning move",
    done_tmp and reward_tmp == 1,
    f"done={done_tmp}, reward={reward_tmp}",
)

root_win = mcts_search(g_win, _zero_predict, num_simulations=96, add_noise=False)
pi_win = mcts_policy(root_win, temperature=0.0)
best_idx = int(np.argmax(pi_win))
best_move = divmod(best_idx, BOARD_SIZE)
check(
    "Sequential MCTS takes immediate win",
    best_move == (7, 11),
    f"best={best_move}",
)

win_child = root_win.children.get((7, 11))
check(
    "Sequential MCTS: winning child scores as good for parent",
    win_child is not None and (-win_child.q_value) > 0.9,
    f"child_q={win_child.q_value if win_child is not None else 'missing'}",
)

root_win_b = mcts_search_batched(
    g_win, _zero_predict,
    num_simulations=96,
    batch_size=8,
    add_noise=False,
)
pi_win_b = mcts_policy(root_win_b, temperature=0.0)
best_idx_b = int(np.argmax(pi_win_b))
best_move_b = divmod(best_idx_b, BOARD_SIZE)
check(
    "Batched MCTS takes immediate win",
    best_move_b == (7, 11),
    f"best={best_move_b}",
)


# ── 4. Network prior NaN stress test (extreme logits) ─────────────────────
print("\n[4] Masked softmax robustness (extreme inputs)")

g = GomokuGame()
g.make_move(7, 7)

# Manually feed the model to check priors don't produce NaN
enc = encode_state(g)
logits, _ = model(enc[np.newaxis], training=False)
logits = logits.numpy().ravel()

# Simulate extreme logits
logits_extreme = logits.copy()
logits_extreme[:] = -1e6
logits_extreme[7 * BOARD_SIZE + 8] = 1e6  # one extreme value

mask = np.full_like(logits_extreme, -1e9)
moves = g.get_valid_moves()
for r, c in moves:
    mask[r * BOARD_SIZE + c] = 0.0
logits_extreme = logits_extreme + mask
logits_extreme = logits_extreme - logits_extreme.max()
probs = np.exp(logits_extreme)
s = probs.sum()
if s <= 0 or not np.isfinite(s):
    probs[:] = 0.0
    for r, c in moves:
        probs[r * BOARD_SIZE + c] = 1.0
    probs /= probs.sum()
else:
    probs /= s

check("No NaN after extreme logits", not np.any(np.isnan(probs)))
check("Probs sum to ~1 after extreme logits", abs(probs.sum() - 1.0) < 1e-5)


# ── 5. Candidate moves adaptive threshold ─────────────────────────────────
print("\n[5] Candidate move generation")

g = GomokuGame()
moves_empty = get_candidate_moves(g.board)
check(
    f"Empty board returns all {BOARD_SIZE*BOARD_SIZE} squares",
    len(moves_empty) == BOARD_SIZE * BOARD_SIZE,
    f"got {len(moves_empty)}",
)

g.make_move(7, 7)
moves_1 = get_candidate_moves(g.board)
check(
    "1 stone (sparse): returns all empty squares",
    len(moves_1) == BOARD_SIZE * BOARD_SIZE - 1,
    f"got {len(moves_1)}",
)

# Place enough stones to trigger nearby filtering (density_threshold=2)
g2 = GomokuGame()
positions = [(7,7),(7,8),(7,9)]
for r, c in positions:
    g2.board[r, c] = PLAYER1 if len(g2.move_history) % 2 == 0 else PLAYER2
    g2.move_history.append((r, c, g2.board[r, c]))
moves_dense = get_candidate_moves(g2.board)
total_empty = np.sum(g2.board == EMPTY)
check(
    "3 stones (dense): nearby filter active, fewer candidates than all empties",
    len(moves_dense) < total_empty,
    f"candidates={len(moves_dense)}, empties={total_empty}",
)


# ── 6. Overfit micro-test ─────────────────────────────────────────────────
print("\n[6] Overfit micro-test (128 fixed positions, 300 steps)")

micro_model = create_model()
optimizer = keras.optimizers.Adam(learning_rate=1e-3)

# Generate a small fixed dataset with SHARP targets (not uniform)
# so the theoretical minimum cross-entropy is near zero.
rng = np.random.RandomState(123)
states, target_pis, target_vs = [], [], []

for _ in range(128):
    g = GomokuGame()
    n = rng.randint(1, 20)
    for _ in range(n):
        moves = g.get_valid_moves()
        if not moves:
            break
        r, c = moves[rng.randint(len(moves))]
        reward, done = g.make_move(r, c)
        if done:
            break
    if not g.get_valid_moves():
        g = GomokuGame()
        g.make_move(7, 7)

    enc = encode_state(g)
    states.append(enc)

    # Sharp target: pick one random valid move (cross-entropy floor ≈ 0)
    valid = g.get_valid_moves()
    pi = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
    r, c = valid[rng.randint(len(valid))]
    pi[r * BOARD_SIZE + c] = 1.0
    target_pis.append(pi)

    target_vs.append(rng.choice([-1.0, 0.0, 1.0]))

states = np.array(states)
target_pis = np.array(target_pis)
target_vs = np.array(target_vs, dtype=np.float32)

@tf.function
def train_micro(s, p, v):
    with tf.GradientTape() as tape:
        logits, val = micro_model(s, training=True)
        val = tf.squeeze(val, axis=1)
        ploss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=p, logits=logits)
        )
        vloss = tf.reduce_mean(tf.square(v - val))
        loss = ploss + vloss
    grads = tape.gradient(loss, micro_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, micro_model.trainable_variables, strict=False))
    return ploss, vloss, loss

initial_loss = None
final_loss = None
for step in range(300):
    pl, vl, tl = train_micro(states, target_pis, target_vs)
    if step == 0:
        initial_loss = float(tl)
    if step == 299:
        final_loss = float(tl)

ratio = final_loss / initial_loss if initial_loss > 0 else float("inf")
check(
    f"Loss decreased significantly (init={initial_loss:.3f} -> final={final_loss:.3f})",
    ratio < 0.15,
    f"ratio = {ratio:.3f} (want < 0.15)",
)
check(
    "Final policy loss < 1.0 (sharp targets, floor near 0)",
    float(pl) < 1.0,
    f"policy loss = {float(pl):.4f}",
)


# ── 7. End-to-end smoke test ──────────────────────────────────────────────
print("\n[7] End-to-end smoke: self-play → replay → train → weight sync → checkpoint")

import tempfile, shutil, pickle
from collections import deque
from gomoku import mcts_search, mcts_search_batched, mcts_policy, select_action

SMOKE_SIMS = 20          # tiny MCTS for speed
SMOKE_GAMES = 2
SMOKE_BATCH = 32
SMOKE_TRAIN_STEPS = 5

smoke_dir = tempfile.mkdtemp(prefix="gomoku_smoke_")
smoke_weights = os.path.join(smoke_dir, "smoke.weights.h5")
smoke_checkpoint = os.path.join(smoke_dir, "smoke_cp.weights.h5")
smoke_config = os.path.join(smoke_dir, "smoke_config.pkl")

try:
    # --- Phase 1: create model & save initial weights ---
    smoke_model = create_model()
    smoke_model.save_weights(smoke_weights)
    check("Phase 1: initial weights saved", os.path.exists(smoke_weights))

    # --- Phase 2: self-play (in-process, no multiprocessing) ---
    replay = deque(maxlen=10_000)
    total_moves = 0

    for game_i in range(SMOKE_GAMES):
        game = GomokuGame()
        trajectory = []
        done = False
        move_num = 0
        reward = 0

        while not done:
            root = mcts_search(
                game, smoke_model,
                num_simulations=SMOKE_SIMS,
                c_puct=1.5,
                add_noise=True,
                dirichlet_alpha=0.15,
                noise_frac=0.25,
            )
            temp = 1.0 if move_num < 10 else 0.1
            pi = mcts_policy(root, temperature=temp)

            # Validate policy on each move
            pi_ok = True
            if np.any(np.isnan(pi)):
                check(f"Phase 2: game {game_i} move {move_num} policy has no NaN", False,
                      f"NaN count: {np.sum(np.isnan(pi))}")
                pi_ok = False
            if abs(pi.sum() - 1.0) >= 1e-5:
                check(f"Phase 2: game {game_i} move {move_num} policy sums to ~1", False,
                      f"sum = {pi.sum():.6f}")
                pi_ok = False
            if not pi_ok:
                # Abort self-play — can't trust downstream results
                done = True
                break

            trajectory.append((encode_state(game), pi, game.current_player))
            row, col = select_action(pi)
            reward, done = game.make_move(row, col)
            move_num += 1

        # Determine winner.  make_move() does NOT flip current_player on a
        # terminal move, so game.current_player is still the player who moved.
        if reward == 1:
            winner = game.current_player
        elif reward == -1:
            winner = -game.current_player
        else:
            winner = 0

        for state, pi, player in trajectory:
            if winner == 0:
                outcome = 0.0
            elif player == winner:
                outcome = 1.0
            else:
                outcome = -1.0
            replay.append((state, pi, np.float32(outcome)))

        total_moves += move_num

    check(
        f"Phase 2: self-play completed ({SMOKE_GAMES} games, {total_moves} total moves)",
        len(replay) > 0,
        f"replay has {len(replay)} positions",
    )

    # Quick data-shape sanity
    s0, p0, v0 = replay[0]
    check(
        "Phase 2: replay entry shapes correct",
        s0.shape == (BOARD_SIZE, BOARD_SIZE, NUM_INPUT_PLANES)
        and p0.shape == (BOARD_SIZE * BOARD_SIZE,)
        and np.isscalar(v0),
        f"state={s0.shape}, pi={p0.shape}, v type={type(v0)}",
    )

    # --- Phase 2b: one game with batched MCTS (verify it integrates) ---
    game_b = GomokuGame()
    done_b = False
    move_num_b = 0
    while not done_b:
        root_b = mcts_search_batched(
            game_b, smoke_model,
            num_simulations=SMOKE_SIMS,
            batch_size=4,
            c_puct=1.5,
            add_noise=True,
        )
        pi_b = mcts_policy(root_b, temperature=1.0)
        if np.any(np.isnan(pi_b)) or abs(pi_b.sum() - 1.0) >= 1e-5:
            check("Phase 2b: batched MCTS policy valid", False,
                  f"nan={np.any(np.isnan(pi_b))}, sum={pi_b.sum():.6f}")
            break
        row_b, col_b = select_action(pi_b)
        _, done_b = game_b.make_move(row_b, col_b)
        move_num_b += 1
    else:
        check(
            f"Phase 2b: batched MCTS self-play completed ({move_num_b} moves)",
            True,
        )

    # --- Phase 3: training steps ---
    smoke_opt = keras.optimizers.Adam(learning_rate=1e-3)

    @tf.function
    def smoke_train_step(mdl, opt, states, target_pi, target_v):
        with tf.GradientTape() as tape:
            logits, value = mdl(states, training=True)
            value = tf.squeeze(value, axis=1)
            ploss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=target_pi, logits=logits)
            )
            vloss = tf.reduce_mean(tf.square(target_v - value))
            loss = ploss + vloss
        grads = tape.gradient(loss, mdl.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        opt.apply_gradients(zip(grads, mdl.trainable_variables, strict=False))
        return ploss, vloss, loss

    actual_batch = min(SMOKE_BATCH, len(replay))
    losses = []
    for _step in range(SMOKE_TRAIN_STEPS):
        idxs = np.random.choice(len(replay), actual_batch, replace=len(replay) < actual_batch)
        batch = [replay[i] for i in idxs]
        s = np.array([b[0] for b in batch])
        p = np.array([b[1] for b in batch])
        v = np.array([b[2] for b in batch])
        pl, vl, tl = smoke_train_step(smoke_model, smoke_opt, s, p, v)
        losses.append(float(tl))

    check(
        f"Phase 3: training ran ({SMOKE_TRAIN_STEPS} steps, losses finite)",
        all(np.isfinite(l) for l in losses),
        f"losses = {losses}",
    )
    check(
        "Phase 3: loss is not stuck at exactly same value",
        len(set(f"{l:.6f}" for l in losses)) > 1 or SMOKE_TRAIN_STEPS == 1,
        f"all losses identical: {losses[0]:.6f}",
    )

    # --- Phase 4: weight sync (save → load into fresh model → compare) ---
    smoke_model.save_weights(smoke_weights)

    smoke_model2 = create_model()
    smoke_model2.load_weights(smoke_weights)

    # Both models should produce identical outputs on the same input
    test_input = np.random.randn(1, BOARD_SIZE, BOARD_SIZE, NUM_INPUT_PLANES).astype(np.float32)
    out1_p, out1_v = smoke_model(test_input, training=False)
    out2_p, out2_v = smoke_model2(test_input, training=False)
    policy_match = np.allclose(out1_p.numpy(), out2_p.numpy(), atol=1e-5)
    value_match = np.allclose(out1_v.numpy(), out2_v.numpy(), atol=1e-5)

    check(
        "Phase 4: weight sync — loaded model matches saved model (policy)",
        policy_match,
        f"max diff = {np.max(np.abs(out1_p.numpy() - out2_p.numpy())):.8f}",
    )
    check(
        "Phase 4: weight sync — loaded model matches saved model (value)",
        value_match,
        f"max diff = {np.max(np.abs(out1_v.numpy() - out2_v.numpy())):.8f}",
    )

    # --- Phase 5: checkpoint save & reload with config ---
    smoke_model.save_weights(smoke_checkpoint)
    with open(smoke_config, "wb") as f:
        pickle.dump({"board_size": BOARD_SIZE, "total_games": SMOKE_GAMES}, f)

    check("Phase 5: checkpoint file exists", os.path.exists(smoke_checkpoint))
    check("Phase 5: config file exists", os.path.exists(smoke_config))

    with open(smoke_config, "rb") as f:
        cfg = pickle.load(f)
    check(
        "Phase 5: config round-trips correctly",
        cfg["board_size"] == BOARD_SIZE and cfg["total_games"] == SMOKE_GAMES,
        f"got {cfg}",
    )

    # Load checkpoint into yet another fresh model
    smoke_model3 = create_model()
    smoke_model3.load_weights(smoke_checkpoint)
    out3_p, out3_v = smoke_model3(test_input, training=False)
    check(
        "Phase 5: checkpoint round-trip — outputs match",
        np.allclose(out1_p.numpy(), out3_p.numpy(), atol=1e-5)
        and np.allclose(out1_v.numpy(), out3_v.numpy(), atol=1e-5),
    )

finally:
    shutil.rmtree(smoke_dir, ignore_errors=True)

print("  (end-to-end smoke test done)")


# ── 8. Batched MCTS correctness ─────────────────────────────────────────────
print("\n[8] Batched MCTS (virtual loss) correctness")

# 8a. Basic legality, NaN, and sum-to-1 on random positions
rng8 = np.random.RandomState(99)
for trial in range(5):
    g = GomokuGame()
    n_moves = rng8.randint(0, 30)
    for _ in range(n_moves):
        moves = g.get_valid_moves()
        if not moves:
            break
        r, c = moves[rng8.randint(len(moves))]
        reward, done = g.make_move(r, c)
        if done:
            break

    if not g.get_valid_moves():
        continue

    root_b = mcts_search_batched(g, model, num_simulations=50, batch_size=8,
                                  add_noise=False)
    pi_b = mcts_policy(root_b, temperature=1.0)

    valid = set(g.get_valid_moves())
    has_nan = np.any(np.isnan(pi_b))
    mass_on_illegal = 0.0
    for idx in range(len(pi_b)):
        r, c = divmod(idx, BOARD_SIZE)
        if pi_b[idx] > 0 and (r, c) not in valid:
            mass_on_illegal += pi_b[idx]

    check(
        f"Batched trial {trial}: no NaN in policy",
        not has_nan,
        f"NaN count: {np.sum(np.isnan(pi_b))}",
    )
    check(
        f"Batched trial {trial}: zero mass on illegal moves",
        mass_on_illegal < 1e-8,
        f"illegal mass = {mass_on_illegal:.6f}",
    )
    check(
        f"Batched trial {trial}: policy sums to ~1",
        abs(pi_b.sum() - 1.0) < 1e-5,
        f"sum = {pi_b.sum():.6f}",
    )

# 8b. Visit-count agreement between sequential and batched MCTS.
# Both are stochastic (Dirichlet noise off helps, but expansion order still
# causes divergence), so we only check that the top-1 move agrees on a
# deterministic-ish position and that total visit counts are in the right
# ballpark.
print()
g_agree = GomokuGame()
g_agree.make_move(7, 7)  # single stone — deterministic enough

n_sims_agree = 200
root_seq = mcts_search(g_agree, model, num_simulations=n_sims_agree,
                       add_noise=False)
pi_seq = mcts_policy(root_seq, temperature=0.01)  # greedy

root_bat = mcts_search_batched(g_agree, model, num_simulations=n_sims_agree,
                                batch_size=8, add_noise=False)
pi_bat = mcts_policy(root_bat, temperature=0.01)

top_seq = np.argmax(pi_seq)
top_bat = np.argmax(pi_bat)

check(
    "Sequential vs batched: same top move (200 sims, no noise)",
    top_seq == top_bat,
    f"seq={divmod(top_seq, BOARD_SIZE)}, bat={divmod(top_bat, BOARD_SIZE)}",
)

# Total visit counts at root should be close to num_simulations (plus the
# initial expansion visit).  Batched MCTS may overshoot by up to batch_size-1.
seq_visits = root_seq.visit_count
bat_visits = root_bat.visit_count
check(
    f"Sequential root visits ≈ {n_sims_agree}",
    abs(seq_visits - n_sims_agree) <= 5,
    f"got {seq_visits}",
)
check(
    f"Batched root visits in [{n_sims_agree}, {n_sims_agree + 8}]",
    n_sims_agree <= bat_visits <= n_sims_agree + 8,
    f"got {bat_visits}",
)

# 8c. Different batch sizes should all produce valid output
for bs in [1, 4, 16, 64]:
    root_bs = mcts_search_batched(g_agree, model, num_simulations=50,
                                   batch_size=bs, add_noise=False)
    pi_bs = mcts_policy(root_bs, temperature=1.0)
    check(
        f"batch_size={bs:2d}: valid policy (no NaN, sums to ~1)",
        not np.any(np.isnan(pi_bs)) and abs(pi_bs.sum() - 1.0) < 1e-5,
        f"nan={np.any(np.isnan(pi_bs))}, sum={pi_bs.sum():.6f}",
    )

# 8d. Virtual loss cleanup: after search completes, every node's value_sum
# and visit_count should be self-consistent (no residual virtual loss).
# Specifically, for every child |q_value| should be ≤ 1 (values are tanh-bounded
# and outcomes are in {-1, 0, +1}).
def _check_tree_values(node, depth=0, max_depth=3):
    """Recursively check that no node has out-of-range q_value."""
    issues = []
    if node.visit_count > 0 and abs(node.q_value) > 1.05:
        issues.append((depth, node.visit_count, node.q_value))
    if depth < max_depth:
        for child in node.children.values():
            issues.extend(_check_tree_values(child, depth + 1, max_depth))
    return issues

issues = _check_tree_values(root_bat)
check(
    "Virtual loss cleanup: all node q_values in [-1.05, 1.05]",
    len(issues) == 0,
    f"{len(issues)} nodes out of range, first: depth={issues[0][0]} "
    f"visits={issues[0][1]} q={issues[0][2]:.4f}" if issues else "",
)


# ── 9. Phased MCTS API ──────────────────────────────────────────────────────
print("\n[9] Phased MCTS API")

# 9a. Assert fires if select_leaves called before expand_root
g_phased = GomokuGame()
g_phased.make_move(7, 7)

ctx_p, rs_p = mcts_begin(g_phased, num_simulations=50, batch_size=8, add_noise=False)
try:
    mcts_select_leaves(ctx_p)
    check("Assert: select_leaves before expand_root", False, "should have raised")
except AssertionError:
    check("Assert: select_leaves before expand_root fires correctly", True)

# 9b. Normal flow — visit counts correct
ctx_p, rs_p = mcts_begin(g_phased, num_simulations=100, batch_size=8, add_noise=False)
lg_p, vl_p = model(rs_p[np.newaxis], training=False)
mcts_expand_root(ctx_p, lg_p.numpy()[0], vl_p.numpy()[0])

while ctx_p["sims_done"] < ctx_p["sims_target"]:
    leaves = mcts_select_leaves(ctx_p)
    if leaves:
        b = np.array(leaves, dtype=np.float32)
        l2, v2 = model(b, training=False)
        mcts_process_results(ctx_p, l2.numpy(), v2.numpy().ravel())
    else:
        mcts_process_results(ctx_p)
    # Verify clearing after each round
    check(
        f"Round cleanup: pending cleared (sims={ctx_p['sims_done']})",
        ctx_p["pending"] is None and ctx_p["eval_list"] is None and ctx_p["n_batch"] == 0,
    )

phased_visits = sum(c.visit_count for c in ctx_p["root"].children.values())
check(
    f"Phased MCTS: visit count = {phased_visits} (expect 100)",
    phased_visits == 100,
)

# 9c. Game state preserved after phased search
check(
    "Phased MCTS: game state preserved",
    len(g_phased.move_history) == 1 and g_phased.board[7, 7] == PLAYER1,
)

# 9d. Shape mismatch assert
ctx_bad, rs_bad = mcts_begin(g_phased, num_simulations=50, batch_size=8, add_noise=False)
lg_bad, vl_bad = model(rs_bad[np.newaxis], training=False)
mcts_expand_root(ctx_bad, lg_bad.numpy()[0], vl_bad.numpy()[0])
leaves_bad = mcts_select_leaves(ctx_bad)
if len(leaves_bad) > 1:
    # Feed wrong number of results — should assert
    b_bad = np.array(leaves_bad[:1], dtype=np.float32)
    l_bad, v_bad = model(b_bad, training=False)
    try:
        mcts_process_results(ctx_bad, l_bad.numpy(), v_bad.numpy().ravel())
        check("Assert: shape mismatch in process_results", False, "should have raised")
    except AssertionError:
        check("Assert: shape mismatch in process_results fires correctly", True)
else:
    # Only 1 leaf — can't test mismatch, skip
    check("Assert: shape mismatch (skipped, only 1 leaf)", True)


# ── 10. Interleaved multi-game coordinator ─────────────────────────────────
print("\n[10] Interleaved multi-game coordinator")

from gomoku import select_action

N_INTERLEAVED = 4
INTER_SIMS = 30
INTER_BATCH = 8

# Set up games
class _SG:
    __slots__ = ('game','sims','trajectory','winner','finished','move_num',
                 'ctx','root_state','phase','move_sims')

inter_games = []
for _ in range(N_INTERLEAVED):
    sg = _SG()
    sg.game = GomokuGame()
    sg.game.make_move(7, 7)  # one opening move
    sg.sims = INTER_SIMS; sg.trajectory = []; sg.winner = 0
    sg.finished = False; sg.ctx = None; sg.root_state = None
    sg.phase = "new_move"; sg.move_sims = INTER_SIMS
    sg.move_num = len(sg.game.move_history)
    inter_games.append(sg)

# Run coordinator for a limited number of moves (not full games — too slow)
MAX_MOVES = 5
total_gpu_calls = 0

for _ in range(MAX_MOVES * 200):  # safety bound on coordinator rounds
    active = [g for g in inter_games if not g.finished and g.move_num < MAX_MOVES + 1]
    if not active:
        break

    states = []; root_evals = []; leaf_evals = []

    for g in active:
        if g.phase == "new_move":
            g.move_sims = g.sims
            g.ctx, g.root_state = mcts_begin(
                g.game, num_simulations=g.move_sims, batch_size=INTER_BATCH,
                c_puct=1.5, add_noise=True, dirichlet_alpha=0.15, noise_frac=0.25)
            idx = len(states); states.append(g.root_state)
            root_evals.append((g, idx)); g.phase = "root_pending"
        elif g.phase == "root_pending":
            pass
        elif g.phase == "searching":
            leaves = mcts_select_leaves(g.ctx)
            if leaves:
                start = len(states); states.extend(leaves)
                leaf_evals.append((g, start, len(leaves)))
            else:
                mcts_process_results(g.ctx)

    if states:
        batch = np.array(states, dtype=np.float32)
        lg, vl = model(batch, training=False)
        lg_np = lg.numpy(); vl_np = vl.numpy().ravel()
        total_gpu_calls += 1

        for g, idx in root_evals:
            mcts_expand_root(g.ctx, lg_np[idx], vl_np[idx])
            g.phase = "searching"
        for g, start, count in leaf_evals:
            mcts_process_results(g.ctx, lg_np[start:start+count], vl_np[start:start+count])

    for g in active:
        if g.ctx is not None and g.ctx["sims_done"] >= g.ctx["sims_target"]:
            pi = mcts_policy(g.ctx["root"], temperature=1.0)
            g.trajectory.append((g.root_state, pi, g.game.current_player, g.move_sims))
            row, col = select_action(pi)
            reward, done = g.game.make_move(row, col)
            g.move_num += 1
            if done:
                if reward == 1: g.winner = g.game.current_player
                elif reward == -1: g.winner = -g.game.current_player
                g.finished = True
            else:
                g.phase = "new_move"
            g.ctx = None

# Verify all games made progress
for i, g in enumerate(inter_games):
    check(
        f"Interleaved game {i}: made moves ({len(g.trajectory)} trajectory entries)",
        len(g.trajectory) > 0,
    )
    # Verify trajectory entries have correct shape
    if g.trajectory:
        s, pi, player, sims = g.trajectory[0]
        check(
            f"Interleaved game {i}: trajectory entry shapes",
            s.shape == (BOARD_SIZE, BOARD_SIZE, NUM_INPUT_PLANES) and
            pi.shape == (BOARD_SIZE * BOARD_SIZE,) and
            abs(pi.sum() - 1.0) < 1e-5,
        )

check(
    f"Interleaved: GPU called {total_gpu_calls} times (shared batching)",
    total_gpu_calls > 0,
)
# With shared batching, GPU calls should be fewer than N_games * moves * rounds_per_move
sequential_estimate = N_INTERLEAVED * MAX_MOVES * (INTER_SIMS // INTER_BATCH + 1)
check(
    f"Interleaved: fewer GPU calls ({total_gpu_calls}) than sequential estimate ({sequential_estimate})",
    total_gpu_calls < sequential_estimate,
)


# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("All checks passed — pipeline looks healthy.")
else:
    print("Some checks failed — investigate before training.")
sys.exit(1 if failed else 0)
