#!/usr/bin/env python3
"""
Gomoku — Play against the trained AI (curses UI).
The AI uses MCTS with the trained network for strong move selection.
"""

import numpy as np
import os, glob, time, curses

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

from gomoku import (
    BOARD_SIZE, WIN_LENGTH, EMPTY, PLAYER1, PLAYER2, NUM_INPUT_PLANES,
    GomokuGame, create_model, encode_state, make_predict_fn,
    mcts_search_batched, mcts_policy,
)

# How hard the AI thinks (increase for stronger but slower play)
AI_SIMULATIONS = 400
AI_MCTS_BATCH  = 8      # small batch = better search quality


class AIPlayer:
    """AI that selects moves via MCTS backed by a trained network."""

    def __init__(self, predict_fn, simulations=AI_SIMULATIONS):
        self.predict_fn = predict_fn
        self.sims = simulations

    def get_move(self, game):
        root = mcts_search_batched(
            game, self.predict_fn,
            num_simulations=self.sims,
            batch_size=AI_MCTS_BATCH,
            c_puct=1.5,
            add_noise=False,       # no exploration noise during play
        )
        pi = mcts_policy(root, temperature=0.05)  # near-greedy
        idx = int(np.argmax(pi))
        row, col = divmod(idx, BOARD_SIZE)

        # Confidence = root value estimate
        value = root.q_value
        return row, col, value


BEST_WEIGHTS  = "weights/gomoku_best.weights.h5"
LATEST_WEIGHTS = "weights/gomoku_weights.weights.h5"


def select_weights(use_latest=False):
    """Return weights path.  Prefers best; falls back to latest."""
    if use_latest:
        if os.path.exists(LATEST_WEIGHTS):
            return LATEST_WEIGHTS, "latest"
    else:
        if os.path.exists(BEST_WEIGHTS):
            return BEST_WEIGHTS, "best"
        # Fall back to latest if no best yet
        if os.path.exists(LATEST_WEIGHTS):
            return LATEST_WEIGHTS, "latest (no best yet)"
    # Last resort: newest checkpoint file
    files = glob.glob("weights/gomoku_*.weights.h5")
    if files:
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0], "checkpoint"
    return None, None


# ── Curses UI ───────────────────────────────────────────────────────────────
def draw_board(stdscr, game, cursor_row, cursor_col, message=""):
    stdscr.clear()
    stdscr.addstr(0, 2, "=== FIVE IN A ROW (GOMOKU) ===", curses.A_BOLD)
    stdscr.addstr(1, 2, "Arrows: move | SPACE: place | U: undo | Q: quit")

    # Column header
    hdr = "    " + "".join(f"{c:3d}" for c in range(game.size))
    stdscr.addstr(3, 0, hdr)

    start = 4
    for r in range(game.size):
        line = f" {r:2d} "
        for c in range(game.size):
            cell = game.board[r, c]
            ch = {EMPTY: "\u00b7", PLAYER1: "X", PLAYER2: "O"}[cell]
            if r == cursor_row and c == cursor_col:
                line += f"[{ch}]"
            else:
                line += f" {ch} "
        stdscr.addstr(start + r, 0, line)

    srow = start + game.size + 2
    turn = "Your turn (X)" if game.current_player == PLAYER1 else "AI thinking (O)"
    stdscr.addstr(srow, 2, f"Status: {turn}")
    if message:
        stdscr.addstr(srow + 1, 2, message, curses.A_BOLD)
    stdscr.addstr(srow + 2, 2, f"Moves: {len(game.move_history)}")
    stdscr.refresh()


def play_game_curses(stdscr, ai, game):
    curses.curs_set(0)
    stdscr.nodelay(False)

    cr, cc = game.size // 2, game.size // 2
    msg = ""

    # Choose sides
    stdscr.clear()
    stdscr.addstr(0, 0, "Do you want to go first? (y/n): ")
    stdscr.refresh()
    while True:
        k = stdscr.getch()
        if k in (ord("y"), ord("Y")):
            human_turn = True; break
        if k in (ord("n"), ord("N")):
            human_turn = False; msg = "AI goes first …"; break

    game_over = False
    while not game_over:
        draw_board(stdscr, game, cr, cc, msg)
        msg = ""

        if human_turn:
            k = stdscr.getch()
            if k in (ord("q"), ord("Q")):
                return False
            elif k in (ord("u"), ord("U")):
                if game.undo_move() and game.undo_move():
                    msg = "Undone!"
                else:
                    msg = "Nothing to undo"
            elif k == curses.KEY_UP:    cr = max(0, cr - 1)
            elif k == curses.KEY_DOWN:  cr = min(game.size - 1, cr + 1)
            elif k == curses.KEY_LEFT:  cc = max(0, cc - 1)
            elif k == curses.KEY_RIGHT: cc = min(game.size - 1, cc + 1)
            elif k == ord(" "):
                if game.board[cr, cc] != EMPTY:
                    msg = "Occupied!"; continue
                reward, done = game.make_move(cr, cc)
                if done:
                    end = "You win!" if reward == 1 else "Draw!"
                    draw_board(stdscr, game, cr, cc, f"{end}  Press any key …")
                    stdscr.getch(); game_over = True
                else:
                    human_turn = False
        else:
            msg = f"AI thinking ({ai.sims} sims) …"
            draw_board(stdscr, game, cr, cc, msg)
            row, col, val = ai.get_move(game)
            assert game.board[row, col] == 0, (
                f"ILLEGAL AI MOVE ({row},{col}), "
                f"board={game.board[row, col]}, "
                f"stones={int(np.count_nonzero(game.board))}"
            )
            reward, done = game.make_move(row, col)
            if done:
                end = "AI wins!" if reward == 1 else "Draw!"
                draw_board(stdscr, game, cr, cc, f"{end}  Press any key …")
                stdscr.getch(); game_over = True
            else:
                msg = f"AI played ({row},{col})  eval {val:+.2f}"
                human_turn = True
    return True


def main(stdscr, use_latest=False):
    wf, label = select_weights(use_latest)
    if not wf:
        stdscr.clear()
        stdscr.addstr(0, 0, "No weights found in weights/ — run train.py first.")
        stdscr.addstr(1, 0, "Press any key to exit …")
        stdscr.refresh(); stdscr.getch(); return

    stdscr.clear()
    stdscr.addstr(0, 0, "Loading model …")
    stdscr.addstr(1, 0, f"  Weights: {wf} ({label})")
    stdscr.refresh()

    model = create_model()
    model.load_weights(wf)
    predict_fn = make_predict_fn(model)
    # Warmup compiled graph
    predict_fn(np.zeros((1, BOARD_SIZE, BOARD_SIZE, NUM_INPUT_PLANES), dtype=np.float32))
    ai = AIPlayer(predict_fn)

    stdscr.addstr(2, 0, "  Ready!  Press any key to start …")
    stdscr.refresh(); stdscr.getch()

    while True:
        game = GomokuGame()
        if not play_game_curses(stdscr, ai, game):
            break
        stdscr.clear()
        stdscr.addstr(0, 0, "Play again? (y/n): ")
        stdscr.refresh()
        while True:
            k = stdscr.getch()
            if k in (ord("n"), ord("N")): return
            if k in (ord("y"), ord("Y")): break

    stdscr.clear()
    stdscr.addstr(0, 0, "Thanks for playing!")
    stdscr.refresh(); time.sleep(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Play Gomoku against the AI")
    parser.add_argument("--latest", action="store_true",
                        help="Use latest training weights instead of best checkpoint")
    args = parser.parse_args()
    curses.wrapper(main, use_latest=args.latest)
