#!/usr/bin/env python3
"""
Gomoku — Play against the trained AI (curses UI).
The AI uses MCTS with the trained network for strong move selection.
"""

import numpy as np
import os, time, curses

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

from gomoku import (
    BOARD_SIZE, EMPTY, PLAYER1, PLAYER2,
    GomokuGame,
)
from entrypoint_shared import (
    AIPlayer,
    resolve_difficulty,
    select_weights,
    load_model_and_predict_fn,
)

# ── Curses UI ───────────────────────────────────────────────────────────────
def draw_board(stdscr, game, cursor_row, cursor_col, human_player, message=""):
    stdscr.clear()
    stdscr.addstr(0, 2, "=== FIVE IN A ROW (GOMOKU) ===", curses.A_BOLD)
    stdscr.addstr(1, 2, "Arrows: move | SPACE: place | U: undo | Q: quit")

    # Column header
    hdr = "   " + "".join(f"{c:3d}" for c in range(game.size))
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
    side = "X" if game.current_player == PLAYER1 else "O"
    actor = "Your turn" if game.current_player == human_player else "AI thinking"
    turn = f"{actor} ({side})"
    stdscr.addstr(srow, 2, f"Status: {turn}")
    if message:
        stdscr.addstr(srow + 1, 2, message, curses.A_BOLD)
    stdscr.addstr(srow + 2, 2, f"Moves: {len(game.move_history)}")
    stdscr.refresh()


def _show_text_and_wait(stdscr, *lines):
    stdscr.clear()
    for idx, line in enumerate(lines):
        stdscr.addstr(idx, 0, line)
    stdscr.refresh()
    stdscr.getch()


def _prompt_human_first(stdscr):
    stdscr.clear()
    stdscr.addstr(0, 0, "Do you want to go first? (y/n): ")
    stdscr.refresh()
    while True:
        k = stdscr.getch()
        if k in (ord("y"), ord("Y")):
            return True, ""
        if k in (ord("n"), ord("N")):
            return False, "AI goes first …"


def _show_game_end(stdscr, game, cr, cc, human_player, end_text):
    draw_board(stdscr, game, cr, cc, human_player, f"{end_text}  Press any key …")
    stdscr.getch()


def _handle_human_turn(stdscr, game, cr, cc, human_player):
    k = stdscr.getch()
    if k in (ord("q"), ord("Q")):
        return cr, cc, False, "", False, True
    if k in (ord("u"), ord("U")):
        msg = "Undone!" if (game.undo_move() and game.undo_move()) else "Nothing to undo"
        return cr, cc, True, msg, False, False
    if k == curses.KEY_UP:
        return max(0, cr - 1), cc, True, "", False, False
    if k == curses.KEY_DOWN:
        return min(game.size - 1, cr + 1), cc, True, "", False, False
    if k == curses.KEY_LEFT:
        return cr, max(0, cc - 1), True, "", False, False
    if k == curses.KEY_RIGHT:
        return cr, min(game.size - 1, cc + 1), True, "", False, False
    if k == ord(" "):
        if game.board[cr, cc] != EMPTY:
            return cr, cc, True, "Occupied!", False, False
        reward, done = game.make_move(cr, cc)
        if done:
            end = "You win!" if reward == 1 else "Draw!"
            _show_game_end(stdscr, game, cr, cc, human_player, end)
            return cr, cc, False, "", True, False
        return cr, cc, False, "", False, False
    return cr, cc, True, "", False, False


def _handle_ai_turn(stdscr, ai, game, cr, cc, human_player):
    msg = f"AI thinking ({ai.difficulty}, {ai.sims} sims) …"
    draw_board(stdscr, game, cr, cc, human_player, msg)
    row, col, val = ai.get_move(game)
    assert game.board[row, col] == 0, (
        f"ILLEGAL AI MOVE ({row},{col}), "
        f"board={game.board[row, col]}, "
        f"stones={int(np.count_nonzero(game.board))}"
    )
    reward, done = game.make_move(row, col)
    if done:
        end = "AI wins!" if reward == 1 else "Draw!"
        _show_game_end(stdscr, game, cr, cc, human_player, end)
        return "", True
    return f"AI played ({row},{col})  eval {val:+.2f}", False


def _resolve_weights(stdscr, use_latest=False, weight_file=None):
    if weight_file:
        wf = os.path.expanduser(weight_file)
        if not wf.endswith(".h5"):
            _show_text_and_wait(
                stdscr,
                "Invalid --weight_file: expected a .h5 file.",
                f"  Got: {weight_file}",
                "Press any key to exit …",
            )
            return None, None
        if not os.path.isfile(wf):
            _show_text_and_wait(
                stdscr,
                "Specified --weight_file does not exist.",
                f"  Path: {wf}",
                "Press any key to exit …",
            )
            return None, None
        return wf, "explicit"

    mode = "latest" if use_latest else "best"
    wf, label = select_weights(mode=mode)
    if not wf:
        _show_text_and_wait(
            stdscr,
            "No weights found in weights/ — run train.py first.",
            "Press any key to exit …",
        )
        return None, None
    return wf, label


def _prompt_play_again(stdscr):
    stdscr.clear()
    stdscr.addstr(0, 0, "Play again? (y/n): ")
    stdscr.refresh()
    while True:
        k = stdscr.getch()
        if k in (ord("n"), ord("N")):
            return False
        if k in (ord("y"), ord("Y")):
            return True


def play_game_curses(stdscr, ai, game):
    curses.curs_set(0)
    stdscr.nodelay(False)

    cr, cc = game.size // 2, game.size // 2
    human_turn, msg = _prompt_human_first(stdscr)

    human_player = PLAYER1 if human_turn else PLAYER2

    game_over = False
    while not game_over:
        draw_board(stdscr, game, cr, cc, human_player, msg)
        msg = ""

        if human_turn:
            cr, cc, human_turn, msg, game_over, quit_game = _handle_human_turn(
                stdscr, game, cr, cc, human_player
            )
            if quit_game:
                return False
        else:
            msg, game_over = _handle_ai_turn(stdscr, ai, game, cr, cc, human_player)
            if not game_over:
                human_turn = True
    return True


def main(stdscr, use_latest=False, weight_file=None, difficulty="medium"):
    wf, label = _resolve_weights(stdscr, use_latest=use_latest, weight_file=weight_file)
    if not wf:
        return

    stdscr.clear()
    stdscr.addstr(0, 0, "Loading model …")
    stdscr.addstr(1, 0, f"  Weights: {wf} ({label})")
    stdscr.refresh()

    _model, predict_fn = load_model_and_predict_fn(wf)
    try:
        difficulty_label, sims = resolve_difficulty(difficulty)
    except ValueError as e:
        _show_text_and_wait(
            stdscr,
            str(e),
            f"  Got: {difficulty}",
            "Press any key to exit …",
        )
        return
    ai = AIPlayer(predict_fn, simulations=sims, difficulty=difficulty_label)
    stdscr.addstr(2, 0, f"  Difficulty: {difficulty_label} ({sims} sims)")

    stdscr.addstr(3, 0, "  Ready!  Press any key to start …")
    stdscr.refresh(); stdscr.getch()

    while True:
        game = GomokuGame()
        if not play_game_curses(stdscr, ai, game):
            break
        if not _prompt_play_again(stdscr):
            return

    stdscr.clear()
    stdscr.addstr(0, 0, "Thanks for playing!")
    stdscr.refresh(); time.sleep(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Play Gomoku against the AI")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--latest", action="store_true",
                       help="Use latest training weights instead of best checkpoint")
    group.add_argument("--weight_file", type=str, default=None,
                       help="Path to a specific .h5 weights file to load")
    parser.add_argument(
        "--difficulty",
        type=str,
        default="medium",
        help="AI difficulty: easy|medium|hard, or a positive integer sims count (e.g. 2500)",
    )
    args = parser.parse_args()
    curses.wrapper(
        main,
        use_latest=args.latest,
        weight_file=args.weight_file,
        difficulty=args.difficulty,
    )
