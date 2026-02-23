#!/usr/bin/env python3
"""
Gomoku - Play with a Qt GUI.

Includes Human-vs-AI (play.py feature parity) and Human-vs-Human modes using
widgets instead of curses/CLI prompts.
"""

import glob
import os
import sys
import time

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

def _load_pyqt6():
    try:
        from PyQt6 import QtCore as _QtCore, QtGui as _QtGui, QtWidgets as _QtWidgets
    except ImportError:
        print("PyQt6 not found. Install it with: pip install PyQt6", file=sys.stderr)
        sys.exit(1)
    return _QtCore, _QtGui, _QtWidgets


QtCore, QtGui, QtWidgets = _load_pyqt6()

from gomoku import (
    BOARD_SIZE,
    EMPTY,
    NUM_INPUT_PLANES,
    PLAYER1,
    PLAYER2,
    GomokuGame,
    create_model,
    make_predict_fn,
    mcts_begin,
    mcts_expand_root,
    mcts_process_results,
    mcts_policy,
    mcts_select_leaves,
    mcts_search_batched,
)

# How hard the AI thinks (increase for stronger but slower play)
AI_SIMULATIONS = 400
AI_MCTS_BATCH = 8  # small batch = better search quality
CELL_MIN_SIZE = 30
INDEX_COL_WIDTH = 26
DIFFICULTY_SIMS = {
    "easy": 100,
    "medium": 400,
    "hard": 1600,
}

BEST_WEIGHTS = "weights/gomoku_best.weights.h5"
LATEST_WEIGHTS = "weights/gomoku_weights.weights.h5"
ANALYSIS_MAX_SIMS = 1_000_000
ANALYSIS_EMIT_INTERVAL_SEC = 0.2


class AIPlayer:
    """AI that selects moves via MCTS backed by a trained network."""

    def __init__(self, predict_fn, simulations=AI_SIMULATIONS, difficulty="medium"):
        self.predict_fn = predict_fn
        self.sims = simulations
        self.difficulty = difficulty

    def get_move(self, game):
        root = mcts_search_batched(
            game,
            self.predict_fn,
            num_simulations=self.sims,
            batch_size=AI_MCTS_BATCH,
            c_puct=1.5,
            add_noise=False,  # no exploration noise during play
        )
        pi = mcts_policy(root, temperature=0.05)  # near-greedy
        idx = int(np.argmax(pi))
        row, col = divmod(idx, BOARD_SIZE)
        return row, col, root.q_value


class SquareCellButton(QtWidgets.QPushButton):
    """Board cell that keeps a square aspect ratio while expanding."""

    def __init__(self, text="."):
        super().__init__(text)
        policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return width

    def sizeHint(self):
        return QtCore.QSize(CELL_MIN_SIZE, CELL_MIN_SIZE)


class AnalysisWorker(QtCore.QThread):
    """Background pondering worker that continuously expands one MCTS root."""

    analysis_update = QtCore.pyqtSignal(int, object, float, int)
    analysis_error = QtCore.pyqtSignal(int, str)

    def __init__(
        self,
        session_id,
        predict_fn,
        game,
        batch_size=AI_MCTS_BATCH,
        c_puct=1.5,
        max_sims=ANALYSIS_MAX_SIMS,
        emit_interval=ANALYSIS_EMIT_INTERVAL_SEC,
        parent=None,
    ):
        super().__init__(parent)
        self.session_id = int(session_id)
        self.predict_fn = predict_fn
        self.game = game.copy()
        self.batch_size = int(batch_size)
        self.c_puct = float(c_puct)
        self.max_sims = int(max_sims)
        self.emit_interval = float(emit_interval)
        self._running = True

    def request_stop(self):
        self._running = False

    def run(self):
        try:
            ctx, root_state = mcts_begin(
                self.game,
                num_simulations=self.max_sims,
                batch_size=self.batch_size,
                c_puct=self.c_puct,
                add_noise=False,
            )

            root_batch = np.array([root_state], dtype=np.float32)
            logits_np, values_np = self.predict_fn(root_batch)
            mcts_expand_root(ctx, logits_np[0], values_np.ravel()[0])

            last_emit = 0.0
            while self._running and ctx["sims_done"] < ctx["sims_target"]:
                leaf_states = mcts_select_leaves(ctx)
                if leaf_states:
                    batch = np.array(leaf_states, dtype=np.float32)
                    b_logits, b_values = self.predict_fn(batch)
                    mcts_process_results(ctx, b_logits, b_values.ravel())
                else:
                    mcts_process_results(ctx)

                now = time.time()
                if now - last_emit >= self.emit_interval:
                    root = ctx["root"]
                    pi = mcts_policy(root, board_size=BOARD_SIZE, temperature=1.0)
                    self.analysis_update.emit(
                        self.session_id,
                        pi,
                        float(root.q_value),
                        int(ctx["sims_done"]),
                    )
                    last_emit = now

            # Emit one final snapshot if we computed anything.
            root = ctx["root"]
            pi = mcts_policy(root, board_size=BOARD_SIZE, temperature=1.0)
            self.analysis_update.emit(
                self.session_id,
                pi,
                float(root.q_value),
                int(ctx["sims_done"]),
            )
        except Exception as e:
            self.analysis_error.emit(self.session_id, str(e))


def select_weights(mode, explicit_path=""):
    """Return (weights_path, label) from UI selection."""
    mode = (mode or "").strip().lower()

    if mode == "file":
        wf = os.path.expanduser((explicit_path or "").strip())
        if not wf:
            raise ValueError("Select a .h5 file for 'Specific file' mode.")
        if not wf.endswith(".h5"):
            raise ValueError("Invalid weight file: expected a .h5 file.")
        if not os.path.isfile(wf):
            raise ValueError(f"Specified weight file does not exist: {wf}")
        return wf, "explicit"

    if mode == "latest":
        if os.path.exists(LATEST_WEIGHTS):
            return LATEST_WEIGHTS, "latest"
    else:
        if os.path.exists(BEST_WEIGHTS):
            return BEST_WEIGHTS, "best"
        if os.path.exists(LATEST_WEIGHTS):
            return LATEST_WEIGHTS, "latest (no best yet)"

    # Last resort: newest checkpoint file
    files = glob.glob("weights/gomoku_*.weights.h5")
    if files:
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0], "checkpoint"

    raise ValueError("No weights found in weights/. Run train.py first.")


class GomokuQtWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GomokuZero - Qt Play")

        self.model = None
        self.predict_fn = None
        self.ai = None
        self.loaded_model_text = "Not loaded"
        self.human_only_mode = False
        self.analysis_worker = None
        self.analysis_session_id = 0
        self.analysis_policy = None
        self.analysis_root_q = None
        self.analysis_sims_done = 0
        self.game = GomokuGame()
        self.human_player = PLAYER1
        self.human_turn = True
        self.game_over = False

        self.board_buttons = []
        self._build_ui()
        self._auto_load_startup_model()
        self._refresh_board()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.board_buttons:
            self._refresh_board()

    def closeEvent(self, event):
        self._stop_analysis(wait=True, clear=False)
        super().closeEvent(event)

    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        controls = QtWidgets.QGroupBox("Setup")
        form = QtWidgets.QGridLayout(controls)
        root.addWidget(controls)

        self.weight_mode = QtWidgets.QComboBox()
        self.weight_mode.addItem("Best checkpoint", "best")
        self.weight_mode.addItem("Latest training", "latest")
        self.weight_mode.addItem("Specific file", "file")
        self.weight_mode.currentIndexChanged.connect(self._on_weight_mode_changed)
        form.addWidget(QtWidgets.QLabel("Weights"), 0, 0)
        form.addWidget(self.weight_mode, 0, 1)

        self.weight_path = QtWidgets.QLineEdit()
        self.weight_path.setPlaceholderText("weights/my_checkpoint.weights.h5")
        self.weight_path.setEnabled(False)
        form.addWidget(self.weight_path, 0, 2)

        self.browse_btn = QtWidgets.QPushButton("Browse...")
        self.browse_btn.setEnabled(False)
        self.browse_btn.clicked.connect(self._browse_weight_file)
        form.addWidget(self.browse_btn, 0, 3)

        self.diff_combo = QtWidgets.QComboBox()
        self.diff_combo.addItem("easy", "easy")
        self.diff_combo.addItem("medium", "medium")
        self.diff_combo.addItem("hard", "hard")
        self.diff_combo.addItem("Custom", "custom")
        self.diff_combo.setCurrentIndex(1)
        self.diff_combo.currentIndexChanged.connect(self._on_difficulty_changed)
        form.addWidget(QtWidgets.QLabel("Difficulty"), 1, 0)
        form.addWidget(self.diff_combo, 1, 1)

        self.custom_sims = QtWidgets.QSpinBox()
        self.custom_sims.setRange(1, 20000)
        self.custom_sims.setValue(AI_SIMULATIONS)
        self.custom_sims.setEnabled(False)
        self.custom_sims.valueChanged.connect(self._on_custom_sims_changed)
        form.addWidget(self.custom_sims, 1, 2)
        form.addWidget(QtWidgets.QLabel("sims"), 1, 3)

        self.mode_btn = QtWidgets.QPushButton("Switch to Human vs Human")
        self.mode_btn.clicked.connect(self._toggle_mode)
        form.addWidget(self.mode_btn, 2, 0, 1, 2)

        self.analysis_check = QtWidgets.QCheckBox("Analysis heatmap")
        self.analysis_check.setChecked(False)
        self.analysis_check.toggled.connect(self._on_analysis_toggled)
        form.addWidget(self.analysis_check, 2, 2, 1, 2)

        self.side_combo = QtWidgets.QComboBox()
        self.side_combo.addItem("Human first (X)", PLAYER1)
        self.side_combo.addItem("AI first (X)", PLAYER2)
        self.side_combo.currentIndexChanged.connect(self._on_side_changed)
        form.addWidget(QtWidgets.QLabel("Side"), 3, 0)
        form.addWidget(self.side_combo, 3, 1)

        btn_row = QtWidgets.QHBoxLayout()
        self.load_btn = QtWidgets.QPushButton("Load Model")
        self.load_btn.clicked.connect(self.load_model)
        btn_row.addWidget(self.load_btn)

        self.new_game_btn = QtWidgets.QPushButton("New Game")
        self.new_game_btn.clicked.connect(self.new_game)
        self.new_game_btn.setEnabled(False)
        btn_row.addWidget(self.new_game_btn)

        self.undo_btn = QtWidgets.QPushButton("Undo")
        self.undo_btn.clicked.connect(self.undo_last_pair)
        self.undo_btn.setEnabled(False)
        btn_row.addWidget(self.undo_btn)

        quit_btn = QtWidgets.QPushButton("Quit")
        quit_btn.clicked.connect(self.close)
        btn_row.addWidget(quit_btn)

        form.addLayout(btn_row, 4, 0, 1, 4)

        info = QtWidgets.QGroupBox("Status")
        info_layout = QtWidgets.QGridLayout(info)
        root.addWidget(info)

        info_layout.addWidget(QtWidgets.QLabel("Model"), 0, 0)
        self.model_label = QtWidgets.QLabel("Not loaded")
        info_layout.addWidget(self.model_label, 0, 1)

        info_layout.addWidget(QtWidgets.QLabel("Mode"), 1, 0)
        self.mode_label = QtWidgets.QLabel("-")
        info_layout.addWidget(self.mode_label, 1, 1)

        info_layout.addWidget(QtWidgets.QLabel("Difficulty"), 2, 0)
        self.difficulty_label = QtWidgets.QLabel("-")
        info_layout.addWidget(self.difficulty_label, 2, 1)

        info_layout.addWidget(QtWidgets.QLabel("Side"), 3, 0)
        self.side_label = QtWidgets.QLabel("-")
        info_layout.addWidget(self.side_label, 3, 1)

        info_layout.addWidget(QtWidgets.QLabel("Turn"), 4, 0)
        self.turn_label = QtWidgets.QLabel("-")
        info_layout.addWidget(self.turn_label, 4, 1)

        info_layout.addWidget(QtWidgets.QLabel("Moves"), 5, 0)
        self.moves_label = QtWidgets.QLabel("0")
        info_layout.addWidget(self.moves_label, 5, 1)

        info_layout.addWidget(QtWidgets.QLabel("Message"), 6, 0)
        self.message_label = QtWidgets.QLabel("Load model to begin.")
        self.message_label.setWordWrap(True)
        info_layout.addWidget(self.message_label, 6, 1)

        info_layout.addWidget(QtWidgets.QLabel("Analysis"), 7, 0)
        self.analysis_label = QtWidgets.QLabel("Off")
        info_layout.addWidget(self.analysis_label, 7, 1)

        board_frame = QtWidgets.QGroupBox("Board")
        board_layout = QtWidgets.QGridLayout(board_frame)
        board_layout.setHorizontalSpacing(1)
        board_layout.setVerticalSpacing(1)
        board_layout.setContentsMargins(6, 6, 6, 6)
        root.addWidget(board_frame, 1)

        board_layout.setColumnMinimumWidth(0, INDEX_COL_WIDTH)
        for c in range(BOARD_SIZE):
            lbl = QtWidgets.QLabel(f"{c:02d}")
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            lbl.setMinimumSize(CELL_MIN_SIZE, CELL_MIN_SIZE)
            lbl.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            board_layout.addWidget(lbl, 0, c + 1)
            board_layout.setColumnMinimumWidth(c + 1, CELL_MIN_SIZE)
            board_layout.setColumnStretch(c + 1, 1)

        mono = QtGui.QFont("Monospace")
        mono.setStyleHint(QtGui.QFont.StyleHint.TypeWriter)

        for r in range(BOARD_SIZE):
            row_lbl = QtWidgets.QLabel(f"{r:02d}")
            row_lbl.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            row_lbl.setFixedSize(INDEX_COL_WIDTH, CELL_MIN_SIZE)
            board_layout.addWidget(row_lbl, r + 1, 0)
            board_layout.setRowMinimumHeight(r + 1, CELL_MIN_SIZE)
            board_layout.setRowStretch(r + 1, 1)

            row_btns = []
            for c in range(BOARD_SIZE):
                btn = SquareCellButton(".")
                btn.setMinimumSize(CELL_MIN_SIZE, CELL_MIN_SIZE)
                btn.setFont(mono)
                btn.clicked.connect(
                    lambda _checked=False, rr=r, cc=c: self.on_cell_clicked(rr, cc)
                )
                board_layout.addWidget(btn, r + 1, c + 1)
                row_btns.append(btn)
            self.board_buttons.append(row_btns)

        self.resize(860, 900)
        self._update_status_box()
        self._apply_mode_state()

    def _selected_difficulty(self):
        mode = self.diff_combo.currentData()
        if mode in DIFFICULTY_SIMS:
            return mode, DIFFICULTY_SIMS[mode]
        return "Custom", int(self.custom_sims.value())

    def _is_human_only(self):
        return self.human_only_mode

    def _selected_side_text(self):
        if self._is_human_only():
            return "Human: X vs Human: O (X first)"
        # In active H-vs-A games, derive from the actual assigned side.
        human = self.human_player if self.ai is not None else int(self.side_combo.currentData())
        if human == PLAYER1:
            return "Human: X (first), AI: O"
        return "Human: O, AI: X (first)"

    def _update_status_box(self):
        self.mode_label.setText(
            "Human vs Human" if self._is_human_only() else "Human vs AI"
        )
        if self._is_human_only():
            self.mode_btn.setText("Switch to Human vs AI (AI move)")
        else:
            self.mode_btn.setText("Switch to Human vs Human")
        diff_label, sims = self._selected_difficulty()
        if self._is_human_only():
            self.difficulty_label.setText("N/A (human only)")
        else:
            self.difficulty_label.setText(f"{diff_label} ({sims} sims)")
        self.side_label.setText(self._selected_side_text())
        if not self.analysis_check.isChecked():
            self.analysis_label.setText("Off")
        elif self._is_human_only():
            self.analysis_label.setText("Unavailable in Human vs Human")
        elif self.ai is None:
            self.analysis_label.setText("Waiting for model")
        elif self.game_over:
            self.analysis_label.setText("Paused (game over)")
        elif not self.human_turn:
            self.analysis_label.setText("Paused (AI turn)")
        elif self.analysis_policy is None:
            self.analysis_label.setText("Starting...")
        else:
            self.analysis_label.setText(
                f"Running: {self.analysis_sims_done} sims, root {self.analysis_root_q:+.2f}"
            )

    def _set_message(self, msg):
        self.message_label.setText(msg)

    def _on_weight_mode_changed(self):
        is_file = (not self._is_human_only()) and self.weight_mode.currentData() == "file"
        self.weight_path.setEnabled(is_file)
        self.browse_btn.setEnabled(is_file)

    def _on_difficulty_changed(self):
        if self._is_human_only():
            self.custom_sims.setEnabled(False)
            self._update_status_box()
            return
        is_custom = self.diff_combo.currentData() == "custom"
        self.custom_sims.setEnabled(is_custom)
        self._update_status_box()
        if self.ai is not None:
            label, sims = self._selected_difficulty()
            self.ai.difficulty = label
            self.ai.sims = sims
            self._set_message(f"Difficulty set to {label} ({sims} sims).")
            self._refresh_board()

    def _on_custom_sims_changed(self):
        if self._is_human_only():
            return
        if self.diff_combo.currentData() != "custom":
            return
        self._update_status_box()
        if self.ai is not None:
            label, sims = self._selected_difficulty()
            self.ai.difficulty = label
            self.ai.sims = sims
            self._set_message(f"Difficulty set to {label} ({sims} sims).")
            self._refresh_board()

    def _on_side_changed(self):
        self._update_status_box()

    def _on_analysis_toggled(self):
        self._restart_analysis_if_needed()
        self._update_status_box()

    def _apply_mode_state(self):
        human_only = self._is_human_only()

        self.weight_mode.setEnabled(not human_only)
        self.diff_combo.setEnabled(not human_only)
        self.side_combo.setEnabled(not human_only)
        self.load_btn.setEnabled(not human_only)
        self.analysis_check.setEnabled(not human_only)

        if human_only:
            self._stop_analysis(wait=False, clear=True)
            self.weight_path.setEnabled(False)
            self.browse_btn.setEnabled(False)
            self.custom_sims.setEnabled(False)
            self.new_game_btn.setEnabled(True)
            self.undo_btn.setEnabled(True)
            self.model_label.setText("Not used (Human vs Human mode)")
            self._set_message("Human vs Human mode selected. Click New Game.")
        else:
            self._on_weight_mode_changed()
            self._on_difficulty_changed()
            self.new_game_btn.setEnabled(self.ai is not None)
            self.undo_btn.setEnabled(self.ai is not None)
            if self.ai is None:
                self.model_label.setText("Not loaded")
                self._set_message("Load model to begin.")
            else:
                self.model_label.setText(self.loaded_model_text)

        self._update_status_box()
        self._refresh_board()
        self._restart_analysis_if_needed()

    def _toggle_mode(self):
        if self._is_human_only():
            if self.ai is None:
                self._show_error("Load a model first, then switch to Human vs AI.")
                return
            self.human_only_mode = False
            # Force AI to move immediately from the current position.
            self.human_player = -self.game.current_player
            self.side_combo.setCurrentIndex(0 if self.human_player == PLAYER1 else 1)
            self.human_turn = False
            self._apply_mode_state()
            if not self.game_over:
                self._begin_ai_turn()
            else:
                self._set_message("Switched to Human vs AI.")
        else:
            self.human_only_mode = True
            self.human_turn = True
            self._apply_mode_state()
            self._set_message("Switched to Human vs Human.")

    def _browse_weight_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select weights file",
            "",
            "H5 files (*.h5);;All files (*)",
        )
        if path:
            self.weight_mode.setCurrentIndex(2)  # Specific file
            self.weight_path.setText(path)

    def _show_error(self, text):
        QtWidgets.QMessageBox.critical(self, "Error", text)

    def _load_model_from_selection(
        self,
        mode,
        explicit_path="",
        show_errors=True,
        startup=False,
    ):
        try:
            weight_file, label = select_weights(mode, explicit_path)
        except ValueError as e:
            if show_errors:
                self._show_error(str(e))
            else:
                self._set_message(f"Startup auto-load skipped: {e}")
            return False

        difficulty_label, sims = self._selected_difficulty()
        self._set_message("Loading model ...")
        QtWidgets.QApplication.processEvents()

        try:
            model = create_model()
            model.load_weights(weight_file)
            predict_fn = make_predict_fn(model)
            predict_fn(
                np.zeros(
                    (1, BOARD_SIZE, BOARD_SIZE, NUM_INPUT_PLANES),
                    dtype=np.float32,
                )
            )
        except Exception as e:
            if show_errors:
                self._show_error(f"Failed to load model:\n{e}")
            else:
                self._set_message(f"Startup auto-load failed: {e}")
            return False

        self.model = model
        self.predict_fn = predict_fn
        self.ai = AIPlayer(
            predict_fn,
            simulations=sims,
            difficulty=difficulty_label,
        )

        self.loaded_model_text = f"{weight_file} ({label})"
        self.model_label.setText(self.loaded_model_text)
        self.new_game_btn.setEnabled(True)
        self.undo_btn.setEnabled(True)
        if startup:
            self._set_message(
                f"Auto-loaded {label} weights. Difficulty: "
                f"{difficulty_label} ({sims} sims)."
            )
        else:
            self._set_message(
                f"Model loaded. Difficulty: {difficulty_label} ({sims} sims). Start a new game."
            )
        self._refresh_board()
        self._restart_analysis_if_needed()
        return True

    def _auto_load_startup_model(self):
        if self._is_human_only():
            return
        self._load_model_from_selection(
            mode="best",
            explicit_path="",
            show_errors=False,
            startup=True,
        )

    def load_model(self):
        if self._is_human_only():
            self._show_error("Model loading is disabled in Human vs Human mode.")
            return
        self._load_model_from_selection(
            mode=self.weight_mode.currentData(),
            explicit_path=self.weight_path.text(),
            show_errors=True,
            startup=False,
        )

    def new_game(self):
        if self._is_human_only():
            self._stop_analysis(wait=False, clear=True)
            self.game = GomokuGame()
            self.human_player = PLAYER1
            self.human_turn = True
            self.game_over = False
            self._set_message("New game started (Human vs Human).")
            self._refresh_board()
            return

        if self.ai is None:
            self._show_error("Load a model first.")
            return

        self.game = GomokuGame()
        self.human_player = int(self.side_combo.currentData())
        self.human_turn = (self.game.current_player == self.human_player)
        self.game_over = False
        self._stop_analysis(wait=False, clear=True)
        self._set_message("New game started.")
        self._refresh_board()

        if not self.human_turn:
            self._begin_ai_turn()
        else:
            self._restart_analysis_if_needed()

    def undo_last_pair(self):
        if self._is_human_only():
            if self.game.undo_move():
                self.game_over = False
                self.human_turn = True
                self._set_message("Undone.")
                self._refresh_board()
                self._restart_analysis_if_needed()
            else:
                self._set_message("Nothing to undo.")
                self._refresh_board()
            return

        if self.ai is None:
            self._show_error("Load a model first.")
            return

        if self.game.undo_move() and self.game.undo_move():
            self.game_over = False
            self.human_turn = (self.game.current_player == self.human_player)
            self._set_message("Undone.")
            self._refresh_board()
            if not self.human_turn:
                self._begin_ai_turn()
            else:
                self._restart_analysis_if_needed()
        else:
            self._set_message("Nothing to undo.")
            self._refresh_board()

    def on_cell_clicked(self, row, col):
        if (not self._is_human_only()) and self.ai is None:
            self._show_error("Load a model first.")
            return
        if self.game_over:
            return
        if (not self._is_human_only()) and (not self.human_turn):
            return
        if self.game.board[row, col] != EMPTY:
            self._set_message("Occupied!")
            return

        reward, done = self.game.make_move(row, col)
        if done:
            self.game_over = True
            if reward == 1 and self._is_human_only():
                winner = "X" if self.game.current_player == PLAYER1 else "O"
                self._set_message(f"{winner} wins! Press New Game to play again.")
            elif reward == 1:
                self._set_message("You win! Press New Game to play again.")
            else:
                self._set_message("Draw! Press New Game to play again.")
            self._refresh_board()
            self._restart_analysis_if_needed()
            return

        if self._is_human_only():
            self.human_turn = True
            next_side = "X" if self.game.current_player == PLAYER1 else "O"
            self._set_message(f"Move at ({row},{col}). Human {next_side} to play.")
            self._refresh_board()
            self._restart_analysis_if_needed()
            return

        self.human_turn = False
        self._set_message(f"You played ({row},{col}).")
        self._refresh_board()
        self._begin_ai_turn()

    def _begin_ai_turn(self):
        self._stop_analysis(wait=True, clear=True)
        if self._is_human_only():
            return
        if self.ai is None or self.game_over:
            return
        self.human_turn = False
        self._set_message(f"AI thinking ({self.ai.difficulty}, {self.ai.sims} sims) ...")
        self._refresh_board()
        # Flush paint events so the human move is visible before AI search blocks UI.
        QtWidgets.QApplication.processEvents(
            QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
        )
        QtCore.QTimer.singleShot(0, self._run_ai_turn)

    def _run_ai_turn(self):
        if self._is_human_only():
            return
        if self.ai is None or self.game_over:
            return
        try:
            row, col, val = self.ai.get_move(self.game)
        except Exception as e:
            self._show_error(f"AI move failed:\n{e}")
            return

        if self.game.board[row, col] != EMPTY:
            self._show_error(f"Illegal AI move ({row},{col}).")
            return

        reward, done = self.game.make_move(row, col)
        if done:
            self.game_over = True
            if reward == 1:
                end = "AI wins!"
            else:
                end = "Draw!"
            self._set_message(f"{end} AI played ({row},{col}) eval {val:+.2f}")
        else:
            self.human_turn = True
            self._set_message(f"AI played ({row},{col}) eval {val:+.2f}")
        self._refresh_board()
        self._restart_analysis_if_needed()

    def _analysis_should_run(self):
        return (
            self.analysis_check.isChecked()
            and (not self._is_human_only())
            and (self.ai is not None)
            and (not self.game_over)
            and self.human_turn
        )

    def _start_analysis(self):
        if not self._analysis_should_run():
            return
        if self.analysis_worker is not None:
            return
        self.analysis_session_id += 1
        self.analysis_policy = None
        self.analysis_root_q = 0.0
        self.analysis_sims_done = 0
        self.analysis_worker = AnalysisWorker(
            session_id=self.analysis_session_id,
            predict_fn=self.predict_fn,
            game=self.game,
            batch_size=AI_MCTS_BATCH,
            c_puct=1.5,
        )
        self.analysis_worker.analysis_update.connect(self._on_analysis_update)
        self.analysis_worker.analysis_error.connect(self._on_analysis_error)
        self.analysis_worker.finished.connect(self._on_analysis_finished)
        self.analysis_worker.start()
        self._update_status_box()

    def _stop_analysis(self, wait=False, clear=False):
        worker = self.analysis_worker
        if worker is not None:
            worker.request_stop()
            if wait:
                worker.wait()
        if clear:
            self.analysis_policy = None
            self.analysis_root_q = None
            self.analysis_sims_done = 0
            self.analysis_session_id += 1
        self._update_status_box()

    def _restart_analysis_if_needed(self):
        should_run = self._analysis_should_run()
        if should_run:
            # Restart only if no worker is active.
            if self.analysis_worker is None:
                self._start_analysis()
        else:
            self._stop_analysis(wait=False, clear=True)

    def _on_analysis_update(self, session_id, policy, root_q, sims_done):
        if int(session_id) != int(self.analysis_session_id):
            return
        self.analysis_policy = np.asarray(policy, dtype=np.float32).ravel()
        self.analysis_root_q = float(root_q)
        self.analysis_sims_done = int(sims_done)
        self._update_status_box()
        self._refresh_board()

    def _on_analysis_error(self, session_id, err_text):
        if int(session_id) != int(self.analysis_session_id):
            return
        self.analysis_label.setText(f"Error: {err_text}")
        self.analysis_policy = None
        self.analysis_root_q = None
        self.analysis_sims_done = 0
        self._stop_analysis(wait=False, clear=False)

    def _on_analysis_finished(self):
        # Worker can finish naturally or after stop request.
        self.analysis_worker = None
        self._update_status_box()

    def _refresh_board(self):
        glyph = {EMPTY: ".", PLAYER1: "X", PLAYER2: "O"}
        last_move = None
        if self.game.move_history:
            last = self.game.move_history[-1]
            last_move = (int(last[0]), int(last[1]))

        analysis_pi = None
        analysis_max = 0.0
        if (self.analysis_policy is not None
                and self._analysis_should_run()
                and len(self.analysis_policy) == BOARD_SIZE * BOARD_SIZE):
            analysis_pi = self.analysis_policy
            legal_mask = (self.game.board.reshape(-1) == EMPTY)
            if np.any(legal_mask):
                analysis_max = float(np.max(analysis_pi[legal_mask]))

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                val = int(self.game.board[r, c])
                btn = self.board_buttons[r][c]
                btn.setText(glyph[val])

                # Scale marker size with current square size.
                cell_px = max(1, min(btn.width(), btn.height()))
                if val == EMPTY:
                    font_px = max(8, int(cell_px * 0.30))
                    color = "#808080"
                    weight = 500
                elif val == PLAYER1:
                    font_px = max(10, int(cell_px * 0.62))
                    color = "#111111"
                    weight = 700
                else:
                    font_px = max(10, int(cell_px * 0.62))
                    color = "#0055aa"
                    weight = 700
                f = btn.font()
                f.setPixelSize(font_px)
                btn.setFont(f)

                is_last_move = (last_move is not None and last_move == (r, c))
                if is_last_move:
                    bg = "#ffe9a8"
                    border = "2px solid #d49100"
                elif val == EMPTY and analysis_pi is not None and analysis_max > 0.0:
                    p = float(analysis_pi[r * BOARD_SIZE + c])
                    t = max(0.0, min(1.0, p / analysis_max))
                    rr = int(245 - 125 * t)
                    gg = int(245 - 25 * t)
                    bb = int(245 - 125 * t)
                    bg = f"rgb({rr},{gg},{bb})"
                    border = "2px solid #3b7d3b" if t >= 0.92 else "1px solid #98c598"
                else:
                    bg = "#f5f5f5"
                    border = "1px solid #b8b8b8"
                btn.setStyleSheet(
                    "QPushButton {"
                    f"color: {color};"
                    f"font-weight: {weight};"
                    f"background-color: {bg};"
                    f"border: {border};"
                    "padding: 0px;"
                    "}"
                    "QPushButton:disabled {"
                    f"color: {color};"
                    f"background-color: {bg};"
                    f"border: {border};"
                    "}"
                )
                if val == EMPTY and analysis_pi is not None:
                    p = float(analysis_pi[r * BOARD_SIZE + c])
                    btn.setToolTip(f"MCTS policy: {100.0 * p:.2f}%")
                else:
                    btn.setToolTip("")
                enabled = (
                    (not self.game_over)
                    and (val == EMPTY)
                    and (
                        self._is_human_only()
                        or (
                            self.ai is not None
                            and self.human_turn
                        )
                    )
                )
                btn.setEnabled(enabled)

        if self.game_over:
            turn_text = "Game over"
        elif self._is_human_only():
            side = "X" if self.game.current_player == PLAYER1 else "O"
            turn_text = f"Human turn ({side})"
        else:
            side = "X" if self.game.current_player == PLAYER1 else "O"
            actor = "Your turn" if self.game.current_player == self.human_player else "AI turn"
            turn_text = f"{actor} ({side})"
        self.turn_label.setText(turn_text)
        self.moves_label.setText(str(len(self.game.move_history)))
        self._update_status_box()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = GomokuQtWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
