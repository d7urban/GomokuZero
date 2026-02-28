#!/usr/bin/env python3
"""
Gomoku - Play with a Qt GUI.

Includes Human-vs-AI (play.py feature parity) and Human-vs-Human modes using
widgets instead of curses/CLI prompts.
"""

import os
import sys
import time
import json

import numpy as np

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
    PLAYER1,
    PLAYER2,
    GomokuGame,
    mcts_begin,
    mcts_expand_root,
    mcts_process_results,
    mcts_policy,
    mcts_select_leaves,
)
from entrypoint_shared import (
    AIPlayer,
    AI_MCTS_BATCH,
    AI_SIMULATIONS,
    load_model_and_predict_fn,
    resolve_difficulty,
    select_weights as select_shared_weights,
)

CELL_MIN_SIZE = 30
INDEX_COL_WIDTH = 26
ANALYSIS_MAX_SIMS = 1_000_000
ANALYSIS_EMIT_INTERVAL_SEC = 0.2


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

    # Use a single Python-object payload to avoid brittle C++ signature
    # resolution across PyQt/SIP versions when emitting from worker threads.
    analysis_update = QtCore.pyqtSignal(object)
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
                        (
                            self.session_id,
                            pi,
                            float(root.q_value),
                            int(ctx["sims_done"]),
                        )
                    )
                    last_emit = now

            # Emit one final snapshot if we computed anything.
            root = ctx["root"]
            pi = mcts_policy(root, board_size=BOARD_SIZE, temperature=1.0)
            self.analysis_update.emit(
                (
                    self.session_id,
                    pi,
                    float(root.q_value),
                    int(ctx["sims_done"]),
                )
            )
        except Exception as e:
            self.analysis_error.emit(self.session_id, str(e))


def select_weights(mode, explicit_path=""):
    """Return (weights_path, label) from UI selection."""
    weight_file, label = select_shared_weights(mode=mode, explicit_path=explicit_path)
    if not weight_file:
        raise ValueError("No weights found in weights/. Run train.py first.")
    return weight_file, label


class GomokuQtWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GomokuZero - Qt Play")

        self.model = None
        self.predict_fn = None
        self.ai = None
        self.loaded_model_text = "Not loaded"
        self.loaded_weight_file = ""
        self.human_only_mode = False
        self.analysis_worker = None
        self.analysis_session_id = 0
        self.analysis_policy = None
        self.analysis_root_q = None
        self.analysis_sims_done = 0
        self.analysis_started_at = 0.0
        self.ai_turn_request_id = 0
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

        self._build_setup_controls(root)
        self._build_status_panel(root)
        self._build_board_panel(root)

        self.resize(860, 900)
        self._update_status_box()
        self._apply_mode_state()

    def _build_setup_controls(self, root_layout):
        controls = QtWidgets.QGroupBox("Setup")
        form = QtWidgets.QGridLayout(controls)
        root_layout.addWidget(controls)

        self._build_weight_controls(form)
        self._build_difficulty_controls(form)
        self._build_mode_controls(form)
        self._build_action_buttons(form)

    def _build_weight_controls(self, form):
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

    def _build_difficulty_controls(self, form):
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

    def _build_mode_controls(self, form):
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

    def _build_action_buttons(self, form):
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

        self.save_game_btn = QtWidgets.QPushButton("Save Game")
        self.save_game_btn.clicked.connect(self.save_game)
        btn_row.addWidget(self.save_game_btn)

        self.load_game_btn = QtWidgets.QPushButton("Load Game")
        self.load_game_btn.clicked.connect(self.load_game)
        btn_row.addWidget(self.load_game_btn)

        quit_btn = QtWidgets.QPushButton("Quit")
        quit_btn.clicked.connect(self.close)
        btn_row.addWidget(quit_btn)

        form.addLayout(btn_row, 4, 0, 1, 4)

    def _build_status_panel(self, root_layout):
        info = QtWidgets.QGroupBox("Status")
        layout = QtWidgets.QGridLayout(info)
        root_layout.addWidget(info)

        self.model_label = self._add_status_row(layout, 0, "Model", "Not loaded")
        self.mode_label = self._add_status_row(layout, 1, "Mode", "-")
        self.difficulty_label = self._add_status_row(layout, 2, "Difficulty", "-")
        self.side_label = self._add_status_row(layout, 3, "Side", "-")
        self.turn_label = self._add_status_row(layout, 4, "Turn", "-")
        self.moves_label = self._add_status_row(layout, 5, "Moves", "0")
        self.message_label = self._add_status_row(
            layout, 6, "Message", "Load model to begin.", word_wrap=True
        )
        self.analysis_label = self._add_status_row(layout, 7, "Analysis", "Off")
        self.analysis_sps_label = self._add_status_row(layout, 8, "Sims/sec", "-")

    def _add_status_row(self, layout, row, label, value, word_wrap=False):
        layout.addWidget(QtWidgets.QLabel(label), row, 0)
        value_label = QtWidgets.QLabel(value)
        if word_wrap:
            value_label.setWordWrap(True)
        layout.addWidget(value_label, row, 1)
        return value_label

    def _build_board_panel(self, root_layout):
        board_frame = QtWidgets.QGroupBox("Board")
        board_layout = QtWidgets.QGridLayout(board_frame)
        board_layout.setHorizontalSpacing(1)
        board_layout.setVerticalSpacing(1)
        board_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.addWidget(board_frame, 1)

        self._add_board_headers(board_layout)
        mono = QtGui.QFont("Monospace")
        mono.setStyleHint(QtGui.QFont.StyleHint.TypeWriter)
        for r in range(BOARD_SIZE):
            self._add_board_row(board_layout, r, mono)

    def _add_board_headers(self, board_layout):
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

    def _add_board_row(self, board_layout, row, mono_font):
        row_lbl = QtWidgets.QLabel(f"{row:02d}")
        row_lbl.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight
            | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        row_lbl.setFixedSize(INDEX_COL_WIDTH, CELL_MIN_SIZE)
        board_layout.addWidget(row_lbl, row + 1, 0)
        board_layout.setRowMinimumHeight(row + 1, CELL_MIN_SIZE)
        board_layout.setRowStretch(row + 1, 1)

        row_btns = []
        for c in range(BOARD_SIZE):
            btn = SquareCellButton(".")
            btn.setMinimumSize(CELL_MIN_SIZE, CELL_MIN_SIZE)
            btn.setFont(mono_font)
            btn.clicked.connect(
                lambda _checked=False, rr=row, cc=c: self.on_cell_clicked(rr, cc)
            )
            board_layout.addWidget(btn, row + 1, c + 1)
            row_btns.append(btn)
        self.board_buttons.append(row_btns)

    def _selected_difficulty(self):
        mode = self.diff_combo.currentData()
        if mode == "custom":
            return resolve_difficulty(str(int(self.custom_sims.value())))
        return resolve_difficulty(mode)

    @staticmethod
    def _set_combo_data(combo, value):
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)
            return True
        return False

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
            self.analysis_sps_label.setText("-")
        elif self.predict_fn is None:
            self.analysis_label.setText("Waiting for model")
            self.analysis_sps_label.setText("-")
        elif self.game_over:
            self.analysis_label.setText("Paused (game over)")
            self.analysis_sps_label.setText("-")
        elif not self.human_turn:
            self.analysis_label.setText("Paused (AI turn)")
            self.analysis_sps_label.setText("-")
        elif self.analysis_policy is None:
            self.analysis_label.setText("Starting...")
            self.analysis_sps_label.setText("-")
        else:
            self.analysis_label.setText(
                f"Running: {self.analysis_sims_done} sims, root {self.analysis_root_q:+.2f}"
            )
            elapsed = max(0.0, time.time() - float(self.analysis_started_at))
            if elapsed > 0.0 and self.analysis_sims_done > 0:
                sps = float(self.analysis_sims_done) / elapsed
                self.analysis_sps_label.setText(f"{sps:.1f}")
            else:
                self.analysis_sps_label.setText("-")

    def _set_message(self, msg):
        self.message_label.setText(msg)

    def _cancel_pending_ai_turn(self):
        # Invalidate any queued _run_ai_turn callback created via QTimer.singleShot.
        self.ai_turn_request_id += 1

    def _on_weight_mode_changed(self):
        is_file = self.weight_mode.currentData() == "file"
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

        self.weight_mode.setEnabled(True)
        self.diff_combo.setEnabled(not human_only)
        self.side_combo.setEnabled(not human_only)
        self.load_btn.setEnabled(True)
        self.analysis_check.setEnabled(True)

        if human_only:
            self._on_weight_mode_changed()
            self.custom_sims.setEnabled(False)
            self.new_game_btn.setEnabled(True)
            self.undo_btn.setEnabled(True)
            if self.predict_fn is None:
                self.model_label.setText("Not loaded")
                self._set_message(
                    "Human vs Human mode selected. Load model for analysis or click New Game."
                )
            else:
                self.model_label.setText(self.loaded_model_text)
                self._set_message("Human vs Human mode selected. Analysis available.")
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
            self._cancel_pending_ai_turn()
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
            model, predict_fn = load_model_and_predict_fn(weight_file)
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
        self.loaded_weight_file = weight_file
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
        self._load_model_from_selection(
            mode=self.weight_mode.currentData(),
            explicit_path=self.weight_path.text(),
            show_errors=True,
            startup=False,
        )

    def _snapshot_dict(self):
        return {
            "format": "gomokuzero-qt-save",
            "version": 1,
            "board_size": int(BOARD_SIZE),
            "saved_at": float(time.time()),
            "human_only_mode": bool(self.human_only_mode),
            "human_player": int(self.human_player),
            "game_over": bool(self.game_over),
            "move_history": [
                [int(r), int(c), int(p)] for r, c, p in self.game.move_history
            ],
            "weight_mode": str(self.weight_mode.currentData()),
            "weight_path": str(self.weight_path.text()).strip(),
            "loaded_weight_file": str(self.loaded_weight_file).strip(),
            "difficulty_mode": str(self.diff_combo.currentData()),
            "custom_sims": int(self.custom_sims.value()),
            "side_combo_player": int(self.side_combo.currentData()),
            "analysis_enabled": bool(self.analysis_check.isChecked()),
        }

    @staticmethod
    def _normalize_save_path(path):
        low = path.lower()
        if low.endswith(".json"):
            return path
        return f"{path}.json"

    @staticmethod
    def _restore_game_from_history(raw_history):
        if not isinstance(raw_history, list):
            raise ValueError("Invalid save: move_history must be a list.")
        game = GomokuGame()
        game_over = False
        for i, move in enumerate(raw_history):
            if not (isinstance(move, (list, tuple)) and len(move) == 3):
                raise ValueError("Invalid save: bad move_history entry.")
            try:
                row = int(move[0])
                col = int(move[1])
                player = int(move[2])
            except Exception as e:
                raise ValueError("Invalid save: non-integer move entry.") from e
            if player not in (PLAYER1, PLAYER2):
                raise ValueError("Invalid save: move player must be X/O.")
            if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
                raise ValueError("Invalid save: move out of board bounds.")
            if int(game.current_player) != player:
                raise ValueError("Invalid save: move order is inconsistent.")
            reward, done = game.make_move(row, col)
            if reward == -1:
                raise ValueError("Invalid save: illegal move in move history.")
            if done and i != len(raw_history) - 1:
                raise ValueError("Invalid save: extra moves after game over.")
            if done:
                game_over = True
        return game, game_over

    def save_game(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save game",
            "",
            "Gomoku save (*.json);;All files (*)",
        )
        if not path:
            return
        path = self._normalize_save_path(path)
        try:
            payload = self._snapshot_dict()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
        except Exception as e:
            self._show_error(f"Failed to save game:\n{e}")
            return
        self._set_message(f"Game saved to {path}")

    def load_game(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load game",
            "",
            "Gomoku save (*.json);;All files (*)",
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                raise ValueError("Invalid save: expected a JSON object.")
            if payload.get("format") != "gomokuzero-qt-save":
                raise ValueError("Invalid save: unsupported file format.")
            version = int(payload.get("version", 0))
            if version != 1:
                raise ValueError(f"Invalid save: unsupported version {version}.")
            save_board_size = int(payload.get("board_size", 0))
            if save_board_size != BOARD_SIZE:
                raise ValueError(
                    f"Invalid save: board size {save_board_size} != {BOARD_SIZE}."
                )

            restored_game, game_over = self._restore_game_from_history(
                payload.get("move_history", [])
            )
            requested_mode = bool(payload.get("human_only_mode", False))
            requested_player = int(payload.get("human_player", PLAYER1))
            if requested_player not in (PLAYER1, PLAYER2):
                raise ValueError("Invalid save: human_player must be X or O.")

            requested_weight_mode = str(payload.get("weight_mode", "best"))
            requested_weight_path = str(payload.get("weight_path", ""))
            requested_weight_file = str(payload.get("loaded_weight_file", "")).strip()
            requested_diff_mode = str(payload.get("difficulty_mode", "medium"))
            requested_custom_sims = int(payload.get("custom_sims", AI_SIMULATIONS))
            requested_side_player = int(
                payload.get("side_combo_player", requested_player)
            )
            requested_analysis = bool(payload.get("analysis_enabled", False))
        except Exception as e:
            self._show_error(f"Failed to load game:\n{e}")
            return

        self._cancel_pending_ai_turn()
        self._stop_analysis(wait=True, clear=True)

        blockers = [
            QtCore.QSignalBlocker(self.weight_mode),
            QtCore.QSignalBlocker(self.weight_path),
            QtCore.QSignalBlocker(self.diff_combo),
            QtCore.QSignalBlocker(self.custom_sims),
            QtCore.QSignalBlocker(self.side_combo),
            QtCore.QSignalBlocker(self.analysis_check),
        ]
        _ = blockers
        self._set_combo_data(self.weight_mode, requested_weight_mode)
        self.weight_path.setText(requested_weight_path)
        self._set_combo_data(self.diff_combo, requested_diff_mode)
        self.custom_sims.setValue(max(1, min(20000, requested_custom_sims)))
        if requested_side_player not in (PLAYER1, PLAYER2):
            requested_side_player = requested_player
        self._set_combo_data(self.side_combo, requested_side_player)
        self.analysis_check.setChecked(requested_analysis)

        if not requested_mode:
            if (
                requested_weight_file
                and (self.ai is None or self.loaded_weight_file != requested_weight_file)
            ):
                if not os.path.isfile(requested_weight_file):
                    self._show_error(
                        "Saved game references a missing weights file:\n"
                        f"{requested_weight_file}\n\n"
                        "Load a model manually, then load the save again."
                    )
                    return
                ok = self._load_model_from_selection(
                    mode="file",
                    explicit_path=requested_weight_file,
                    show_errors=True,
                    startup=False,
                )
                if not ok:
                    return
            if self.ai is None:
                self._show_error(
                    "This save is Human vs AI but no model is loaded.\n"
                    "Load a model first, then load the save again."
                )
                return
            difficulty_label, sims = self._selected_difficulty()
            self.ai.difficulty = difficulty_label
            self.ai.sims = sims

        # _load_model_from_selection can restart analysis; stop it before swapping game.
        self._stop_analysis(wait=True, clear=True)
        self.human_only_mode = requested_mode
        self.game = restored_game
        self.game_over = game_over
        if self.human_only_mode:
            self.human_player = PLAYER1
            self.human_turn = True
        else:
            self.human_player = requested_player
            self.human_turn = (
                (not self.game_over) and (self.game.current_player == self.human_player)
            )

        self._apply_mode_state()
        self._refresh_board()

        if (not self.human_only_mode) and (not self.game_over) and (not self.human_turn):
            self._begin_ai_turn()
        else:
            self._restart_analysis_if_needed()

        self._set_message(f"Game loaded from {path}")

    def new_game(self):
        self._cancel_pending_ai_turn()
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
        # Undo changes board/player-to-move; reset any stale analysis first.
        self._cancel_pending_ai_turn()
        self._stop_analysis(wait=True, clear=True)

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
                self._restart_analysis_if_needed()
            return

        if self.ai is None:
            self._show_error("Load a model first.")
            return

        # H-vs-A undo semantics:
        # - If AI is currently thinking (human_turn=False), undo one pending human move.
        # - Otherwise (human_turn=True), undo a full human+AI pair.
        n_to_undo = 1 if not self.human_turn else 2
        if len(self.game.move_history) < n_to_undo:
            self._set_message("Nothing to undo.")
            self._refresh_board()
            self._restart_analysis_if_needed()
            return

        for _ in range(n_to_undo):
            self.game.undo_move()
        self.game_over = False
        self.human_turn = (self.game.current_player == self.human_player)
        if n_to_undo == 1:
            self._set_message("Undid last move (canceled AI turn).")
        else:
            self._set_message("Undone.")
        self._refresh_board()
        if not self.human_turn:
            self._begin_ai_turn()
        else:
            self._restart_analysis_if_needed()

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
        self._cancel_pending_ai_turn()
        request_id = self.ai_turn_request_id
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
        QtCore.QTimer.singleShot(0, lambda rid=request_id: self._run_ai_turn(rid))

    def _run_ai_turn(self, request_id=None):
        if request_id is not None and int(request_id) != int(self.ai_turn_request_id):
            return
        if self._is_human_only():
            return
        if self.ai is None or self.game_over or self.human_turn:
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
            and (self.predict_fn is not None)
            and (not self.game_over)
            and self.human_turn
        )

    def _analysis_worker_matches_position(self):
        worker = self.analysis_worker
        if worker is None:
            return False
        worker_game = getattr(worker, "game", None)
        if worker_game is None:
            return False
        if getattr(worker, "predict_fn", None) is not self.predict_fn:
            return False
        if int(worker_game.current_player) != int(self.game.current_player):
            return False
        if len(worker_game.move_history) != len(self.game.move_history):
            return False
        return np.array_equal(worker_game.board, self.game.board)

    def _start_analysis(self):
        if not self._analysis_should_run():
            return
        if self.analysis_worker is not None:
            return
        self.analysis_session_id += 1
        self.analysis_policy = None
        self.analysis_root_q = 0.0
        self.analysis_sims_done = 0
        self.analysis_started_at = time.time()
        worker = AnalysisWorker(
            session_id=self.analysis_session_id,
            predict_fn=self.predict_fn,
            game=self.game,
            batch_size=AI_MCTS_BATCH,
            c_puct=1.5,
        )
        self.analysis_worker = worker
        worker.analysis_update.connect(self._on_analysis_update)
        worker.analysis_error.connect(self._on_analysis_error)
        worker.finished.connect(lambda w=worker: self._on_analysis_finished(w))
        worker.start()
        self._update_status_box()

    def _stop_analysis(self, wait=False, clear=False):
        worker = self.analysis_worker
        if worker is not None:
            worker.request_stop()
            if wait:
                worker.wait()
                # When waiting synchronously, clear dead worker immediately.
                # Otherwise _start_analysis() may see a stale non-None handle
                # before the finished signal callback runs.
                if self.analysis_worker is worker:
                    self.analysis_worker = None
        if clear:
            self.analysis_policy = None
            self.analysis_root_q = None
            self.analysis_sims_done = 0
            self.analysis_started_at = 0.0
            self.analysis_session_id += 1
        self._update_status_box()

    def _restart_analysis_if_needed(self):
        should_run = self._analysis_should_run()
        if should_run:
            if self.analysis_worker is None:
                self._start_analysis()
            elif not self._analysis_worker_matches_position():
                # Board/player/model changed: restart analysis from current position.
                self._stop_analysis(wait=True, clear=True)
                self._start_analysis()
        else:
            self._stop_analysis(wait=False, clear=True)

    def _on_analysis_update(self, payload):
        session_id, policy, root_q, sims_done = payload
        if int(session_id) != int(self.analysis_session_id):
            return
        self.analysis_policy = np.asarray(policy, dtype=np.float32).ravel()
        self.analysis_root_q = float(root_q)
        self.analysis_sims_done = int(sims_done)
        if self.analysis_started_at <= 0.0:
            self.analysis_started_at = time.time()
        self._update_status_box()
        self._refresh_board()

    def _on_analysis_error(self, session_id, err_text):
        if int(session_id) != int(self.analysis_session_id):
            return
        self.analysis_label.setText(f"Error: {err_text}")
        self.analysis_policy = None
        self.analysis_root_q = None
        self.analysis_sims_done = 0
        self.analysis_started_at = 0.0
        self._stop_analysis(wait=False, clear=False)

    def _on_analysis_finished(self, worker):
        # Ignore stale finished callbacks from older workers.
        if self.analysis_worker is worker:
            self.analysis_worker = None
        worker.deleteLater()
        self._update_status_box()

    def _refresh_board(self):
        glyph = {EMPTY: ".", PLAYER1: "X", PLAYER2: "O"}
        last_move = self._last_move()
        analysis_pi, analysis_max = self._analysis_overlay()

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                val = int(self.game.board[r, c])
                btn = self.board_buttons[r][c]
                btn.setText(glyph[val])
                color, weight = self._apply_cell_font(btn, val)
                bg, border = self._cell_bg_border(
                    r, c, val, last_move, analysis_pi, analysis_max
                )
                self._set_button_style(btn, color, weight, bg, border)
                self._set_cell_tooltip(btn, val, r, c, analysis_pi)
                btn.setEnabled(self._cell_enabled(val))

        self.turn_label.setText(self._turn_text())
        self.moves_label.setText(str(len(self.game.move_history)))
        self._update_status_box()

    def _last_move(self):
        if not self.game.move_history:
            return None
        last = self.game.move_history[-1]
        return int(last[0]), int(last[1])

    def _analysis_overlay(self):
        if (
            self.analysis_policy is None
            or not self._analysis_should_run()
            or len(self.analysis_policy) != BOARD_SIZE * BOARD_SIZE
        ):
            return None, 0.0
        analysis_pi = self.analysis_policy
        legal_mask = self.game.board.reshape(-1) == EMPTY
        if not np.any(legal_mask):
            return analysis_pi, 0.0
        return analysis_pi, float(np.max(analysis_pi[legal_mask]))

    def _apply_cell_font(self, btn, val):
        cell_px = max(1, min(btn.width(), btn.height()))
        if val == EMPTY:
            font_px = max(8, int(cell_px * 0.30))
            color, weight = "#808080", 500
        elif val == PLAYER1:
            font_px = max(10, int(cell_px * 0.62))
            color, weight = "#111111", 700
        else:
            font_px = max(10, int(cell_px * 0.62))
            color, weight = "#0055aa", 700
        f = btn.font()
        f.setPixelSize(font_px)
        btn.setFont(f)
        return color, weight

    def _cell_bg_border(self, row, col, val, last_move, analysis_pi, analysis_max):
        if last_move is not None and last_move == (row, col):
            return "#ffe9a8", "2px solid #d49100"
        if val == EMPTY and analysis_pi is not None and analysis_max > 0.0:
            p = float(analysis_pi[row * BOARD_SIZE + col])
            t = max(0.0, min(1.0, p / analysis_max))
            rr = int(245 - 125 * t)
            gg = int(245 - 25 * t)
            bb = int(245 - 125 * t)
            bg = f"rgb({rr},{gg},{bb})"
            border = "2px solid #3b7d3b" if t >= 0.92 else "1px solid #98c598"
            return bg, border
        return "#f5f5f5", "1px solid #b8b8b8"

    def _set_button_style(self, btn, color, weight, bg, border):
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

    def _set_cell_tooltip(self, btn, val, row, col, analysis_pi):
        if val == EMPTY and analysis_pi is not None:
            p = float(analysis_pi[row * BOARD_SIZE + col])
            btn.setToolTip(f"MCTS policy: {100.0 * p:.2f}%")
            return
        btn.setToolTip("")

    def _cell_enabled(self, val):
        return (
            (not self.game_over)
            and (val == EMPTY)
            and (
                self._is_human_only()
                or (self.ai is not None and self.human_turn)
            )
        )

    def _turn_text(self):
        if self.game_over:
            return "Game over"
        side = "X" if self.game.current_player == PLAYER1 else "O"
        if self._is_human_only():
            return f"Human turn ({side})"
        actor = "Your turn" if self.game.current_player == self.human_player else "AI turn"
        return f"{actor} ({side})"


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = GomokuQtWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
