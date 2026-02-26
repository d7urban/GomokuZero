#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./train_tournament_loop.sh --games-goal <total_games> [--tournament-dir <dir>] [--python <python_bin>] [-- <extra_tournament_args...>]

Description:
  Repeats this cycle until train_state game_count reaches --games-goal:
    1) run train.py (expects 10,000 games per run from train.py NUM_GAMES)
    2) run eval_tournament.py --tournament-dir <dir>
  Auto-sets Swiss rounds each cycle from player count in tournament dir:
    <=70 players -> 6 rounds, 71-140 -> 7 rounds, >140 -> 8 rounds
  Pass --swiss-rounds in extra tournament args to override.

Examples:
  ./train_tournament_loop.sh --games-goal 50000
  ./train_tournament_loop.sh --games-goal 120000 --tournament-dir botb-weights -- --mcmahon-rounds 6
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

games_goal=""
tournament_dir="weights"
python_bin="${PYTHON_BIN:-python3}"
tournament_extra_args=()

while (($# > 0)); do
  case "$1" in
    --games-goal)
      if (($# < 2)); then
        echo "Missing value for --games-goal" >&2
        exit 1
      fi
      games_goal="$2"
      shift 2
      ;;
    --tournament-dir)
      if (($# < 2)); then
        echo "Missing value for --tournament-dir" >&2
        exit 1
      fi
      tournament_dir="$2"
      shift 2
      ;;
    --python)
      if (($# < 2)); then
        echo "Missing value for --python" >&2
        exit 1
      fi
      python_bin="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      tournament_extra_args=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$games_goal" ]]; then
  echo "--games-goal is required" >&2
  usage >&2
  exit 1
fi
if ! [[ "$games_goal" =~ ^[0-9]+$ ]] || ((games_goal <= 0)); then
  echo "--games-goal must be a positive integer" >&2
  exit 1
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python binary not found: $python_bin" >&2
  exit 1
fi
if [[ ! -f "train.py" ]]; then
  echo "train.py not found in $SCRIPT_DIR" >&2
  exit 1
fi
if [[ ! -f "eval_tournament.py" ]]; then
  echo "eval_tournament.py not found in $SCRIPT_DIR" >&2
  exit 1
fi

read_game_count() {
  "$python_bin" - <<'PY'
import pickle
import os

paths = ("weights/train_state.pkl", "weights/model_config.pkl")
for path in paths:
    if not os.path.exists(path):
        continue
    try:
        with open(path, "rb") as f:
            state = pickle.load(f)
    except Exception:
        continue
    if not isinstance(state, dict):
        continue

    # train.py persists total_games (current schema). Keep legacy fallback.
    for key in ("total_games", "game_count"):
        val = state.get(key)
        if val is None:
            continue
        try:
            games = int(val)
        except Exception:
            continue
        print(max(0, games))
        raise SystemExit(0)

print(0)
PY
}

has_swiss_rounds_override() {
  local arg
  for arg in "${tournament_extra_args[@]}"; do
    case "$arg" in
      --swiss-rounds|--swiss-rounds=*)
        return 0
        ;;
    esac
  done
  return 1
}

count_tournament_players() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    echo "0"
    return 0
  fi
  find "$dir" -maxdepth 1 -type f \
    \( -name "*.weights.h5" -o -name "*.h5" -o -name "*.keras" \) \
    -printf '%f\n' \
    | awk 'tolower($0)!="gomoku_best.weights.h5" && tolower($0)!="gomoku_weights.weights.h5" {c++} END {print c+0}'
}

auto_swiss_rounds_for_players() {
  local players="$1"
  if ((players > 140)); then
    echo "8"
  elif ((players > 70)); then
    echo "7"
  else
    echo "6"
  fi
}

train_chunk_games="$(sed -nE 's/^[[:space:]]*NUM_GAMES[[:space:]]*=[[:space:]]*([0-9]+).*/\1/p' train.py | head -n1)"
if [[ -z "$train_chunk_games" ]]; then
  echo "Could not parse NUM_GAMES from train.py" >&2
  exit 1
fi
if [[ "$train_chunk_games" != "10000" ]]; then
  echo "Warning: train.py NUM_GAMES is $train_chunk_games (expected 10000)." >&2
fi

current_games="$(read_game_count)"
echo "Current games: $current_games"
echo "Goal games: $games_goal"
echo "Train chunk (NUM_GAMES): $train_chunk_games"

if ((current_games >= games_goal)); then
  echo "Goal already reached. Nothing to do."
  exit 0
fi

cycle=0
while ((current_games < games_goal)); do
  cycle=$((cycle + 1))
  before_games="$current_games"

  echo
  echo "=== Cycle $cycle: train.py ==="
  "$python_bin" train.py 2>/dev/null

  current_games="$(read_game_count)"
  if ((current_games <= before_games)); then
    echo "Training did not advance game count (before=$before_games, after=$current_games)." >&2
    exit 1
  fi
  echo "Training advanced by $((current_games - before_games)) games (total=$current_games)."

  echo
  echo "=== Cycle $cycle: eval_tournament.py ==="
  tournament_args=(--tournament-dir "$tournament_dir")
  if has_swiss_rounds_override; then
    echo "Swiss rounds: using explicit value from extra args."
  else
    player_count="$(count_tournament_players "$tournament_dir")"
    swiss_rounds="$(auto_swiss_rounds_for_players "$player_count")"
    tournament_args+=(--swiss-rounds "$swiss_rounds")
    echo "Swiss rounds: auto=$swiss_rounds (players=$player_count)."
  fi
  tournament_args+=("${tournament_extra_args[@]}")

  "$python_bin" eval_tournament.py "${tournament_args[@]}" 2>/dev/null

  echo "Cycle $cycle complete: $current_games / $games_goal games."
done

echo
echo "Reached games goal: $current_games >= $games_goal"
