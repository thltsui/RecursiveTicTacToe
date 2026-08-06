#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
EPISODES="${EPISODES:-20000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
EVAL_GAMES="${EVAL_GAMES:-300}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
GAMMA="${GAMMA:-0.99}"
TARGET_UPDATE_FREQ="${TARGET_UPDATE_FREQ:-500}"
BUFFER_CAPACITY="${BUFFER_CAPACITY:-100000}"
TRAIN_EVERY="${TRAIN_EVERY:-4}"
LEARNING_STARTS="${LEARNING_STARTS:-1000}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cpu}"
RUN_NAME="${RUN_NAME:-practical_dqn_seed${SEED}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRIPT_DIR}/checkpoints/${RUN_NAME}}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/results/${RUN_NAME}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

mkdir -p "${CHECKPOINT_DIR}" "${OUT_DIR}"

cd "${REPO_ROOT}"
echo "Running practical DQN experiment"
echo "  episodes=${EPISODES} eval_every=${EVAL_EVERY} eval_games=${EVAL_GAMES}"
echo "  seed=${SEED} device=${DEVICE}"
echo "  learning_rate=${LEARNING_RATE} gamma=${GAMMA} batch_size=${BATCH_SIZE}"
echo "  target_update_freq=${TARGET_UPDATE_FREQ} buffer_capacity=${BUFFER_CAPACITY}"
echo "  train_every=${TRAIN_EVERY} learning_starts=${LEARNING_STARTS}"
echo "  results=${OUT_DIR}"

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" experiments/dqn/train.py \
  --episodes "${EPISODES}" \
  --eval-every "${EVAL_EVERY}" \
  --eval-games "${EVAL_GAMES}" \
  --lr "${LEARNING_RATE}" \
  --gamma "${GAMMA}" \
  --batch-size "${BATCH_SIZE}" \
  --target-update-freq "${TARGET_UPDATE_FREQ}" \
  --buffer-capacity "${BUFFER_CAPACITY}" \
  --train-every "${TRAIN_EVERY}" \
  --learning-starts "${LEARNING_STARTS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --out-dir "${OUT_DIR}" 2>&1 | tee "${OUT_DIR}/console.log"
