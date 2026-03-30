#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/run_finetuning.py"
CONFIG_FILE="$SCRIPT_DIR/config_large.yaml"
SEEDS=(42 123 456)

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
  echo "[ERROR] 스크립트를 찾을 수 없습니다: $PYTHON_SCRIPT"
  exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[ERROR] 설정 파일을 찾을 수 없습니다: $CONFIG_FILE"
  exit 1
fi

echo "[INFO] QNLI Large 파인튜닝 시작"
echo "[INFO] Config: $CONFIG_FILE"
echo "[INFO] Seeds: ${SEEDS[*]}"

for seed in "${SEEDS[@]}"; do
  echo "----------------------------------------"
  echo "[INFO] seed=$seed 실행"
  python3 "$PYTHON_SCRIPT" --config "$CONFIG_FILE" --seed "$seed"
done

echo "----------------------------------------"
echo "[INFO] 모든 시드 실행 완료"
