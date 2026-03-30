#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/run_finetuning.py"
SEEDS=(42 123 456)
CONFIGS=(
  "$SCRIPT_DIR/config_large.yaml"
  "$SCRIPT_DIR/config_no_nsp.yaml"
)

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
  echo "[ERROR] 스크립트를 찾을 수 없습니다: $PYTHON_SCRIPT"
  exit 1
fi

for config in "${CONFIGS[@]}"; do
  if [[ ! -f "$config" ]]; then
    echo "[ERROR] 설정 파일을 찾을 수 없습니다: $config"
    exit 1
  fi
done

FAILED=0

run_with_config() {
  local config_file="$1"
  local label="$2"

  echo "=================================================="
  echo "[INFO] QNLI ${label} 파인튜닝 시작"
  echo "[INFO] Config: $config_file"
  echo "[INFO] Seeds: ${SEEDS[*]}"

  for seed in "${SEEDS[@]}"; do
    echo "--------------------------------------------------"
    echo "[INFO] ${label} seed=$seed 실행"

    if ! python3 "$PYTHON_SCRIPT" --config "$config_file" --seed "$seed"; then
      echo "[ERROR] ${label} seed=$seed 실행 실패"
      FAILED=1
    fi
  done
}

run_with_config "$SCRIPT_DIR/config_large.yaml" "Large"
run_with_config "$SCRIPT_DIR/config_no_nsp.yaml" "No-NSP"

echo "=================================================="
if [[ $FAILED -eq 0 ]]; then
  echo "[INFO] Large + No-NSP 시드 실행 완료"
else
  echo "[WARN] 일부 실행이 실패했습니다. 로그를 확인하세요."
fi

exit "$FAILED"
