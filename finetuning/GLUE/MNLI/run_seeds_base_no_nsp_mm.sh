#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/run_finetuning.py"
BASE_CONFIG="$SCRIPT_DIR/config_base.yaml"
NO_NSP_CONFIG="$SCRIPT_DIR/config_no_nsp.yaml"
SEEDS=(42 123 456)
CHECKPOINT_ROOT="$SCRIPT_DIR/finetuning_results/MNLI/prev/checkpoints"

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
  echo "[ERROR] 스크립트를 찾을 수 없습니다: $PYTHON_SCRIPT"
  exit 1
fi

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "[ERROR] Base 설정 파일을 찾을 수 없습니다: $BASE_CONFIG"
  exit 1
fi

if [[ ! -f "$NO_NSP_CONFIG" ]]; then
  echo "[ERROR] No-NSP 설정 파일을 찾을 수 없습니다: $NO_NSP_CONFIG"
  exit 1
fi

run_group() {
  local label="$1"
  local config="$2"
  local variant="$3"

  echo "========================================"
  echo "[INFO] $label 실행 시작"
  echo "[INFO] Config: $config"
  echo "[INFO] Seeds: ${SEEDS[*]}"
  echo "========================================"

  for seed in "${SEEDS[@]}"; do
    local pattern
    local checkpoint
    local -a matches=()

    if [[ "$variant" == "base" ]]; then
      # base run_id 예: seed42_20260331_123456
      pattern="$CHECKPOINT_ROOT/seed${seed}_[0-9]*/best_model.pt"
    else
      pattern="$CHECKPOINT_ROOT/seed${seed}_no_nsp_*/best_model.pt"
    fi

    shopt -s nullglob
    matches=($pattern)
    shopt -u nullglob

    if [[ ${#matches[@]} -eq 0 ]]; then
      echo "[WARN] $label | seed=$seed 체크포인트를 찾지 못해 건너뜁니다."
      echo "[WARN] 검색 패턴: $pattern"
      continue
    fi

    # 동일 seed의 여러 실행 중 가장 최근 파일을 사용
    checkpoint="$(ls -1t "${matches[@]}" | head -n 1)"

    echo "----------------------------------------"
    echo "[INFO] $label | seed=$seed 평가 실행"
    echo "[INFO] checkpoint: $checkpoint"
    # MNLI run_finetuning.py는 eval 모드에서도 matched/mismatched를 모두 평가함.
    python3 "$PYTHON_SCRIPT" \
      --config "$config" \
      --mode eval \
      --seed "$seed" \
      --checkpoint_path "$checkpoint"
  done

  echo "----------------------------------------"
  echo "[INFO] $label 실행 완료"
}

run_group "MNLI Base" "$BASE_CONFIG" "base"
run_group "MNLI No-NSP" "$NO_NSP_CONFIG" "no_nsp"

echo "========================================"
echo "[INFO] Base vs No-NSP 저장 체크포인트 평가(mm 포함) 완료"
echo "========================================"
