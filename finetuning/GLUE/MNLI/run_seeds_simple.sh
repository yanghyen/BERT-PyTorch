#!/bin/bash

# MNLI 다중 시드 파인튜닝 간단 실행 스크립트
# 사용법: ./run_seeds_simple.sh

echo "MNLI 파인튜닝 시작 - 시드: 43, 123, 456"

# 실행할 시드 목록
SEEDS=(43 123 456)

for SEED in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "시드 $SEED로 MNLI 파인튜닝 시작..."
    echo "=========================================="
    
    # config.yaml에서 시드 값 변경
    sed -i "s/seed: [0-9]*/seed: $SEED/" config.yaml
    
    # 현재 시드 확인
    echo "현재 설정된 시드: $(grep 'seed:' config.yaml | awk '{print $2}')"
    
    # 파인튜닝 실행
    if python3 run_finetuning.py; then
        echo "✅ 시드 $SEED 완료!"
    else
        echo "❌ 시드 $SEED 실패!"
        echo "계속 진행하시겠습니까? (y/n)"
        read -r choice
        if [[ $choice != "y" && $choice != "Y" ]]; then
            echo "스크립트를 중단합니다."
            exit 1
        fi
    fi
    
    echo ""
done

echo "🎉 모든 시드 실행 완료!"