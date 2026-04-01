#!/bin/bash

# BERT Large CoLA 파인튜닝 - 다중 시드 실행 스크립트

echo "=== BERT Large CoLA 파인튜닝 시작 ==="
echo "시작 시간: $(date)"

# 기본 설정
CONFIG_FILE="config_large.yaml"
SEEDS=(42 123 456 789 999)
BASE_DIR="./finetuning_results/CoLA"

# 결과 디렉토리 생성
mkdir -p ${BASE_DIR}/multiple_seeds_results

# 각 시드별로 파인튜닝 실행
for seed in "${SEEDS[@]}"; do
    echo ""
    echo "=== 시드 ${seed} 파인튜닝 시작 ==="
    echo "시작 시간: $(date)"
    
    # 파인튜닝 실행
    python run_finetuning.py \
        --config ${CONFIG_FILE} \
        --mode both \
        --seed ${seed}
    
    if [ $? -eq 0 ]; then
        echo "시드 ${seed} 파인튜닝 완료"
    else
        echo "시드 ${seed} 파인튜닝 실패"
        exit 1
    fi
    
    echo "종료 시간: $(date)"
done

# 결과 요약 생성
echo ""
echo "=== 다중 시드 결과 요약 생성 ==="

SUMMARY_FILE="${BASE_DIR}/multiple_seeds_results/summary_$(date +%Y%m%d_%H%M%S).txt"

echo "BERT Large CoLA 파인튜닝 다중 시드 결과 요약" > ${SUMMARY_FILE}
echo "생성 시간: $(date)" >> ${SUMMARY_FILE}
echo "사용된 시드: ${SEEDS[*]}" >> ${SUMMARY_FILE}
echo "설정 파일: ${CONFIG_FILE}" >> ${SUMMARY_FILE}
echo "" >> ${SUMMARY_FILE}

# 각 시드별 결과 수집
echo "시드별 결과:" >> ${SUMMARY_FILE}
echo "============" >> ${SUMMARY_FILE}

for seed in "${SEEDS[@]}"; do
    echo "" >> ${SUMMARY_FILE}
    echo "시드 ${seed}:" >> ${SUMMARY_FILE}
    
    # 가장 최근 평가 결과 찾기
    EVAL_DIR="${BASE_DIR}/evaluation"
    LATEST_RESULT=$(find ${EVAL_DIR} -name "*seed${seed}*" -type d | sort | tail -1)
    
    if [ -n "${LATEST_RESULT}" ] && [ -f "${LATEST_RESULT}/evaluation_results.json" ]; then
        # JSON에서 주요 메트릭 추출
        python3 -c "
import json
try:
    with open('${LATEST_RESULT}/evaluation_results.json', 'r') as f:
        results = json.load(f)
    
    if 'dev' in results:
        dev_results = results['dev']
        print(f'  Matthews 상관계수: {dev_results.get(\"matthews_corr\", \"N/A\"):.4f}')
        print(f'  정확도: {dev_results.get(\"accuracy\", \"N/A\"):.4f}')
        print(f'  F1 점수: {dev_results.get(\"f1\", \"N/A\"):.4f}')
        print(f'  정밀도: {dev_results.get(\"precision\", \"N/A\"):.4f}')
        print(f'  재현율: {dev_results.get(\"recall\", \"N/A\"):.4f}')
    else:
        print('  결과를 찾을 수 없습니다.')
except Exception as e:
    print(f'  오류: {e}')
" >> ${SUMMARY_FILE}
    else
        echo "  결과 파일을 찾을 수 없습니다." >> ${SUMMARY_FILE}
    fi
done

# 평균 성능 계산
echo "" >> ${SUMMARY_FILE}
echo "평균 성능:" >> ${SUMMARY_FILE}
echo "==========" >> ${SUMMARY_FILE}

python3 -c "
import json
import numpy as np
import glob
import os

# 모든 결과 파일 수집
result_files = []
for seed in [42, 123, 456, 789, 999]:
    pattern = '${BASE_DIR}/evaluation/*seed{}_*/evaluation_results.json'.format(seed)
    files = glob.glob(pattern)
    if files:
        result_files.append(max(files, key=os.path.getctime))  # 가장 최근 파일

matthews_scores = []
accuracies = []
f1_scores = []
precisions = []
recalls = []

for file_path in result_files:
    try:
        with open(file_path, 'r') as f:
            results = json.load(f)
        
        if 'dev' in results:
            dev_results = results['dev']
            matthews_scores.append(dev_results.get('matthews_corr', 0))
            accuracies.append(dev_results.get('accuracy', 0))
            f1_scores.append(dev_results.get('f1', 0))
            precisions.append(dev_results.get('precision', 0))
            recalls.append(dev_results.get('recall', 0))
    except Exception as e:
        print(f'파일 처리 오류: {file_path}, {e}')

if matthews_scores:
    print(f'Matthews 상관계수: {np.mean(matthews_scores):.4f} ± {np.std(matthews_scores):.4f}')
    print(f'정확도: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}')
    print(f'F1 점수: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}')
    print(f'정밀도: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}')
    print(f'재현율: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}')
    print(f'실행된 시드 수: {len(matthews_scores)}')
else:
    print('계산할 결과가 없습니다.')
" >> ${SUMMARY_FILE}

echo "" >> ${SUMMARY_FILE}
echo "전체 완료 시간: $(date)" >> ${SUMMARY_FILE}

echo ""
echo "=== 모든 시드 파인튜닝 완료 ==="
echo "결과 요약: ${SUMMARY_FILE}"
echo "완료 시간: $(date)"

# 요약 파일 내용 출력
echo ""
echo "=== 결과 요약 ==="
cat ${SUMMARY_FILE}