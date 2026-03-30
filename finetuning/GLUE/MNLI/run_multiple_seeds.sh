#!/bin/bash

# MNLI 다중 시드 파인튜닝 자동 실행 스크립트
# 사용법: ./run_multiple_seeds.sh

set -e  # 에러 발생 시 스크립트 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 스크립트 시작 시간 기록
SCRIPT_START_TIME=$(date +%s)
SCRIPT_START_DATE=$(date '+%Y-%m-%d %H:%M:%S')

log_info "MNLI 다중 시드 파인튜닝 시작: $SCRIPT_START_DATE"
log_info "실행할 시드: 43, 123, 456"

# 실행할 시드 목록
SEEDS=(43 123 456)

# 현재 디렉토리 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.yaml"
PYTHON_SCRIPT="$SCRIPT_DIR/run_finetuning.py"

# 필요한 파일 존재 확인
if [ ! -f "$CONFIG_FILE" ]; then
    log_error "config.yaml 파일을 찾을 수 없습니다: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
    log_error "run_finetuning.py 파일을 찾을 수 없습니다: $PYTHON_SCRIPT"
    exit 1
fi

# 결과 저장을 위한 배열
declare -a RESULTS
TOTAL_SEEDS=${#SEEDS[@]}
COMPLETED_SEEDS=0
FAILED_SEEDS=0

# 전체 결과 요약 파일
SUMMARY_FILE="$SCRIPT_DIR/multiple_seeds_summary_$(date +%Y%m%d_%H%M%S).txt"

log_info "결과 요약 파일: $SUMMARY_FILE"

# 각 시드에 대해 순차 실행
for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    SEED_NUM=$((i + 1))
    
    log_info "=========================================="
    log_info "시드 $SEED 실행 중... ($SEED_NUM/$TOTAL_SEEDS)"
    log_info "=========================================="
    
    # 시드별 시작 시간 기록
    SEED_START_TIME=$(date +%s)
    SEED_START_DATE=$(date '+%Y-%m-%d %H:%M:%S')
    
    # config.yaml에서 시드 값 임시 변경
    log_info "config.yaml에서 시드를 $SEED로 변경 중..."
    
    # 백업 생성
    cp "$CONFIG_FILE" "$CONFIG_FILE.backup"
    
    # 시드 값 변경 (sed 사용)
    sed -i "s/seed: [0-9]*/seed: $SEED/" "$CONFIG_FILE"
    
    # 변경된 시드 확인
    CURRENT_SEED=$(grep "seed:" "$CONFIG_FILE" | awk '{print $2}')
    log_info "현재 설정된 시드: $CURRENT_SEED"
    
    # 파인튜닝 실행
    log_info "MNLI 파인튜닝 시작 (시드: $SEED)..."
    
    if python3 "$PYTHON_SCRIPT"; then
        # 성공한 경우
        SEED_END_TIME=$(date +%s)
        SEED_DURATION=$((SEED_END_TIME - SEED_START_TIME))
        SEED_END_DATE=$(date '+%Y-%m-%d %H:%M:%S')
        
        log_success "시드 $SEED 파인튜닝 완료!"
        log_info "소요 시간: $SEED_DURATION초 ($(($SEED_DURATION / 60))분 $(($SEED_DURATION % 60))초)"
        
        RESULTS[$i]="SUCCESS: 시드 $SEED - 시작: $SEED_START_DATE, 완료: $SEED_END_DATE, 소요시간: ${SEED_DURATION}초"
        COMPLETED_SEEDS=$((COMPLETED_SEEDS + 1))
        
    else
        # 실패한 경우
        SEED_END_TIME=$(date +%s)
        SEED_DURATION=$((SEED_END_TIME - SEED_START_TIME))
        SEED_END_DATE=$(date '+%Y-%m-%d %H:%M:%S')
        
        log_error "시드 $SEED 파인튜닝 실패!"
        log_warning "소요 시간: $SEED_DURATION초"
        
        RESULTS[$i]="FAILED: 시드 $SEED - 시작: $SEED_START_DATE, 실패: $SEED_END_DATE, 소요시간: ${SEED_DURATION}초"
        FAILED_SEEDS=$((FAILED_SEEDS + 1))
        
        # 실패 시에도 계속 진행할지 물어보기
        log_warning "시드 $SEED 실행이 실패했습니다. 다음 시드로 계속 진행하시겠습니까? (y/n)"
        read -r CONTINUE_CHOICE
        if [[ $CONTINUE_CHOICE != "y" && $CONTINUE_CHOICE != "Y" ]]; then
            log_info "사용자 요청으로 스크립트를 중단합니다."
            break
        fi
    fi
    
    # config.yaml 원복
    mv "$CONFIG_FILE.backup" "$CONFIG_FILE"
    log_info "config.yaml 원복 완료"
    
    # 다음 시드가 있다면 잠시 대기
    if [ $SEED_NUM -lt $TOTAL_SEEDS ]; then
        log_info "다음 시드 실행 전 5초 대기..."
        sleep 5
    fi
done

# 전체 실행 완료
SCRIPT_END_TIME=$(date +%s)
SCRIPT_DURATION=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
SCRIPT_END_DATE=$(date '+%Y-%m-%d %H:%M:%S')

log_info "=========================================="
log_info "모든 시드 실행 완료!"
log_info "=========================================="

# 결과 요약 생성
{
    echo "MNLI 다중 시드 파인튜닝 결과 요약"
    echo "=================================="
    echo ""
    echo "실행 시간: $SCRIPT_START_DATE ~ $SCRIPT_END_DATE"
    echo "총 소요 시간: $SCRIPT_DURATION초 ($(($SCRIPT_DURATION / 60))분 $(($SCRIPT_DURATION % 60))초)"
    echo ""
    echo "실행 시드: ${SEEDS[*]}"
    echo "총 시드 수: $TOTAL_SEEDS"
    echo "성공한 시드: $COMPLETED_SEEDS"
    echo "실패한 시드: $FAILED_SEEDS"
    echo ""
    echo "상세 결과:"
    echo "----------"
    for result in "${RESULTS[@]}"; do
        echo "$result"
    done
    echo ""
    echo "생성 시간: $(date '+%Y-%m-%d %H:%M:%S')"
} > "$SUMMARY_FILE"

# 콘솔에도 요약 출력
log_success "실행 완료 요약:"
log_info "- 총 시드 수: $TOTAL_SEEDS"
log_info "- 성공한 시드: $COMPLETED_SEEDS"
log_info "- 실패한 시드: $FAILED_SEEDS"
log_info "- 총 소요 시간: $(($SCRIPT_DURATION / 60))분 $(($SCRIPT_DURATION % 60))초"
log_info "- 상세 결과는 다음 파일에 저장되었습니다: $SUMMARY_FILE"

# 성공률 계산 및 출력
if [ $TOTAL_SEEDS -gt 0 ]; then
    SUCCESS_RATE=$(echo "scale=1; $COMPLETED_SEEDS * 100 / $TOTAL_SEEDS" | bc -l 2>/dev/null || echo "N/A")
    log_info "- 성공률: $SUCCESS_RATE%"
fi

# 최종 상태에 따른 종료 코드
if [ $FAILED_SEEDS -eq 0 ]; then
    log_success "모든 시드가 성공적으로 완료되었습니다! 🎉"
    exit 0
elif [ $COMPLETED_SEEDS -gt 0 ]; then
    log_warning "일부 시드가 실패했지만 $COMPLETED_SEEDS개 시드는 성공했습니다."
    exit 1
else
    log_error "모든 시드가 실패했습니다."
    exit 2
fi