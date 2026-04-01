# CoLA (Corpus of Linguistic Acceptability) 파인튜닝

이 디렉토리는 BERT 모델을 CoLA 태스크에 파인튜닝하기 위한 코드와 설정을 포함합니다.

## CoLA 태스크 소개

CoLA (Corpus of Linguistic Acceptability)는 문법적 수용성을 판단하는 이진 분류 태스크입니다.

- **태스크 유형**: 단일 문장 이진 분류
- **클래스**: 
  - 0: Unacceptable (문법적으로 수용 불가능)
  - 1: Acceptable (문법적으로 수용 가능)
- **주요 평가 지표**: Matthews 상관계수 (Matthews Correlation Coefficient)
- **데이터 크기**: 
  - 훈련: ~8,551개
  - 검증: ~1,043개
  - 테스트: ~1,063개

## 파일 구조

```
CoLA/
├── README.md                 # 이 파일
├── __init__.py              # 모듈 초기화
├── config_large.yaml        # BERT Large 설정
├── dataset.py               # CoLA 데이터셋 클래스
├── finetuning_model.py      # BERT 분류 모델
├── train.py                 # 훈련 로직
├── evaluate.py              # 평가 함수
├── run_finetuning.py        # 메인 실행 스크립트
├── run_seeds_large.sh       # 다중 시드 실행 스크립트
├── data/                    # 데이터 디렉토리 (자동 다운로드)
└── finetuning_results/      # 결과 저장 디렉토리
    ├── checkpoints/         # 모델 체크포인트
    ├── logs/               # 훈련 로그
    └── evaluation/         # 평가 결과
```

## 사용법

### 1. 단일 실행

```bash
# 훈련 + 평가
python run_finetuning.py --config config_large.yaml --mode both --seed 42

# 훈련만
python run_finetuning.py --config config_large.yaml --mode train --seed 42

# 평가만 (체크포인트 필요)
python run_finetuning.py --config config_large.yaml --mode eval --checkpoint ./finetuning_results/CoLA/checkpoints/best_model.pth
```

### 2. 다중 시드 실행

```bash
# 5개 시드로 실행 (42, 123, 456, 789, 999)
./run_seeds_large.sh
```

## 설정 파일

`config_large.yaml`에서 주요 하이퍼파라미터를 설정할 수 있습니다:

```yaml
# 모델 설정
model:
  model_path: "/path/to/bert/large/model.pth"
  hidden_size: 1024
  num_layers: 24
  num_attention_heads: 16

# 데이터 설정
data:
  batch_size: 16
  max_length: 128

# 훈련 설정
training:
  learning_rate: 1.0e-5
  num_epochs: 5
  warmup_ratio: 0.1
```

## 주요 특징

### 1. Matthews 상관계수 중심 평가
CoLA의 주요 평가 지표인 Matthews 상관계수를 중심으로 모델을 평가하고 최적화합니다.

### 2. 작은 데이터셋 대응
CoLA는 상대적으로 작은 데이터셋이므로:
- 작은 배치 크기 (16) 사용
- 더 많은 에포크 (5) 훈련
- 더 작은 학습률 (1e-5) 사용
- 자주 평가 (100 스텝마다)

### 3. 조기 종료
Matthews 상관계수가 개선되지 않으면 조기 종료하여 과적합을 방지합니다.

## 평가 지표

CoLA 평가에서 제공되는 지표들:

- **Matthews 상관계수**: 주요 평가 지표 (-1 ~ 1, 높을수록 좋음)
- **정확도**: 전체 정확도
- **정밀도**: 긍정 클래스 정밀도
- **재현율**: 긍정 클래스 재현율
- **F1 점수**: 정밀도와 재현율의 조화평균

## 예상 성능

BERT Large 모델의 CoLA 태스크 예상 성능:

- Matthews 상관계수: ~0.60-0.65
- 정확도: ~0.82-0.85

## 문제 해결

### 1. 메모리 부족
```bash
# 배치 크기 줄이기
# config_large.yaml에서 batch_size를 8로 변경
```

### 2. 데이터 다운로드 실패
```bash
# 수동으로 CoLA 데이터 다운로드
# https://dl.fbaipublicfiles.com/glue/data/CoLA.zip
# data/CoLA/ 디렉토리에 압축 해제
```

### 3. 모델 경로 오류
```bash
# config_large.yaml에서 model_path 확인
# 실제 BERT 모델 파일 경로로 수정
```

## 참고 자료

- [CoLA 논문](https://arxiv.org/abs/1805.12471)
- [GLUE 벤치마크](https://gluebenchmark.com/)
- [BERT 논문](https://arxiv.org/abs/1810.04805)