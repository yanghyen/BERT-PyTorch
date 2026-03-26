# BERT MNLI 파인튜닝

이 프로젝트는 사전 훈련된 BERT 모델을 Multi-Genre Natural Language Inference (MNLI) 데이터셋으로 파인튜닝하는 코드입니다.

## 프로젝트 구조

```
finetuning/GLUE/MNLI/
├── dataset.py          # MNLI 데이터셋 로딩 및 전처리
├── finetuning_model.py # BERT 분류 모델 정의 (3-class)
├── train.py           # 훈련 로직
├── evaluate.py        # 평가 및 분석
├── run_finetuning.py  # 메인 실행 스크립트
├── config.yaml        # 설정 파일
├── requirements.txt   # 필요 패키지
└── README.md         # 이 파일
```

## MNLI 태스크 개요

MNLI (Multi-Genre Natural Language Inference)는 자연어 추론 태스크로, 주어진 전제(premise)와 가설(hypothesis) 사이의 논리적 관계를 분류합니다.

### 클래스
- **Entailment (함의)**: 전제가 참이면 가설도 반드시 참
- **Contradiction (모순)**: 전제가 참이면 가설은 반드시 거짓
- **Neutral (중립)**: 전제만으로는 가설의 참/거짓을 판단할 수 없음

### 예시
- **전제**: "A man is walking his dog in the park."
- **가설**: "A person is outside with an animal."
- **레이블**: Entailment

## 설치

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 사용법

### 1. 기본 실행 (config.yaml 사용)

```bash
python run_finetuning.py
```

모든 설정은 `config.yaml`에서 관리됩니다. 기본적으로 훈련과 평가를 모두 수행합니다.

### 2. 특정 설정 오버라이드

```bash
# 다른 seed로 실행
python run_finetuning.py --seed 123

# 배치 크기와 학습률 변경 (MNLI는 메모리 사용량이 크므로 작은 배치 크기 권장)
python run_finetuning.py --batch_size 8 --learning_rate 1e-5

# 최대 시퀀스 길이 조정 (MNLI는 두 문장이므로 더 긴 시퀀스 필요)
python run_finetuning.py --max_length 384

# 훈련만 실행
python run_finetuning.py --mode train

# 평가만 실행 (matched와 mismatched 모두)
python run_finetuning.py --mode eval --evaluate_both_dev --checkpoint_path ./results/MNLI/checkpoints/seed42_20240324_185530/best_model.pt
```

### 3. 다른 config 파일 사용

```bash
python run_finetuning.py --config my_mnli_config.yaml
```

### 4. config.yaml 수정

주요 설정들을 `config.yaml`에서 직접 수정:

```yaml
# 모델 경로 변경
model:
  model_path: "/path/to/your/bert/model.pth"

# 데이터 설정 (MNLI 특화)
data:
  max_length: 256  # 두 문장이므로 더 긴 시퀀스
  batch_size: 16   # 메모리 사용량 고려

# 훈련 설정 조정
training:
  learning_rate: 2.0e-5
  num_epochs: 3

# MNLI 특화 설정
mnli_specific:
  use_matched_dev: true
  use_mismatched_dev: true
```

## 주요 매개변수

### 모델 관련
- `--model_path`: 사전 훈련된 BERT 모델 경로
- `--hidden_size`: BERT 숨겨진 차원 크기 (기본값: 768)
- `--num_layers`: BERT 레이어 수 (기본값: 12)
- `--num_attention_heads`: 어텐션 헤드 수 (기본값: 12)

### 데이터 관련
- `--data_dir`: MNLI 데이터 디렉토리 (기본값: ./data/MNLI)
- `--max_length`: 최대 시퀀스 길이 (기본값: 256, MNLI는 더 긴 시퀀스 필요)
- `--batch_size`: 배치 크기 (기본값: 16, 메모리 사용량 고려)

### 훈련 관련
- `--learning_rate`: 학습률 (기본값: 2e-5)
- `--num_epochs`: 에포크 수 (기본값: 3)
- `--weight_decay`: 가중치 감쇠 (기본값: 0.01)
- `--warmup_ratio`: 웜업 비율 (기본값: 0.1)

### MNLI 특화
- `--evaluate_both_dev`: matched와 mismatched dev 모두 평가

## 데이터셋

MNLI (Multi-Genre Natural Language Inference) 데이터셋은 다양한 장르의 텍스트에서 자연어 추론을 수행하는 태스크입니다.

- **훈련 데이터**: 392,702개 샘플
- **검증 데이터 (Matched)**: 9,815개 샘플  
- **검증 데이터 (Mismatched)**: 9,832개 샘플
- **테스트 데이터**: Matched/Mismatched 각각 9,796개, 9,847개 샘플
- **클래스**: 3개 (contradiction: 0, neutral: 1, entailment: 2)

### Matched vs Mismatched
- **Matched**: 훈련 데이터와 같은 장르의 텍스트
- **Mismatched**: 훈련 데이터에 없는 새로운 장르의 텍스트

데이터는 첫 실행 시 자동으로 다운로드됩니다.

## 결과

### 성능 메트릭
- **정확도 (Accuracy)**
- **정밀도 (Precision)**: Macro/Micro 평균
- **재현율 (Recall)**: Macro/Micro 평균
- **F1 점수**: Macro/Micro 평균
- **클래스별 성능**

### 출력 파일

실행할 때마다 seed와 타임스탬프 기반으로 고유한 디렉토리가 생성됩니다:

- `./results/MNLI/checkpoints/seed42_20240324_185530/`: 훈련된 모델 체크포인트
- `./results/MNLI/logs/seed42_20240324_185530/`: 훈련 로그
- `./results/MNLI/evaluation/seed42_20240324_185530/`: 평가 결과 및 분석
  - `confusion_matrix_matched.png`: Matched 혼동 행렬
  - `confusion_matrix_mismatched.png`: Mismatched 혼동 행렬
  - `matched_vs_mismatched.png`: 성능 비교 차트
  - `class_performance.png`: 클래스별 성능
  - `analysis.json`: 상세 분석 결과

## 예상 성능

BERT-base 모델로 MNLI에서 기대할 수 있는 성능:
- **Matched 정확도**: ~84-86%
- **Mismatched 정확도**: ~83-85%
- **F1 점수**: ~84-86%

일반적으로 Matched 성능이 Mismatched보다 1-2% 높습니다.

## 하이퍼파라미터 튜닝

`config.yaml` 파일에서 다양한 하이퍼파라미터 조합을 확인할 수 있습니다:

```yaml
alternatives:
  learning_rates: [1e-5, 2e-5, 3e-5, 5e-5]
  batch_sizes: [8, 16, 32]  # MNLI는 메모리 사용량이 더 크므로 작은 배치
  num_epochs: [2, 3, 4, 5]
  warmup_ratios: [0.06, 0.1, 0.2]
  max_lengths: [128, 256, 384]  # 다양한 시퀀스 길이 실험
```

## 문제 해결

### 1. 메모리 부족 오류
- 배치 크기를 줄여보세요: `--batch_size 8`
- 시퀀스 길이를 줄여보세요: `--max_length 128`
- Gradient accumulation 사용 고려

### 2. 모델 파일을 찾을 수 없음
- 모델 경로가 올바른지 확인: `--model_path path/to/your/model.pth`

### 3. 데이터 다운로드 실패
- 수동으로 MNLI 데이터를 다운로드하여 `./data/MNLI/` 디렉토리에 배치
- 필요한 파일: `train.tsv`, `dev_matched.tsv`, `dev_mismatched.tsv`

### 4. 느린 훈련 속도
- GPU 사용 확인
- 더 작은 배치 크기나 시퀀스 길이 사용
- Mixed precision training 고려

## 고급 기능

### Weights & Biases 연동
```bash
python run_finetuning.py --use_wandb --wandb_project my-bert-mnli-project
```

### 조기 종료
```bash
python run_finetuning.py --early_stopping_patience 3
```

### 그래디언트 클리핑
```bash
python run_finetuning.py --max_grad_norm 1.0
```

### Matched/Mismatched 동시 평가
```bash
python run_finetuning.py --evaluate_both_dev
```

## MNLI vs SST-2 차이점

| 특성 | SST-2 | MNLI |
|------|-------|------|
| 태스크 | 감정 분석 | 자연어 추론 |
| 입력 | 단일 문장 | 두 문장 (premise, hypothesis) |
| 클래스 수 | 2개 | 3개 |
| 시퀀스 길이 | 128 | 256+ |
| 배치 크기 | 32 | 16 |
| 평가 셋 | 단일 | Matched/Mismatched |

## 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다.

## 참고 자료

- [BERT 논문](https://arxiv.org/abs/1810.04805)
- [GLUE 벤치마크](https://gluebenchmark.com/)
- [MNLI 데이터셋](https://cims.nyu.edu/~sbowman/multinli/)
- [자연어 추론 개요](https://nlp.stanford.edu/~wcmac/papers/snli_emnlp_2015.pdf)