# BERT SST-2 파인튜닝

이 프로젝트는 사전 훈련된 BERT 모델을 Stanford Sentiment Treebank (SST-2) 데이터셋으로 파인튜닝하는 코드입니다.

## 프로젝트 구조

```
finetuning/GLUE/SST-2/
├── dataset.py          # SST-2 데이터셋 로딩 및 전처리
├── model.py           # BERT 분류 모델 정의
├── train.py           # 훈련 로직
├── evaluate.py        # 평가 및 분석
├── run_finetuning.py  # 메인 실행 스크립트
├── config.yaml        # 설정 파일
├── requirements.txt   # 필요 패키지
└── README.md         # 이 파일
```

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

# 배치 크기와 학습률 변경
python run_finetuning.py --batch_size 16 --learning_rate 1e-5

# 훈련만 실행
python run_finetuning.py --mode train

# 평가만 실행
python run_finetuning.py --mode eval --checkpoint_path ./results/SST-2/checkpoints/seed42_20240324_185530/best_model.pt
```

### 3. 다른 config 파일 사용

```bash
python run_finetuning.py --config my_config.yaml
```

### 4. config.yaml 수정

주요 설정들을 `config.yaml`에서 직접 수정:

```yaml
# 모델 경로 변경
model:
  model_path: "/path/to/your/bert/model.pth"

# 훈련 설정 조정
training:
  learning_rate: 1.0e-5
  batch_size: 16
  num_epochs: 5

# 실행 모드 변경
misc:
  seed: 123
  mode: "train"  # train, eval, both
```

## 주요 매개변수

### 모델 관련
- `--model_path`: 사전 훈련된 BERT 모델 경로
- `--hidden_size`: BERT 숨겨진 차원 크기 (기본값: 768)
- `--num_layers`: BERT 레이어 수 (기본값: 12)
- `--num_attention_heads`: 어텐션 헤드 수 (기본값: 12)

### 데이터 관련
- `--data_dir`: SST-2 데이터 디렉토리 (기본값: ./data/SST-2)
- `--max_length`: 최대 시퀀스 길이 (기본값: 128)
- `--batch_size`: 배치 크기 (기본값: 32)

### 훈련 관련
- `--learning_rate`: 학습률 (기본값: 2e-5)
- `--num_epochs`: 에포크 수 (기본값: 3)
- `--weight_decay`: 가중치 감쇠 (기본값: 0.01)
- `--warmup_ratio`: 웜업 비율 (기본값: 0.1)

## 데이터셋

SST-2 (Stanford Sentiment Treebank) 데이터셋은 영화 리뷰의 감정을 이진 분류(긍정/부정)하는 태스크입니다.

- **훈련 데이터**: 67,349개 샘플
- **검증 데이터**: 872개 샘플  
- **테스트 데이터**: 1,821개 샘플
- **클래스**: 2개 (부정: 0, 긍정: 1)

데이터는 첫 실행 시 자동으로 다운로드됩니다.

## 결과

### 성능 메트릭
- **정확도 (Accuracy)**
- **정밀도 (Precision)**
- **재현율 (Recall)**
- **F1 점수**

### 출력 파일

실행할 때마다 seed와 타임스탬프 기반으로 고유한 디렉토리가 생성됩니다:

- `./results/SST-2/checkpoints/seed42_20240324_185530/`: 훈련된 모델 체크포인트
- `./results/SST-2/logs/seed42_20240324_185530/`: 훈련 로그
- `./results/SST-2/evaluation/seed42_20240324_185530/`: 평가 결과 및 분석
  - `confusion_matrix.png`: 혼동 행렬
  - `confidence_distribution.png`: 신뢰도 분포
  - `analysis.json`: 상세 분석 결과

## 예상 성능

BERT-base 모델로 SST-2에서 기대할 수 있는 성능:
- **정확도**: ~92-94%
- **F1 점수**: ~92-94%

## 하이퍼파라미터 튜닝

`config.yaml` 파일에서 다양한 하이퍼파라미터 조합을 확인할 수 있습니다:

```yaml
alternatives:
  learning_rates: [1e-5, 2e-5, 3e-5, 5e-5]
  batch_sizes: [16, 32, 64]
  num_epochs: [2, 3, 4, 5]
  warmup_ratios: [0.06, 0.1, 0.2]
```

## 문제 해결

### 1. 메모리 부족 오류
- 배치 크기를 줄여보세요: `--batch_size 16`
- 시퀀스 길이를 줄여보세요: `--max_length 64`

### 2. 모델 파일을 찾을 수 없음
- 모델 경로가 올바른지 확인: `--model_path path/to/your/model.pth`

### 3. 데이터 다운로드 실패
- 수동으로 SST-2 데이터를 다운로드하여 `./data/SST-2/` 디렉토리에 배치

## 고급 기능

### Weights & Biases 연동
```bash
python run_finetuning.py --use_wandb --wandb_project my-bert-project
```

### 조기 종료
```bash
python run_finetuning.py --early_stopping_patience 3
```

### 그래디언트 클리핑
```bash
python run_finetuning.py --max_grad_norm 1.0
```

## 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다.

## 참고 자료

- [BERT 논문](https://arxiv.org/abs/1810.04805)
- [GLUE 벤치마크](https://gluebenchmark.com/)
- [SST 데이터셋](https://nlp.stanford.edu/sentiment/)