# BERT QNLI 파인튜닝

이 프로젝트는 사전 훈련된 BERT 모델을 Question Natural Language Inference (QNLI) 데이터셋으로 파인튜닝하는 코드입니다.

## 프로젝트 구조

```
finetuning/GLUE/QNLI/
├── dataset.py          # QNLI 데이터셋 로딩 및 전처리
├── finetuning_model.py # BERT 분류 모델 정의
├── train.py           # 훈련 로직
├── evaluate.py        # 평가 및 분석
├── run_finetuning.py  # 메인 실행 스크립트
├── config.yaml        # 설정 파일
├── requirements.txt   # 필요 패키지
└── README.md         # 이 파일
```

## QNLI 태스크 소개

QNLI (Question Natural Language Inference)는 GLUE 벤치마크의 일부로, 질문-문단 쌍이 주어졌을 때 문단이 질문에 대한 답을 포함하는지 판단하는 이진 분류 태스크입니다.

- **입력**: 질문(question)과 문단(sentence) 쌍
- **출력**: entailment (답 포함) 또는 not_entailment (답 미포함)
- **평가 메트릭**: 정확도(Accuracy)

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
python run_finetuning.py --batch_size 8 --learning_rate 1e-5

# 최대 시퀀스 길이 조정
python run_finetuning.py --max_length 384

# 훈련만 실행
python run_finetuning.py --mode train

# 평가만 실행
python run_finetuning.py --mode eval --checkpoint_path ./results/QNLI/checkpoints/seed42_20240324_185530/best_model.pt
```

### 3. 다른 config 파일 사용

```bash
python run_finetuning.py --config my_qnli_config.yaml
```

### 4. config.yaml 수정

주요 설정들을 `config.yaml`에서 직접 수정:

```yaml
# 모델 경로 변경
model:
  model_path: "/path/to/your/bert/model.pth"

# 데이터 설정 조정 (QNLI는 더 긴 시퀀스 필요)
data:
  max_length: 512
  batch_size: 16

# 훈련 설정 조정
training:
  learning_rate: 2.0e-5
  num_epochs: 3

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
- `--data_dir`: QNLI 데이터 디렉토리 (기본값: ./data/QNLI)
- `--max_length`: 최대 시퀀스 길이 (기본값: 512, QNLI는 질문+문단이므로 길게 설정)
- `--batch_size`: 배치 크기 (기본값: 16, 긴 시퀀스로 인해 SST-2보다 작게 설정)

### 훈련 관련
- `--learning_rate`: 학습률 (기본값: 2e-5)
- `--num_epochs`: 에포크 수 (기본값: 3)
- `--weight_decay`: 가중치 감쇠 (기본값: 0.01)
- `--warmup_ratio`: 웜업 비율 (기본값: 0.1)

## 데이터셋

QNLI 데이터셋은 SQuAD 1.1에서 파생된 데이터로 질문-문단 쌍의 함의 관계를 판단합니다.

- **훈련 데이터**: 104,743개 샘플
- **검증 데이터**: 5,463개 샘플  
- **테스트 데이터**: 5,463개 샘플
- **클래스**: 2개 (entailment: 0, not_entailment: 1)

데이터는 첫 실행 시 자동으로 다운로드됩니다.

### 데이터 형식

```
질문: What is the capital of France?
문단: Paris is the capital and most populous city of France.
레이블: entailment (문단이 질문에 대한 답을 포함)

질문: What is the population of Tokyo?
문단: Paris is the capital and most populous city of France.
레이블: not_entailment (문단이 질문에 대한 답을 포함하지 않음)
```

## 결과

### 성능 메트릭
- **정확도 (Accuracy)** - 주요 평가 지표
- **정밀도 (Precision)**
- **재현율 (Recall)**
- **F1 점수**

### 출력 파일

실행할 때마다 seed와 타임스탬프 기반으로 고유한 디렉토리가 생성됩니다:

- `./results/QNLI/checkpoints/seed42_20240324_185530/`: 훈련된 모델 체크포인트
- `./results/QNLI/logs/seed42_20240324_185530/`: 훈련 로그
- `./results/QNLI/evaluation/seed42_20240324_185530/`: 평가 결과 및 분석
  - `confusion_matrix.png`: 혼동 행렬
  - `confidence_distribution.png`: 신뢰도 분포
  - `analysis.json`: 상세 분석 결과

## 예상 성능

BERT-base 모델로 QNLI에서 기대할 수 있는 성능:
- **정확도**: ~90-92%
- **F1 점수**: ~90-92%

## SST-2와의 차이점

1. **입력 형식**: 
   - SST-2: 단일 문장
   - QNLI: 질문-문단 쌍 ([CLS] question [SEP] sentence [SEP])

2. **시퀀스 길이**: 
   - SST-2: 128 토큰
   - QNLI: 512 토큰 (질문+문단이므로 더 길어짐)

3. **배치 크기**: 
   - SST-2: 32
   - QNLI: 16 (메모리 제약으로 인해 더 작게 설정)

4. **태스크 특성**:
   - SST-2: 감정 분석 (긍정/부정)
   - QNLI: 자연어 추론 (함의/비함의)

## 하이퍼파라미터 튜닝

`config.yaml` 파일에서 다양한 하이퍼파라미터 조합을 확인할 수 있습니다:

```yaml
alternatives:
  learning_rates: [1e-5, 2e-5, 3e-5, 5e-5]
  batch_sizes: [8, 16, 32]
  num_epochs: [2, 3, 4, 5]
  warmup_ratios: [0.06, 0.1, 0.2]
  max_lengths: [256, 384, 512]
```

## 문제 해결

### 1. 메모리 부족 오류
- 배치 크기를 줄여보세요: `--batch_size 8`
- 시퀀스 길이를 줄여보세요: `--max_length 384`
- 그래디언트 체크포인팅 사용 고려

### 2. 모델 파일을 찾을 수 없음
- 모델 경로가 올바른지 확인: `--model_path path/to/your/model.pth`

### 3. 데이터 다운로드 실패
- 수동으로 QNLI 데이터를 다운로드하여 `./data/QNLI/` 디렉토리에 배치

### 4. 훈련 시간이 너무 오래 걸림
- 더 작은 배치 크기나 더 짧은 시퀀스 길이 사용
- GPU 사용 확인
- 조기 종료 설정 조정: `--early_stopping_patience 2`

## 고급 기능

### Weights & Biases 연동
```bash
python run_finetuning.py --use_wandb --wandb_project my-qnli-project
```

### 조기 종료
```bash
python run_finetuning.py --early_stopping_patience 3
```

### 그래디언트 클리핑
```bash
python run_finetuning.py --max_grad_norm 1.0
```

## 성능 최적화 팁

1. **시퀀스 길이 최적화**: 대부분의 질문-문단 쌍이 512 토큰보다 짧다면 384로 줄여서 속도 향상
2. **배치 크기 조정**: GPU 메모리에 맞게 최대한 크게 설정
3. **학습률 스케줄링**: 웜업 비율을 조정하여 안정적인 훈련
4. **조기 종료**: 과적합 방지를 위해 적절한 patience 설정

## 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다.

## 참고 자료

- [BERT 논문](https://arxiv.org/abs/1810.04805)
- [GLUE 벤치마크](https://gluebenchmark.com/)
- [QNLI 데이터셋](https://rajpurkar.github.io/SQuAD-explorer/)
- [자연어 추론 태스크 소개](https://nlp.stanford.edu/projects/snli/)