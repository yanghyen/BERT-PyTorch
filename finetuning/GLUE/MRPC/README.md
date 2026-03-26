# BERT MRPC 파인튜닝

이 프로젝트는 사전 훈련된 BERT 모델을 Microsoft Research Paraphrase Corpus (MRPC) 데이터셋으로 파인튜닝하는 코드입니다.

## 프로젝트 구조

```
finetuning/GLUE/MRPC/
├── dataset.py          # MRPC 데이터셋 로딩 및 전처리
├── finetuning_model.py # BERT 분류 모델 정의 (2-class)
├── train.py           # 훈련 로직
├── evaluate.py        # 평가 및 분석
├── run_finetuning.py  # 메인 실행 스크립트
├── config.yaml        # 설정 파일
├── requirements.txt   # 필요 패키지
└── README.md         # 이 파일
```

## MRPC 태스크 개요

MRPC (Microsoft Research Paraphrase Corpus)는 두 문장이 의미적으로 동등한지(paraphrase) 판단하는 이진 분류 태스크입니다.

### 클래스
- **Not Paraphrase (0)**: 두 문장이 의미적으로 다름
- **Paraphrase (1)**: 두 문장이 의미적으로 동등함

### 예시
- **문장1**: "The DVD-CCA then appealed to the state Supreme Court."
- **문장2**: "The DVD CCA appealed that decision to the U.S. Supreme Court."
- **레이블**: Not Paraphrase (0)

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

# 더 많은 에포크로 훈련 (MRPC는 작은 데이터셋)
python run_finetuning.py --num_epochs 10

# F1 점수 기준으로 최고 모델 선택
python run_finetuning.py --use_f1_for_best_model

# 클래스 불균형 처리
python run_finetuning.py --use_class_weights

# 훈련만 실행
python run_finetuning.py --mode train

# 평가만 실행
python run_finetuning.py --mode eval --checkpoint_path ./results/MRPC/checkpoints/seed42_20240324_185530/best_model.pt
```

### 3. 데이터셋 분석

```bash
# 데이터셋 통계 분석
python run_finetuning.py --analyze_dataset
```

### 4. 다른 config 파일 사용

```bash
python run_finetuning.py --config my_mrpc_config.yaml
```

### 5. config.yaml 수정

주요 설정들을 `config.yaml`에서 직접 수정:

```yaml
# 모델 경로 변경
model:
  model_path: "/path/to/your/bert/model.pth"

# 데이터 설정 (MRPC 특화)
data:
  max_length: 128  # MRPC는 상대적으로 짧은 문장
  batch_size: 32   # 작은 데이터셋이므로 큰 배치 가능

# 훈련 설정 조정
training:
  learning_rate: 2.0e-5
  num_epochs: 5    # 작은 데이터셋이므로 더 많은 에포크

# MRPC 특화 설정
mrpc_specific:
  use_f1_for_best_model: true  # F1 점수 기준 모델 선택
  class_weights: null          # 클래스 불균형 처리
```

## 주요 매개변수

### 모델 관련
- `--model_path`: 사전 훈련된 BERT 모델 경로
- `--hidden_size`: BERT 숨겨진 차원 크기 (기본값: 768)
- `--num_layers`: BERT 레이어 수 (기본값: 12)
- `--num_attention_heads`: 어텐션 헤드 수 (기본값: 12)

### 데이터 관련
- `--data_dir`: MRPC 데이터 디렉토리 (기본값: ./data/MRPC)
- `--max_length`: 최대 시퀀스 길이 (기본값: 128)
- `--batch_size`: 배치 크기 (기본값: 32)

### 훈련 관련
- `--learning_rate`: 학습률 (기본값: 2e-5)
- `--num_epochs`: 에포크 수 (기본값: 5, MRPC는 작은 데이터셋)
- `--weight_decay`: 가중치 감쇠 (기본값: 0.01)
- `--warmup_ratio`: 웜업 비율 (기본값: 0.1)

### MRPC 특화
- `--use_f1_for_best_model`: F1 점수를 기준으로 최고 모델 선택
- `--use_class_weights`: 클래스 불균형 처리를 위한 가중치 사용
- `--analyze_dataset`: 데이터셋 통계 분석 실행

## 데이터셋

MRPC (Microsoft Research Paraphrase Corpus) 데이터셋은 뉴스 기사에서 추출한 문장 쌍의 의미적 동등성을 판단하는 태스크입니다.

- **훈련 데이터**: 3,668개 샘플
- **검증 데이터**: 408개 샘플  
- **테스트 데이터**: 1,725개 샘플
- **클래스**: 2개 (not_paraphrase: 0, paraphrase: 1)

### 데이터셋 특성
- **작은 크기**: GLUE 태스크 중 가장 작은 데이터셋
- **클래스 불균형**: Paraphrase 클래스가 상대적으로 적음
- **도전적**: 의미적 유사성 판단이 어려운 경우가 많음

데이터는 첫 실행 시 자동으로 다운로드됩니다.

## 결과

### 성능 메트릭
- **정확도 (Accuracy)**
- **정밀도 (Precision)**: Paraphrase 클래스 기준
- **재현율 (Recall)**: Paraphrase 클래스 기준
- **F1 점수**: Paraphrase 클래스 기준 (주요 메트릭)
- **클래스별 성능**

### 출력 파일

실행할 때마다 seed와 타임스탬프 기반으로 고유한 디렉토리가 생성됩니다:

- `./results/MRPC/checkpoints/seed42_20240324_185530/`: 훈련된 모델 체크포인트
- `./results/MRPC/logs/seed42_20240324_185530/`: 훈련 로그
- `./results/MRPC/evaluation/seed42_20240324_185530/`: 평가 결과 및 분석
  - `confusion_matrix.png`: 혼동 행렬
  - `class_performance.png`: 클래스별 성능
  - `confidence_distribution.png`: 신뢰도 분포
  - `paraphrase_analysis.png`: Paraphrase 특화 분석
  - `analysis.json`: 상세 분석 결과

## 예상 성능

BERT-base 모델로 MRPC에서 기대할 수 있는 성능:
- **정확도**: ~88-90%
- **F1 점수**: ~83-85%
- **정밀도**: ~80-85%
- **재현율**: ~85-90%

MRPC는 작은 데이터셋이므로 성능 변동이 클 수 있습니다.

## 하이퍼파라미터 튜닝

`config.yaml` 파일에서 다양한 하이퍼파라미터 조합을 확인할 수 있습니다:

```yaml
alternatives:
  learning_rates: [1e-5, 2e-5, 3e-5, 5e-5]
  batch_sizes: [16, 32, 64]  # 작은 데이터셋이므로 큰 배치도 가능
  num_epochs: [3, 5, 7, 10]  # 작은 데이터셋 특성상 더 많은 에포크
  warmup_ratios: [0.06, 0.1, 0.2]
  max_lengths: [64, 128, 256]
```

## 문제 해결

### 1. 성능이 낮은 경우
- 더 많은 에포크로 훈련: `--num_epochs 10`
- 다른 학습률 시도: `--learning_rate 3e-5`
- 클래스 가중치 사용: `--use_class_weights`
- F1 점수 기준 모델 선택: `--use_f1_for_best_model`

### 2. 과적합 문제
- 조기 종료 patience 줄이기
- 더 강한 정규화 적용
- 드롭아웃 증가

### 3. 모델 파일을 찾을 수 없음
- 모델 경로가 올바른지 확인: `--model_path path/to/your/model.pth`

### 4. 데이터 다운로드 실패

자동 다운로드가 실패하는 경우 여러 방법을 시도합니다:

**방법 1: HuggingFace datasets 사용 (권장)**
```bash
pip install datasets
python test_download.py
```

**방법 2: 수동 다운로드**
- [Microsoft 공식 사이트](https://www.microsoft.com/en-us/download/details.aspx?id=52398)에서 다운로드
- `./data/MRPC/` 디렉토리에 배치
- 필요한 파일: `train.tsv`, `dev.tsv`, `test.tsv`

**방법 3: 다운로드 테스트**
```bash
python test_download.py  # 다운로드 테스트 및 디버깅
```

### 5. 클래스 불균형 문제
- 클래스 가중치 사용: `--use_class_weights`
- F1 점수 기준 평가: `--use_f1_for_best_model`

## 고급 기능

### Weights & Biases 연동
```bash
python run_finetuning.py --use_wandb --wandb_project my-bert-mrpc-project
```

### 조기 종료
```bash
python run_finetuning.py --early_stopping_patience 5
```

### 그래디언트 클리핑
```bash
python run_finetuning.py --max_grad_norm 1.0
```

### 클래스 불균형 처리
```bash
python run_finetuning.py --use_class_weights
```

### 데이터셋 분석
```bash
python run_finetuning.py --analyze_dataset
```

## MRPC vs 다른 GLUE 태스크 비교

| 특성 | SST-2 | MNLI | MRPC |
|------|-------|------|------|
| 태스크 | 감정 분석 | 자연어 추론 | 의미적 동등성 |
| 입력 | 단일 문장 | 두 문장 | 두 문장 |
| 클래스 수 | 2개 | 3개 | 2개 |
| 데이터 크기 | 67K | 393K | 3.7K |
| 시퀀스 길이 | 128 | 256 | 128 |
| 주요 메트릭 | Accuracy | Accuracy | F1 Score |
| 에포크 수 | 3 | 3 | 5-10 |

## 성능 최적화 팁

1. **작은 데이터셋 특성 활용**:
   - 더 많은 에포크 훈련
   - 큰 배치 크기 사용 가능
   - 더 자주 검증 수행

2. **클래스 불균형 대응**:
   - 클래스 가중치 사용
   - F1 점수 기준 평가
   - 정밀도-재현율 균형 고려

3. **하이퍼파라미터 튜닝**:
   - 학습률 세밀 조정
   - 웜업 비율 최적화
   - 드롭아웃 조정

## 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다.

## 참고 자료

- [BERT 논문](https://arxiv.org/abs/1810.04805)
- [GLUE 벤치마크](https://gluebenchmark.com/)
- [MRPC 데이터셋](https://www.microsoft.com/en-us/download/details.aspx?id=52398)
- [Paraphrase Detection 개요](https://aclanthology.org/I05-5002.pdf)