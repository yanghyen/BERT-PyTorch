"""
CoLA (Corpus of Linguistic Acceptability) 파인튜닝 모듈

이 모듈은 BERT 모델을 CoLA 태스크에 파인튜닝하기 위한 
데이터셋, 모델, 훈련, 평가 기능을 제공합니다.

CoLA는 문법적 수용성을 판단하는 이진 분류 태스크로,
Matthews 상관계수가 주요 평가 지표입니다.

주요 구성요소:
- dataset.py: CoLA 데이터셋 로딩 및 전처리
- finetuning_model.py: BERT 기반 분류 모델
- train.py: 파인튜닝 훈련 로직
- evaluate.py: 모델 평가 및 메트릭 계산
- run_finetuning.py: 메인 실행 스크립트
"""

from .dataset import CoLADataset, create_data_loaders, get_label_names, get_num_labels
from .finetuning_model import BERTForSequenceClassification, create_classification_model
from .train import BERTTrainer
from .evaluate import evaluate_model, compute_metrics, analyze_predictions

__version__ = "1.0.0"
__author__ = "BERT CoLA Finetuning Team"

__all__ = [
    'CoLADataset',
    'create_data_loaders', 
    'get_label_names',
    'get_num_labels',
    'BERTForSequenceClassification',
    'create_classification_model',
    'BERTTrainer',
    'evaluate_model',
    'compute_metrics',
    'analyze_predictions'
]