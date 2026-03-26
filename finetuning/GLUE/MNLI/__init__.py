# BERT MNLI 파인튜닝 패키지

"""
BERT MNLI (Multi-Genre Natural Language Inference) 파인튜닝 패키지

이 패키지는 사전 훈련된 BERT 모델을 MNLI 태스크에 파인튜닝하기 위한 
모든 필요한 구성 요소를 제공합니다.

주요 모듈:
- dataset: MNLI 데이터셋 로딩 및 전처리
- finetuning_model: BERT 기반 3-class 분류 모델
- train: 훈련 로직 및 트레이너
- evaluate: 평가 및 분석 도구
- run_finetuning: 메인 실행 스크립트

사용 예시:
    python run_finetuning.py --config config.yaml
"""

__version__ = "1.0.0"
__author__ = "BERT MNLI Team"

# 주요 클래스 및 함수 임포트
from .dataset import (
    MNLIDataset,
    create_data_loaders,
    create_matched_mismatched_loaders,
    get_label_names,
    get_num_labels
)

from .finetuning_model import (
    BERTForSequenceClassification,
    load_pretrained_bert,
    create_classification_model
)

from .train import BERTTrainer

from .evaluate import (
    compute_metrics,
    evaluate_model,
    evaluate_matched_mismatched,
    analyze_predictions,
    plot_confusion_matrix,
    plot_class_performance,
    plot_confidence_distribution
)

__all__ = [
    # Dataset
    'MNLIDataset',
    'create_data_loaders',
    'create_matched_mismatched_loaders',
    'get_label_names',
    'get_num_labels',
    
    # Model
    'BERTForSequenceClassification',
    'load_pretrained_bert',
    'create_classification_model',
    
    # Training
    'BERTTrainer',
    
    # Evaluation
    'compute_metrics',
    'evaluate_model',
    'evaluate_matched_mismatched',
    'analyze_predictions',
    'plot_confusion_matrix',
    'plot_class_performance',
    'plot_confidence_distribution'
]