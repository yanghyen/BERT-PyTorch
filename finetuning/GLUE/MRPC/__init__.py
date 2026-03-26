# BERT MRPC 파인튜닝 패키지

"""
BERT MRPC (Microsoft Research Paraphrase Corpus) 파인튜닝 패키지

이 패키지는 사전 훈련된 BERT 모델을 MRPC 태스크에 파인튜닝하기 위한 
모든 필요한 구성 요소를 제공합니다.

주요 모듈:
- dataset: MRPC 데이터셋 로딩 및 전처리
- finetuning_model: BERT 기반 2-class 분류 모델 (클래스 가중치 지원)
- train: 훈련 로직 및 트레이너 (F1 기준 모델 선택 지원)
- evaluate: 평가 및 분석 도구 (Paraphrase 특화 분석)
- run_finetuning: 메인 실행 스크립트

사용 예시:
    python run_finetuning.py --config config.yaml --use_f1_for_best_model
"""

__version__ = "1.0.0"
__author__ = "BERT MRPC Team"

# 주요 클래스 및 함수 임포트
from .dataset import (
    MRPCDataset,
    create_data_loaders,
    get_label_names,
    get_num_labels,
    analyze_dataset_statistics
)

from .finetuning_model import (
    BERTForSequenceClassification,
    load_pretrained_bert,
    create_classification_model,
    calculate_class_weights
)

from .train import BERTTrainer

from .evaluate import (
    compute_metrics,
    evaluate_model,
    analyze_predictions,
    analyze_paraphrase_patterns,
    plot_confusion_matrix,
    plot_class_performance,
    plot_confidence_distribution,
    plot_paraphrase_analysis
)

__all__ = [
    # Dataset
    'MRPCDataset',
    'create_data_loaders',
    'get_label_names',
    'get_num_labels',
    'analyze_dataset_statistics',
    
    # Model
    'BERTForSequenceClassification',
    'load_pretrained_bert',
    'create_classification_model',
    'calculate_class_weights',
    
    # Training
    'BERTTrainer',
    
    # Evaluation
    'compute_metrics',
    'evaluate_model',
    'analyze_predictions',
    'analyze_paraphrase_patterns',
    'plot_confusion_matrix',
    'plot_class_performance',
    'plot_confidence_distribution',
    'plot_paraphrase_analysis'
]