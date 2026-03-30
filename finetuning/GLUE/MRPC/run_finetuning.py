#!/usr/bin/env python3
"""
BERT MRPC 파인튜닝 메인 실행 스크립트
"""

import os
import sys
import argparse
import json
import yaml
import torch
import numpy as np
import random
from datetime import datetime
import logging

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import create_data_loaders, get_num_labels, analyze_dataset_statistics
from finetuning_model import create_classification_model, calculate_class_weights
from train import BERTTrainer
from evaluate import evaluate_model, plot_confusion_matrix, analyze_predictions, plot_paraphrase_analysis
from sklearn.metrics import confusion_matrix


def set_seed(seed: int):
    """재현 가능한 결과를 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 결정적 동작 설정 (성능이 약간 저하될 수 있음)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str):
    """로깅 설정"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'finetuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml"):
    """YAML 설정 파일 로드"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def config_to_args(config: dict):
    """config dict를 argparse.Namespace로 변환"""
    args = argparse.Namespace()
    
    # 모델 설정
    model_config = config.get('model', {})
    args.model_path = model_config.get('model_path', '/home/ssai/Workspace/BERT_repo/runs/L12_H768_A12_seed42/model_full.pth')
    args.vocab_size = model_config.get('vocab_size', 30522)
    args.hidden_size = model_config.get('hidden_size', 768)
    args.num_layers = model_config.get('num_layers', 12)
    args.num_attention_heads = model_config.get('num_attention_heads', 12)
    args.dropout = model_config.get('dropout', 0.1)
    
    # 데이터 설정
    data_config = config.get('data', {})
    args.data_dir = data_config.get('data_dir', './data/MRPC')
    args.max_length = data_config.get('max_length', 128)
    args.batch_size = data_config.get('batch_size', 32)
    args.num_workers = data_config.get('num_workers', 4)
    
    # 훈련 설정
    training_config = config.get('training', {})
    args.learning_rate = training_config.get('learning_rate', 2e-5)
    args.weight_decay = training_config.get('weight_decay', 0.01)
    args.num_epochs = training_config.get('num_epochs', 5)
    args.warmup_ratio = training_config.get('warmup_ratio', 0.1)
    args.max_grad_norm = training_config.get('max_grad_norm', 1.0)
    args.early_stopping_patience = training_config.get('early_stopping_patience', 5)
    
    # 디렉토리 설정
    dir_config = config.get('directories', {})
    args.save_dir = dir_config.get('save_dir', './finetuning_results/MRPC/checkpoints')
    args.log_dir = dir_config.get('log_dir', './finetuning_results/MRPC/logs')
    args.eval_dir = dir_config.get('eval_dir', './finetuning_results/MRPC/evaluation')
    
    # 기타 설정
    misc_config = config.get('misc', {})
    args.seed = misc_config.get('seed', 42)
    args.mode = misc_config.get('mode', 'both')
    args.checkpoint_path = misc_config.get('checkpoint_path')
    args.use_wandb = misc_config.get('use_wandb', False)
    args.wandb_project = misc_config.get('wandb_project', 'bert-mrpc-finetuning')
    
    # MRPC 특화 설정
    mrpc_config = config.get('mrpc_specific', {})
    args.use_f1_for_best_model = mrpc_config.get('use_f1_for_best_model', True)
    args.use_class_weights = mrpc_config.get('class_weights') is not None
    args.class_weights_config = mrpc_config.get('class_weights')
    
    return args


def parse_args():
    """명령행 인자 파싱 (config 오버라이드용)"""
    parser = argparse.ArgumentParser(description='BERT MRPC 파인튜닝')
    
    # 설정 파일: 스크립트 위치 기준으로 config.yaml을 찾아 어느 디렉토리에서 실행해도 동일하게 동작합니다.
    _default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
    parser.add_argument('--config', type=str, default=_default_config,
                       help='YAML 설정 파일 경로')
    
    # 주요 오버라이드 옵션들
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'both'],
                       help='실행 모드 (config 오버라이드)')
    parser.add_argument('--seed', type=int,
                       help='랜덤 시드 (config 오버라이드)')
    parser.add_argument('--batch_size', type=int,
                       help='배치 크기 (config 오버라이드)')
    parser.add_argument('--learning_rate', type=float,
                       help='학습률 (config 오버라이드)')
    parser.add_argument('--num_epochs', type=int,
                       help='에포크 수 (config 오버라이드)')
    parser.add_argument('--model_path', type=str,
                       help='모델 경로 (config 오버라이드)')
    parser.add_argument('--checkpoint_path', type=str,
                       help='평가할 체크포인트 경로 (eval 모드용)')
    parser.add_argument('--max_length', type=int,
                       help='최대 시퀀스 길이 (config 오버라이드)')
    parser.add_argument('--use_f1_for_best_model', action='store_true',
                       help='F1 점수를 기준으로 최고 모델 선택')
    parser.add_argument('--use_class_weights', action='store_true',
                       help='클래스 불균형 처리를 위한 가중치 사용')
    parser.add_argument('--analyze_dataset', action='store_true',
                       help='데이터셋 통계 분석 실행')
    
    return parser.parse_args()


def train_model(args, logger):
    """모델 훈련"""
    logger.info("=" * 50)
    logger.info("BERT MRPC 파인튜닝 훈련 시작")
    logger.info("=" * 50)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"사용 디바이스: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU 정보: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 데이터 로더 생성
    logger.info("데이터 로더 생성 중...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers
    )
    
    logger.info(f"훈련 샘플 수: {len(train_loader.dataset)}")
    logger.info(f"검증 샘플 수: {len(val_loader.dataset)}")
    logger.info(f"테스트 샘플 수: {len(test_loader.dataset)}")
    
    # 클래스 가중치 계산 (옵션)
    class_weights = None
    if args.use_class_weights:
        logger.info("클래스 가중치 계산 중...")
        train_labels = []
        for batch in train_loader:
            train_labels.extend(batch['labels'].tolist())
        class_weights = calculate_class_weights(train_labels, device)
        logger.info(f"클래스 가중치: {class_weights}")
    
    # 모델 생성
    logger.info("모델 생성 중...")
    model = create_classification_model(
        model_path=args.model_path,
        num_labels=get_num_labels(),  # MRPC: 2개 클래스
        vocab_size=args.vocab_size,
        hidden=args.hidden_size,
        n_layers=args.num_layers,
        attn_heads=args.num_attention_heads,
        dropout=args.dropout,
        class_weights=class_weights
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"전체 파라미터 수: {total_params:,}")
    logger.info(f"훈련 가능한 파라미터 수: {trainable_params:,}")
    
    # 트레이너 생성
    warmup_steps = int(args.warmup_ratio * len(train_loader) * args.num_epochs)
    logger.info(f"웜업 스텝 수: {warmup_steps}")
    logger.info(f"최고 모델 선택 기준: {'F1' if args.use_f1_for_best_model else 'Accuracy'}")
    
    trainer = BERTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        max_grad_norm=args.max_grad_norm,
        device=device,
        save_dir=args.save_dir,
        use_wandb=args.use_wandb,
        project_name=args.wandb_project,
        use_f1_for_best_model=args.use_f1_for_best_model
    )
    
    # 훈련 실행
    logger.info("훈련 시작...")
    start_time = datetime.now()
    
    history = trainer.train(
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience
    )
    
    end_time = datetime.now()
    training_time = end_time - start_time
    
    logger.info("훈련 완료!")
    logger.info(f"총 훈련 시간: {training_time}")
    logger.info(f"최고 검증 정확도: {max(history['val_accuracy']):.4f}")
    logger.info(f"최고 검증 F1: {max(history['val_f1']):.4f}")
    logger.info(f"최고 검증 정밀도: {max(history['val_precision']):.4f}")
    logger.info(f"최고 검증 재현율: {max(history['val_recall']):.4f}")
    
    # 훈련 결과 저장
    results = {
        'args': vars(args),
        'training_time': str(training_time),
        'best_val_accuracy': max(history['val_accuracy']),
        'best_val_f1': max(history['val_f1']),
        'best_val_precision': max(history['val_precision']),
        'best_val_recall': max(history['val_recall']),
        'history': history
    }
    
    results_path = os.path.join(args.save_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return os.path.join(args.save_dir, 'best_model.pt')


def evaluate_model_main(args, logger, checkpoint_path=None):
    """모델 평가"""
    logger.info("=" * 50)
    logger.info("BERT MRPC 모델 평가 시작")
    logger.info("=" * 50)
    
    if checkpoint_path is None:
        checkpoint_path = args.checkpoint_path or os.path.join(args.save_dir, 'best_model.pt')
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        return
    
    logger.info(f"체크포인트 로드: {checkpoint_path}")
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"사용 디바이스: {device}")
    
    # 데이터 로더 생성
    logger.info("데이터 로더 생성 중...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers
    )
    
    # 모델 생성
    logger.info("모델 생성 중...")
    model = create_classification_model(
        model_path=args.model_path,
        num_labels=get_num_labels(),
        vocab_size=args.vocab_size,
        hidden=args.hidden_size,
        n_layers=args.num_layers,
        attn_heads=args.num_attention_heads,
        dropout=args.dropout
    )
    
    # 체크포인트에서 파인튜닝된 가중치 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("파인튜닝된 가중치 로드 완료")
    
    # 평가 디렉토리 생성
    os.makedirs(args.eval_dir, exist_ok=True)
    
    # 검증 데이터 평가
    logger.info("검증 데이터 평가 중...")
    val_results = evaluate_model(model, val_loader, device)
    
    if 'metrics' in val_results:
        metrics = val_results['metrics']
        logger.info("검증 데이터 결과:")
        logger.info(f"  정확도: {metrics['accuracy']:.4f}")
        logger.info(f"  정밀도: {metrics['precision']:.4f}")
        logger.info(f"  재현율: {metrics['recall']:.4f}")
        logger.info(f"  F1 점수: {metrics['f1']:.4f}")
        logger.info(f"  손실: {val_results['loss']:.4f}")
    
    # 테스트 데이터 평가 (레이블이 있는 경우)
    logger.info("테스트 데이터 평가 중...")
    test_results = evaluate_model(model, test_loader, device)
    logger.info(f"테스트 데이터 예측 수: {len(test_results['predictions'])}")
    
    # 시각화 및 분석
    if 'metrics' in val_results:
        logger.info("결과 분석 및 시각화 중...")
        
        # 혼동 행렬
        cm = confusion_matrix(val_results['labels'], val_results['predictions'])
        plot_confusion_matrix(
            cm,
            ['not_paraphrase', 'paraphrase'],
            os.path.join(args.eval_dir, 'confusion_matrix.png')
        )
        
        # 상세 분석
        analysis = analyze_predictions(
            val_results['predictions'],
            val_results['labels'],
            val_results['probabilities'],
            save_dir=args.eval_dir
        )
        
        # Paraphrase 특화 분석 시각화
        plot_paraphrase_analysis(
            analysis['paraphrase_analysis'],
            os.path.join(args.eval_dir, 'paraphrase_analysis.png')
        )
        
        logger.info("분석 결과:")
        logger.info(f"  평균 신뢰도: {analysis['average_confidence']:.4f}")
        logger.info(f"  정답 신뢰도: {analysis['correct_confidence']:.4f}")
        logger.info(f"  오답 신뢰도: {analysis['wrong_confidence']:.4f}")
        
        # Paraphrase 특화 분석 결과
        paraphrase_stats = analysis['paraphrase_analysis']['paraphrase_confidence_stats']
        logger.info("Paraphrase 분석:")
        logger.info(f"  실제 Paraphrase 평균 신뢰도: {paraphrase_stats['true_paraphrase_avg_conf']:.4f}")
        logger.info(f"  실제 Not Paraphrase 평균 신뢰도: {paraphrase_stats['false_paraphrase_avg_conf']:.4f}")
    
    # 평가 결과 저장
    eval_results = {
        'checkpoint_path': checkpoint_path,
        'val_results': {
            'metrics': val_results.get('metrics', {}),
            'loss': val_results.get('loss', 0.0)
        },
        'test_predictions_count': len(test_results['predictions']),
        'args': vars(args)
    }
    
    results_path = os.path.join(args.eval_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    logger.info(f"평가 완료! 결과가 {args.eval_dir}에 저장되었습니다.")


def main():
    """메인 함수"""
    # 명령행 인자 파싱
    cmd_args = parse_args()
    
    # 데이터셋 분석 (옵션)
    if cmd_args.analyze_dataset:
        print("MRPC 데이터셋 분석 실행 중...")
        try:
            analyze_dataset_statistics(cmd_args.data_dir if hasattr(cmd_args, 'data_dir') else './data/MRPC')
        except Exception as e:
            print(f"데이터셋 분석 실패: {e}")
        return
    
    # config 파일 로드
    try:
        config = load_config(cmd_args.config)
        print(f"설정 파일 로드 완료: {cmd_args.config}")
        print(f"로드된 모델 경로: {config.get('model', {}).get('model_path', 'None')}")
    except FileNotFoundError as e:
        print(f"경고: {e}")
        print("기본 설정으로 실행합니다.")
        config = {}
    
    # config를 args로 변환
    args = config_to_args(config)
    print(f"변환된 모델 경로: {args.model_path}")
    
    # 명령행 인자로 오버라이드
    for key, value in vars(cmd_args).items():
        if value is not None and key != 'config':
            setattr(args, key, value)
    
    # 시드 설정
    set_seed(args.seed)
    
    # seed 기반 실행 ID로 디렉토리 구조 업데이트
    # 특정 설정 파일(config_large/config_no_nsp)인 경우 BERT 모드를 경로에 포함
    config_name = os.path.basename(cmd_args.config)
    mode_suffix = ""
    if config_name == "config_large.yaml":
        mode_suffix = "_large"
    elif config_name == "config_no_nsp.yaml":
        mode_suffix = "_no_nsp"

    run_id = f"seed{args.seed}{mode_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.save_dir = os.path.join(args.save_dir, run_id)
    args.log_dir = os.path.join(args.log_dir, run_id)
    args.eval_dir = os.path.join(args.eval_dir, run_id)
    
    # 로깅 설정
    logger = setup_logging(args.log_dir)
    
    # 사용된 설정 출력
    logger.info(f"설정 파일: {cmd_args.config}")
    logger.info(f"실행 ID: {run_id}")
    
    # 설정 출력
    logger.info("실행 설정:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    
    # 실행 모드에 따른 처리
    if args.mode == 'train':
        checkpoint_path = train_model(args, logger)
        
    elif args.mode == 'eval':
        evaluate_model_main(args, logger)
        
    elif args.mode == 'both':
        # 훈련 후 평가
        checkpoint_path = train_model(args, logger)
        logger.info("\n" + "=" * 50)
        logger.info("훈련 완료, 평가 시작")
        logger.info("=" * 50)
        evaluate_model_main(args, logger, checkpoint_path)
    
    logger.info("모든 작업 완료!")


if __name__ == "__main__":
    main()