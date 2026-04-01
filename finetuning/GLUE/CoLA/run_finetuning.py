#!/usr/bin/env python3
"""
BERT CoLA 파인튜닝 메인 실행 스크립트
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

from dataset import create_data_loaders, get_num_labels
from finetuning_model import create_classification_model
from train import BERTTrainer
from evaluate import evaluate_model, plot_confusion_matrix, analyze_predictions
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


def load_config(config_path: str = "config_large.yaml"):
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
    args.model_path = model_config.get('model_path', '/home/ssai/Workspace/BERT_repo/runs/L24_H1024_A16_seed42/model_full.pth')
    args.vocab_size = model_config.get('vocab_size', 30522)
    args.hidden_size = model_config.get('hidden_size', 1024)
    args.num_layers = model_config.get('num_layers', 24)
    args.num_attention_heads = model_config.get('num_attention_heads', 16)
    args.dropout = model_config.get('dropout', 0.1)
    
    # 데이터 설정
    data_config = config.get('data', {})
    args.data_dir = data_config.get('data_dir', './data/CoLA')
    args.max_length = data_config.get('max_length', 128)
    args.batch_size = data_config.get('batch_size', 16)
    args.num_workers = data_config.get('num_workers', 4)
    args.tokenizer_name = data_config.get('tokenizer_name', 'bert-base-uncased')
    
    # 훈련 설정
    training_config = config.get('training', {})
    args.learning_rate = training_config.get('learning_rate', 1e-5)
    args.weight_decay = training_config.get('weight_decay', 0.01)
    args.num_epochs = training_config.get('num_epochs', 5)
    args.warmup_ratio = training_config.get('warmup_ratio', 0.1)
    args.max_grad_norm = training_config.get('max_grad_norm', 1.0)
    args.early_stopping_patience = training_config.get('early_stopping_patience', 5)
    
    # 평가 설정
    eval_config = config.get('evaluation', {})
    args.eval_steps = eval_config.get('eval_steps', 100)
    args.save_steps = eval_config.get('save_steps', 200)
    
    # 디렉토리 설정
    dir_config = config.get('directories', {})
    args.save_dir = dir_config.get('save_dir', './finetuning_results/CoLA/checkpoints')
    args.log_dir = dir_config.get('log_dir', './finetuning_results/CoLA/logs')
    args.eval_dir = dir_config.get('eval_dir', './finetuning_results/CoLA/evaluation')
    
    # 기타 설정
    misc_config = config.get('misc', {})
    args.seed = misc_config.get('seed', 456)
    args.mode = misc_config.get('mode', 'both')
    args.use_wandb = misc_config.get('use_wandb', False)
    args.wandb_project = misc_config.get('wandb_project', 'bert-large-cola-finetuning')
    args.checkpoint_path = misc_config.get('checkpoint_path', None)
    
    return args


def train_model(args, logger):
    """모델 훈련"""
    logger.info("=== CoLA 파인튜닝 훈련 시작 ===")
    
    # 디렉토리 생성
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.eval_dir, exist_ok=True)
    
    # 데이터 로더 생성
    logger.info("데이터 로더 생성 중...")
    train_loader, dev_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        tokenizer_name=args.tokenizer_name,
        num_workers=args.num_workers
    )
    
    logger.info(f"훈련 데이터: {len(train_loader.dataset)}개")
    logger.info(f"검증 데이터: {len(dev_loader.dataset)}개")
    logger.info(f"테스트 데이터: {len(test_loader.dataset)}개")
    
    # 모델 생성
    logger.info("BERT 분류 모델 생성 중...")
    model = create_classification_model(
        model_path=args.model_path,
        num_labels=get_num_labels(),
        vocab_size=args.vocab_size,
        hidden=args.hidden_size,
        n_layers=args.num_layers,
        attn_heads=args.num_attention_heads,
        dropout=args.dropout
    )
    
    # GPU 사용 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"사용 디바이스: {device}")
    model = model.to(device)
    
    # 트레이너 생성
    trainer = BERTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=dev_loader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        save_dir=args.save_dir,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        early_stopping_patience=args.early_stopping_patience,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        device=device
    )
    
    # 훈련 실행
    logger.info("훈련 시작...")
    best_model_path = trainer.train()
    logger.info(f"훈련 완료! 최고 모델: {best_model_path}")
    
    return best_model_path


def evaluate_model_performance(args, model_path, logger):
    """모델 성능 평가"""
    logger.info("=== CoLA 모델 성능 평가 ===")
    
    # 데이터 로더 생성
    train_loader, dev_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        tokenizer_name=args.tokenizer_name,
        num_workers=args.num_workers
    )
    
    # 모델 로드
    logger.info(f"모델 로드 중: {model_path}")
    model = create_classification_model(
        model_path=args.model_path,
        num_labels=get_num_labels(),
        vocab_size=args.vocab_size,
        hidden=args.hidden_size,
        n_layers=args.num_layers,
        attn_heads=args.num_attention_heads,
        dropout=args.dropout
    )
    
    # 파인튜닝된 가중치 로드
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 평가 실행
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join(args.eval_dir, f'seed{args.seed}_{timestamp}')
    os.makedirs(eval_dir, exist_ok=True)
    
    results = {}
    
    # 검증 데이터 평가
    logger.info("검증 데이터 평가 중...")
    dev_metrics = evaluate_model(model, dev_loader, device)
    results['dev'] = dev_metrics
    
    logger.info("검증 결과:")
    logger.info(f"  정확도: {dev_metrics['accuracy']:.4f}")
    logger.info(f"  정밀도: {dev_metrics['precision']:.4f}")
    logger.info(f"  재현율: {dev_metrics['recall']:.4f}")
    logger.info(f"  F1 점수: {dev_metrics['f1']:.4f}")
    logger.info(f"  Matthews 상관계수: {dev_metrics['matthews_corr']:.4f}")
    
    # 테스트 데이터 평가 (레이블이 있는 경우)
    if hasattr(test_loader.dataset, 'labels') and test_loader.dataset.labels[0] != -1:
        logger.info("테스트 데이터 평가 중...")
        test_metrics = evaluate_model(model, test_loader, device)
        results['test'] = test_metrics
        
        logger.info("테스트 결과:")
        logger.info(f"  정확도: {test_metrics['accuracy']:.4f}")
        logger.info(f"  정밀도: {test_metrics['precision']:.4f}")
        logger.info(f"  재현율: {test_metrics['recall']:.4f}")
        logger.info(f"  F1 점수: {test_metrics['f1']:.4f}")
        logger.info(f"  Matthews 상관계수: {test_metrics['matthews_corr']:.4f}")
    
    # 결과 저장
    results_file = os.path.join(eval_dir, 'evaluation_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 혼동 행렬 및 예측 분석
    logger.info("예측 분석 중...")
    predictions, true_labels, probabilities = analyze_predictions(model, dev_loader, device)
    
    # 혼동 행렬 플롯
    cm = confusion_matrix(true_labels, predictions)
    plot_confusion_matrix(
        cm, 
        class_names=['Unacceptable', 'Acceptable'],
        save_path=os.path.join(eval_dir, 'confusion_matrix.png')
    )
    
    # 예측 결과 저장
    prediction_results = {
        'predictions': predictions.tolist(),
        'true_labels': true_labels.tolist(),
        'probabilities': probabilities.tolist()
    }
    
    predictions_file = os.path.join(eval_dir, 'predictions.json')
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(prediction_results, f, indent=2)
    
    logger.info(f"평가 결과가 저장되었습니다: {eval_dir}")
    
    return results


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='BERT CoLA 파인튜닝')
    parser.add_argument('--config', type=str, default='config_large.yaml',
                        help='설정 파일 경로')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'both'], 
                        default=None, help='실행 모드')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='평가용 체크포인트 경로')
    parser.add_argument('--seed', type=int, default=None,
                        help='랜덤 시드')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    config_args = config_to_args(config)
    
    # 명령행 인수로 설정 덮어쓰기
    if args.mode is not None:
        config_args.mode = args.mode
    if args.checkpoint is not None:
        config_args.checkpoint_path = args.checkpoint
    if args.seed is not None:
        config_args.seed = args.seed
    
    # 시드 설정
    set_seed(config_args.seed)
    
    # 로깅 설정
    logger = setup_logging(config_args.log_dir)
    
    logger.info("=== BERT CoLA 파인튜닝 시작 ===")
    logger.info(f"설정 파일: {args.config}")
    logger.info(f"실행 모드: {config_args.mode}")
    logger.info(f"랜덤 시드: {config_args.seed}")
    logger.info(f"모델 경로: {config_args.model_path}")
    logger.info(f"데이터 디렉토리: {config_args.data_dir}")
    
    try:
        if config_args.mode in ['train', 'both']:
            # 훈련 실행
            best_model_path = train_model(config_args, logger)
            
            if config_args.mode == 'both':
                # 훈련 후 바로 평가
                evaluate_model_performance(config_args, best_model_path, logger)
        
        elif config_args.mode == 'eval':
            # 평가만 실행
            if config_args.checkpoint_path is None:
                raise ValueError("평가 모드에서는 --checkpoint 경로가 필요합니다.")
            
            evaluate_model_performance(config_args, config_args.checkpoint_path, logger)
        
        logger.info("=== 실행 완료 ===")
        
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()