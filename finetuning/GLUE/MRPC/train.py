"""
BERT MRPC 파인튜닝 학습 모듈
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from finetuning_model import BERTForSequenceClassification, create_classification_model, calculate_class_weights
from dataset import create_data_loaders
from evaluate import evaluate_model, compute_metrics


class BERTTrainer:
    """BERT MRPC 파인튜닝 트레이너"""
    
    def __init__(self,
                 model: BERTForSequenceClassification,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 learning_rate: float = 2e-5,
                 weight_decay: float = 0.01,
                 warmup_steps: int = 0,
                 max_grad_norm: float = 1.0,
                 device: str = 'cuda',
                 save_dir: str = './checkpoints',
                 use_wandb: bool = False,
                 project_name: str = 'bert-mrpc-finetuning',
                 use_f1_for_best_model: bool = True):
        """
        Args:
            model: BERT 분류 모델
            train_loader: 훈련 데이터 로더
            val_loader: 검증 데이터 로더
            learning_rate: 학습률
            weight_decay: 가중치 감쇠
            warmup_steps: 웜업 스텝 수
            max_grad_norm: 그래디언트 클리핑 최대값
            device: 디바이스 ('cuda' 또는 'cpu')
            save_dir: 체크포인트 저장 디렉토리
            use_wandb: Weights & Biases 사용 여부
            project_name: wandb 프로젝트 이름
            use_f1_for_best_model: F1 점수를 기준으로 최고 모델 선택 여부
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.max_grad_norm = max_grad_norm
        self.use_wandb = use_wandb
        self.use_f1_for_best_model = use_f1_for_best_model
        
        # 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        # 옵티마이저 설정
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=1e-8
        )
        
        # 스케줄러 설정
        total_steps = len(train_loader) * 10  # 기본 에포크 수 가정
        if warmup_steps == 0:
            warmup_steps = int(0.1 * total_steps)  # 전체 스텝의 10%를 웜업으로
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 로깅 설정
        self.setup_logging()
        
        # wandb 초기화
        if use_wandb:
            if not WANDB_AVAILABLE:
                self.logger.warning("wandb가 설치되지 않았습니다. wandb 로깅을 비활성화합니다.")
                self.use_wandb = False
            else:
                wandb.init(
                    project=project_name,
                    config={
                        'learning_rate': learning_rate,
                        'weight_decay': weight_decay,
                        'warmup_steps': warmup_steps,
                        'batch_size': train_loader.batch_size,
                        'model_name': 'BERT-MRPC',
                        'use_f1_for_best_model': use_f1_for_best_model
                    }
                )
    
    def setup_logging(self):
        """로깅 설정"""
        log_file = os.path.join(self.save_dir, 'training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """한 에포크 훈련"""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch}',
            dynamic_ncols=True,
            mininterval=0.5,
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # 데이터를 디바이스로 이동
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 순전파
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            # 역전파
            self.optimizer.zero_grad()
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # 옵티마이저 및 스케줄러 업데이트
            self.optimizer.step()
            self.scheduler.step()
            
            # 통계 업데이트
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # 진행률 표시 업데이트
            current_lr = self.scheduler.get_last_lr()[0]
            running_acc = (total_correct / total_samples) if total_samples > 0 else 0.0
            progress_bar.set_postfix({
                'L': f'{loss.item():.3f}',
                'A': f'{running_acc:.3f}',
                'LR': f'{current_lr:.2e}'
            })
            
            # wandb 로깅 (배치 단위)
            if self.use_wandb and WANDB_AVAILABLE and batch_idx % 50 == 0:  # MRPC는 작은 데이터셋이므로 더 자주 로깅
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/batch_accuracy': (predictions == labels).float().mean().item(),
                    'train/learning_rate': current_lr,
                    'train/step': epoch * len(self.train_loader) + batch_idx
                })
        
        # 에포크 평균 계산
        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """검증"""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(
                self.val_loader,
                desc='Validation',
                dynamic_ncols=True,
                mininterval=0.5,
            ):
                # 데이터를 디바이스로 이동
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 순전파
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                
                loss = outputs['loss']
                logits = outputs['logits']
                
                total_loss += loss.item()
                
                # 예측 및 레이블 수집
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 메트릭 계산
        metrics = compute_metrics(all_predictions, all_labels)
        metrics['loss'] = total_loss / len(self.val_loader)
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }
        
        # 일반 체크포인트 저장
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # 최고 성능 모델 저장
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            metric_name = "F1" if self.use_f1_for_best_model else "Accuracy"
            self.logger.info(f"새로운 최고 성능 모델 저장 ({metric_name}): {best_path}")
        
        self.logger.info(f"체크포인트 저장: {checkpoint_path}")
    
    def train(self, 
              num_epochs: int = 5,
              eval_steps: int = 100,
              save_steps: int = 200,
              early_stopping_patience: int = 5) -> Dict[str, List[float]]:
        """전체 훈련 과정"""
        
        self.logger.info("MRPC 훈련 시작")
        self.logger.info(f"에포크 수: {num_epochs}")
        self.logger.info(f"훈련 배치 수: {len(self.train_loader)}")
        self.logger.info(f"검증 배치 수: {len(self.val_loader)}")
        self.logger.info(f"최고 모델 기준: {'F1' if self.use_f1_for_best_model else 'Accuracy'}")
        
        best_metric = 0.0
        patience_counter = 0
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': []
        }
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            
            # 훈련
            train_metrics = self.train_epoch(epoch)
            
            # 검증
            val_metrics = self.validate(epoch)
            
            # 기록 업데이트
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])
            history['val_precision'].append(val_metrics['precision'])
            history['val_recall'].append(val_metrics['recall'])
            
            epoch_time = time.time() - epoch_start_time
            
            # 로깅
            self.logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val F1: {val_metrics['f1']:.4f}, "
                f"Val Precision: {val_metrics['precision']:.4f}, "
                f"Val Recall: {val_metrics['recall']:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # wandb 로깅
            if self.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_metrics['loss'],
                    'train/accuracy': train_metrics['accuracy'],
                    'val/loss': val_metrics['loss'],
                    'val/accuracy': val_metrics['accuracy'],
                    'val/f1': val_metrics['f1'],
                    'val/precision': val_metrics['precision'],
                    'val/recall': val_metrics['recall'],
                    'epoch_time': epoch_time
                })
            
            # 최고 성능 모델 확인
            current_metric = val_metrics['f1'] if self.use_f1_for_best_model else val_metrics['accuracy']
            is_best = current_metric > best_metric
            
            if is_best:
                best_metric = current_metric
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 체크포인트 저장
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # 조기 종료 확인
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"조기 종료: {early_stopping_patience} 에포크 동안 성능 향상 없음")
                break
        
        total_time = time.time() - start_time
        metric_name = "F1" if self.use_f1_for_best_model else "Accuracy"
        self.logger.info(f"훈련 완료! 총 시간: {total_time:.2f}s")
        self.logger.info(f"최고 검증 {metric_name}: {best_metric:.4f}")
        
        # 훈련 기록 저장
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
        
        return history


def main():
    """메인 훈련 함수"""
    
    # 설정
    config = {
        'model_path': '../../../runs/L12_H768_A12_seed42/model_full.pth',
        'data_dir': './data/MRPC',
        'save_dir': './checkpoints',
        'batch_size': 32,
        'learning_rate': 2e-5,
        'num_epochs': 5,
        'max_length': 128,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'max_grad_norm': 1.0,
        'early_stopping_patience': 5,
        'use_wandb': False,
        'seed': 42,
        'use_f1_for_best_model': True,
        'use_class_weights': False  # 클래스 불균형 처리 여부
    }
    
    # 시드 설정
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 데이터 로더 생성
    print("데이터 로더 생성 중...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        num_workers=4
    )
    
    print(f"훈련 샘플 수: {len(train_loader.dataset)}")
    print(f"검증 샘플 수: {len(val_loader.dataset)}")
    print(f"테스트 샘플 수: {len(test_loader.dataset)}")
    
    # 클래스 가중치 계산 (옵션)
    class_weights = None
    if config['use_class_weights']:
        print("클래스 가중치 계산 중...")
        train_labels = [batch['labels'].item() for batch in train_loader.dataset]
        class_weights = calculate_class_weights(train_labels, device)
        print(f"클래스 가중치: {class_weights}")
    
    # 모델 생성
    print("모델 생성 중...")
    model = create_classification_model(
        model_path=config['model_path'],
        num_labels=2,  # MRPC: 2개 클래스
        hidden=768,
        n_layers=12,
        attn_heads=12,
        dropout=0.1,
        class_weights=class_weights
    )
    
    # 트레이너 생성
    warmup_steps = int(config['warmup_ratio'] * len(train_loader) * config['num_epochs'])
    
    trainer = BERTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        warmup_steps=warmup_steps,
        max_grad_norm=config['max_grad_norm'],
        device=device,
        save_dir=config['save_dir'],
        use_wandb=config['use_wandb'],
        use_f1_for_best_model=config['use_f1_for_best_model']
    )
    
    # 훈련 실행
    print("훈련 시작...")
    history = trainer.train(
        num_epochs=config['num_epochs'],
        early_stopping_patience=config['early_stopping_patience']
    )
    
    print("훈련 완료!")
    print(f"최고 검증 정확도: {max(history['val_accuracy']):.4f}")
    print(f"최고 검증 F1: {max(history['val_f1']):.4f}")
    print(f"최고 검증 정밀도: {max(history['val_precision']):.4f}")
    print(f"최고 검증 재현율: {max(history['val_recall']):.4f}")


if __name__ == "__main__":
    main()