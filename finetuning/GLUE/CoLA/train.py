"""
BERT CoLA 파인튜닝 학습 모듈
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
import wandb

from finetuning_model import BERTForSequenceClassification, create_classification_model
from dataset import create_data_loaders
from evaluate import evaluate_model, compute_metrics


class BERTTrainer:
    """BERT CoLA 파인튜닝 트레이너"""
    
    def __init__(self,
                 model: BERTForSequenceClassification,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 learning_rate: float = 1e-5,
                 weight_decay: float = 0.01,
                 num_epochs: int = 5,
                 warmup_ratio: float = 0.1,
                 max_grad_norm: float = 1.0,
                 device: str = 'cuda',
                 save_dir: str = './checkpoints',
                 eval_steps: int = 100,
                 save_steps: int = 200,
                 early_stopping_patience: int = 5,
                 use_wandb: bool = False,
                 wandb_project: str = 'bert-cola-finetuning'):
        """
        Args:
            model: BERT 분류 모델
            train_loader: 훈련 데이터 로더
            val_loader: 검증 데이터 로더
            learning_rate: 학습률
            weight_decay: 가중치 감쇠
            num_epochs: 훈련 에포크 수
            warmup_ratio: 웜업 비율
            max_grad_norm: 그래디언트 클리핑 최대값
            device: 디바이스 ('cuda' 또는 'cpu')
            save_dir: 체크포인트 저장 디렉토리
            eval_steps: 평가 주기
            save_steps: 저장 주기
            early_stopping_patience: 조기 종료 인내심
            use_wandb: Weights & Biases 사용 여부
            wandb_project: wandb 프로젝트 이름
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.early_stopping_patience = early_stopping_patience
        self.use_wandb = use_wandb
        
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
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(warmup_ratio * total_steps)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        # wandb 초기화
        if use_wandb:
            wandb.init(
                project=wandb_project,
                config={
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'num_epochs': num_epochs,
                    'warmup_ratio': warmup_ratio,
                    'batch_size': train_loader.batch_size,
                    'model_name': 'BERT-CoLA'
                }
            )
        
        # 훈련 상태 추적
        self.best_matthews_corr = -1.0
        self.best_model_path = None
        self.patience_counter = 0
        self.global_step = 0
        self.train_history = []
    
    def train_epoch(self) -> Dict[str, float]:
        """한 에포크 훈련"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        
        for batch in progress_bar:
            # 데이터를 디바이스로 이동
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 그래디언트 초기화
            self.optimizer.zero_grad()
            
            # 순전파
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            loss = outputs['loss']
            
            # 역전파
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # 옵티마이저 업데이트
            self.optimizer.step()
            self.scheduler.step()
            
            # 손실 누적
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # 진행 상황 업데이트
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # 주기적 평가 및 저장
            if self.global_step % self.eval_steps == 0:
                val_metrics = self.evaluate()
                self.model.train()  # 다시 훈련 모드로
                
                # wandb 로깅
                if self.use_wandb:
                    wandb.log({
                        'train_loss': avg_loss,
                        'val_matthews_corr': val_metrics['matthews_corr'],
                        'val_accuracy': val_metrics['accuracy'],
                        'val_f1': val_metrics['f1'],
                        'learning_rate': self.scheduler.get_last_lr()[0],
                        'global_step': self.global_step
                    })
                
                # 최고 성능 모델 저장
                if val_metrics['matthews_corr'] > self.best_matthews_corr:
                    self.best_matthews_corr = val_metrics['matthews_corr']
                    self.best_model_path = self.save_checkpoint(
                        f'best_model_step_{self.global_step}.pth',
                        val_metrics
                    )
                    self.patience_counter = 0
                    self.logger.info(f"새로운 최고 Matthews 상관계수: {self.best_matthews_corr:.4f}")
                else:
                    self.patience_counter += 1
                
                # 조기 종료 체크
                if self.patience_counter >= self.early_stopping_patience:
                    self.logger.info(f"조기 종료: {self.early_stopping_patience} 스텝 동안 개선 없음")
                    return {'loss': avg_loss, 'early_stop': True}
            
            # 주기적 체크포인트 저장
            if self.global_step % self.save_steps == 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}.pth')
        
        return {'loss': total_loss / num_batches, 'early_stop': False}
    
    def evaluate(self) -> Dict[str, float]:
        """검증 데이터로 모델 평가"""
        self.logger.info("검증 데이터 평가 중...")
        metrics = evaluate_model(self.model, self.val_loader, self.device)
        
        self.logger.info(f"검증 결과 - Matthews 상관계수: {metrics['matthews_corr']:.4f}, "
                        f"정확도: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        
        return metrics
    
    def save_checkpoint(self, filename: str, metrics: Optional[Dict] = None) -> str:
        """체크포인트 저장"""
        checkpoint_path = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_matthews_corr': self.best_matthews_corr,
        }
        
        if metrics:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"체크포인트 저장: {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str):
        """체크포인트 로드"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.best_matthews_corr = checkpoint.get('best_matthews_corr', -1.0)
        
        self.logger.info(f"체크포인트 로드: {checkpoint_path}")
    
    def train(self) -> str:
        """전체 훈련 과정"""
        self.logger.info(f"CoLA 파인튜닝 시작 - {self.num_epochs} 에포크")
        self.logger.info(f"훈련 데이터: {len(self.train_loader.dataset)}개")
        self.logger.info(f"검증 데이터: {len(self.val_loader.dataset)}개")
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            self.logger.info(f"\n에포크 {epoch + 1}/{self.num_epochs}")
            
            # 에포크 훈련
            epoch_results = self.train_epoch()
            
            # 훈련 기록 저장
            self.train_history.append({
                'epoch': epoch + 1,
                'train_loss': epoch_results['loss'],
                'global_step': self.global_step,
                'best_matthews_corr': self.best_matthews_corr
            })
            
            # 조기 종료 체크
            if epoch_results.get('early_stop', False):
                break
            
            # 에포크 종료 후 평가
            val_metrics = self.evaluate()
            
            # wandb 에포크 로깅
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'epoch_train_loss': epoch_results['loss'],
                    'epoch_val_matthews_corr': val_metrics['matthews_corr'],
                    'epoch_val_accuracy': val_metrics['accuracy'],
                    'epoch_val_f1': val_metrics['f1']
                })
        
        # 훈련 완료
        total_time = time.time() - start_time
        self.logger.info(f"\n훈련 완료! 총 시간: {total_time:.2f}초")
        self.logger.info(f"최고 Matthews 상관계수: {self.best_matthews_corr:.4f}")
        self.logger.info(f"최고 모델 경로: {self.best_model_path}")
        
        # 훈련 기록 저장
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.train_history, f, indent=2)
        
        # wandb 종료
        if self.use_wandb:
            wandb.finish()
        
        return self.best_model_path or self.save_checkpoint('final_model.pth')


if __name__ == "__main__":
    # 테스트 코드
    print("CoLA 트레이너 테스트 중...")
    
    # 더미 설정으로 테스트
    from dataset import create_data_loaders
    from finetuning_model import create_classification_model
    
    # 작은 배치 크기로 테스트
    train_loader, val_loader, _ = create_data_loaders(
        batch_size=4,
        max_length=64
    )
    
    # 작은 모델로 테스트 (실제 경로는 수정 필요)
    model = create_classification_model(
        model_path="../../../runs/L24_H1024_A16_seed42/model_full.pth",
        num_labels=2,
        hidden=1024,
        n_layers=24,
        attn_heads=16
    )
    
    # 트레이너 생성
    trainer = BERTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=1,  # 테스트용으로 1 에포크만
        eval_steps=10,
        save_steps=20,
        use_wandb=False
    )
    
    print("트레이너 초기화 완료!")
    print(f"총 훈련 스텝: {len(train_loader)}")
    print(f"총 검증 샘플: {len(val_loader.dataset)}")