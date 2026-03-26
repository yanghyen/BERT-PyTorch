"""
BERT 분류 모델 모듈
기존 BERT 모델을 로드하고 분류 헤드를 추가하여 MNLI 태스크에 맞게 조정
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import sys
import os

# 프로젝트 루트의 src 디렉토리를 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

from model.bert import BERT, LayerNorm, GELU


class BERTForSequenceClassification(nn.Module):
    """BERT 기반 시퀀스 분류 모델 (MNLI용 3-class)"""
    
    def __init__(self, 
                 bert_model: BERT,
                 num_labels: int = 3,
                 dropout: float = 0.1,
                 hidden_size: Optional[int] = None):
        """
        Args:
            bert_model: 사전 훈련된 BERT 모델
            num_labels: 분류할 레이블 수 (MNLI: 3개)
            dropout: 드롭아웃 확률
            hidden_size: 숨겨진 차원 크기 (None이면 BERT의 hidden 사용)
        """
        super().__init__()
        
        self.bert = bert_model
        self.num_labels = num_labels
        
        # BERT의 hidden size 가져오기
        if hidden_size is None:
            hidden_size = self.bert.hidden
        
        # 분류 헤드 구성
        # BERT 스타일: [CLS] -> Dense -> Tanh -> Dropout -> Classifier
        self.pooler = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # 분류 헤드 가중치 초기화
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """분류 헤드 가중치 초기화 (BERT 스타일)"""
        for module in [self.pooler, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: 입력 토큰 ID [batch_size, seq_len]
            attention_mask: 어텐션 마스크 [batch_size, seq_len] (사용되지 않음 - BERT 내부에서 패딩 마스크 생성)
            token_type_ids: 토큰 타입 ID [batch_size, seq_len]
            labels: 정답 레이블 [batch_size] (선택사항)
        
        Returns:
            딕셔너리 형태의 출력:
            - logits: 분류 로짓 [batch_size, num_labels]
            - loss: 손실 값 (labels가 주어진 경우)
            - hidden_states: BERT 출력 [batch_size, seq_len, hidden_size]
        """
        # 토큰 타입 ID가 없으면 0으로 초기화
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # BERT 인코딩
        hidden_states = self.bert(input_ids, token_type_ids)
        
        # [CLS] 토큰의 표현 추출 (첫 번째 토큰)
        cls_output = hidden_states[:, 0, :]  # [batch_size, hidden_size]
        
        # 풀링 및 분류
        pooled_output = self.pooler(cls_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        outputs = {
            'logits': logits,
            'hidden_states': hidden_states
        }
        
        # 손실 계산 (레이블이 주어진 경우)
        if labels is not None:
            if self.num_labels == 1:
                # 회귀 태스크
                loss = F.mse_loss(logits.squeeze(), labels.float())
            else:
                # 분류 태스크 (MNLI는 3-class 분류)
                # ignore_index=-1로 잘못된 라벨 무시
                loss = F.cross_entropy(logits, labels, ignore_index=-1)
            outputs['loss'] = loss
        
        return outputs
    
    def predict(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """예측 수행"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, token_type_ids)
            logits = outputs['logits']
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
        
        return predictions, probabilities


def load_pretrained_bert(model_path: str, 
                        vocab_size: int = 30522,
                        hidden: int = 768,
                        n_layers: int = 12,
                        attn_heads: int = 12,
                        dropout: float = 0.1) -> BERT:
    """사전 훈련된 BERT 모델 로드"""
    
    print(f"BERT 모델을 로드하는 중: {model_path}")
    
    # BERT 모델 생성
    bert_model = BERT(
        vocab_size=vocab_size,
        hidden=hidden,
        n_layers=n_layers,
        attn_heads=attn_heads,
        dropout=dropout
    )
    
    # 체크포인트 로드
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # 체크포인트 포맷(dict / nn.Module / state_dict) 모두 처리
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    elif hasattr(checkpoint, 'state_dict'):
        # 저장된 객체가 nn.Module(BERTLM 등)인 경우
        state_dict = checkpoint.state_dict()
    else:
        raise TypeError(
            f"지원하지 않는 체크포인트 타입입니다: {type(checkpoint)}"
        )
    
    # BERTLM에서 BERT 부분만 추출
    bert_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('bert.'):
            # BERTLM state_dict: bert.* 형태
            new_key = key[5:]
            bert_state_dict[new_key] = value
        elif key.startswith('embedding.') or key.startswith('transformer_blocks.'):
            # BERT 단독 state_dict: 접두사 없는 내부 키를 그대로 사용
            bert_state_dict[key] = value
    
    # 모델에 가중치 로드
    missing_keys, unexpected_keys = bert_model.load_state_dict(bert_state_dict, strict=False)
    
    if missing_keys:
        print(f"누락된 키: {missing_keys}")
    if unexpected_keys:
        print(f"예상치 못한 키: {unexpected_keys}")
    
    print("BERT 모델 로드 완료!")
    return bert_model


def create_classification_model(model_path: str,
                              num_labels: int = 3,
                              vocab_size: int = 30522,
                              hidden: int = 768,
                              n_layers: int = 12,
                              attn_heads: int = 12,
                              dropout: float = 0.1) -> BERTForSequenceClassification:
    """분류 모델 생성 (MNLI용 3-class)"""
    
    # 사전 훈련된 BERT 로드
    bert_model = load_pretrained_bert(
        model_path=model_path,
        vocab_size=vocab_size,
        hidden=hidden,
        n_layers=n_layers,
        attn_heads=attn_heads,
        dropout=dropout
    )
    
    # 분류 모델 생성
    classification_model = BERTForSequenceClassification(
        bert_model=bert_model,
        num_labels=num_labels,
        dropout=dropout,
        hidden_size=hidden
    )
    
    return classification_model


if __name__ == "__main__":
    # 테스트 코드
    print("BERT MNLI 분류 모델 테스트 중...")
    
    # 모델 경로 설정
    model_path = "../../../runs/L12_H768_A12_seed42/model_full.pth"
    
    try:
        # 분류 모델 생성 (MNLI: 3개 클래스)
        model = create_classification_model(
            model_path=model_path,
            num_labels=3,
            hidden=768,
            n_layers=12,
            attn_heads=12
        )
        
        print(f"모델 생성 완료!")
        print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
        print(f"훈련 가능한 파라미터 수: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # 더미 입력으로 테스트 (MNLI는 두 문장 입력)
        batch_size = 2
        seq_len = 256  # MNLI는 더 긴 시퀀스 사용
        
        input_ids = torch.randint(0, 30522, (batch_size, seq_len))
        token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        # 첫 번째 문장은 0, 두 번째 문장은 1로 설정
        sep_pos = seq_len // 2  # 임시로 중간 지점을 구분자로 설정
        token_type_ids[:, sep_pos:] = 1
        
        labels = torch.randint(0, 3, (batch_size,))  # 0: contradiction, 1: neutral, 2: entailment
        
        # 순전파 테스트
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=token_type_ids, labels=labels)
            
            print(f"\n출력 형태:")
            print(f"  Logits: {outputs['logits'].shape}")
            print(f"  Loss: {outputs['loss'].item():.4f}")
            print(f"  Hidden states: {outputs['hidden_states'].shape}")
        
        # 예측 테스트
        predictions, probabilities = model.predict(input_ids, token_type_ids=token_type_ids)
        print(f"\n예측 결과:")
        print(f"  Predictions: {predictions}")
        print(f"  Probabilities: {probabilities}")
        
        # 레이블 이름 매핑
        label_names = ["contradiction", "neutral", "entailment"]
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            print(f"  샘플 {i+1}: {label_names[pred.item()]} (confidence: {prob[pred].item():.4f})")
        
    except FileNotFoundError as e:
        print(f"오류: {e}")
        print("모델 파일 경로를 확인해주세요.")
    except Exception as e:
        print(f"예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()