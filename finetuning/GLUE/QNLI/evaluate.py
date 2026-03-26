"""
BERT QNLI 모델 평가 모듈
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from finetuning_model import BERTForSequenceClassification
from dataset import get_label_names


def compute_metrics(predictions: List[int], 
                   labels: List[int]) -> Dict[str, float]:
    """분류 메트릭 계산"""
    
    # 기본 메트릭
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', pos_label=1
    )
    
    # 클래스별 메트릭
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        labels, predictions, average=None
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'support_per_class': support.tolist()
    }


def evaluate_model(model: BERTForSequenceClassification,
                  data_loader: DataLoader,
                  device: str = 'cuda',
                  return_predictions: bool = False) -> Dict:
    """모델 평가"""
    
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            # 데이터를 디바이스로 이동
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # 순전파
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels if labels[0] != -1 else None  # test 데이터는 레이블이 -1
            )
            
            logits = outputs['logits']
            
            # 손실 계산 (레이블이 있는 경우)
            if 'loss' in outputs:
                total_loss += outputs['loss'].item()
            
            # 예측 및 확률 계산
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            # 결과 수집
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # 레이블 수집 (test 데이터가 아닌 경우)
            if labels[0] != -1:
                all_labels.extend(labels.cpu().numpy())
    
    # 결과 정리
    results = {
        'predictions': all_predictions,
        'probabilities': all_probabilities
    }
    
    # 레이블이 있는 경우 메트릭 계산
    if all_labels:
        results['labels'] = all_labels
        results['metrics'] = compute_metrics(all_predictions, all_labels)
        results['loss'] = total_loss / len(data_loader)
    
    return results


def analyze_predictions(predictions: List[int],
                       labels: List[int],
                       probabilities: List[List[float]],
                       texts: Optional[List[str]] = None,
                       save_dir: str = './analysis') -> Dict:
    """예측 결과 분석"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 혼동 행렬
    cm = confusion_matrix(labels, predictions)
    
    # 신뢰도 분석
    confidences = [max(prob) for prob in probabilities]
    correct_mask = np.array(predictions) == np.array(labels)
    
    # 잘못 분류된 샘플 분석
    wrong_indices = np.where(~correct_mask)[0]
    wrong_samples = []
    
    for idx in wrong_indices[:20]:  # 상위 20개만
        sample_info = {
            'index': int(idx),
            'true_label': int(labels[idx]),
            'predicted_label': int(predictions[idx]),
            'confidence': float(confidences[idx]),
            'probabilities': [float(p) for p in probabilities[idx]]
        }
        if texts:
            sample_info['text'] = texts[idx]
        wrong_samples.append(sample_info)
    
    # 신뢰도별 정확도
    confidence_bins = np.linspace(0.5, 1.0, 11)
    confidence_accuracy = []
    
    for i in range(len(confidence_bins) - 1):
        bin_mask = (np.array(confidences) >= confidence_bins[i]) & \
                   (np.array(confidences) < confidence_bins[i + 1])
        if bin_mask.sum() > 0:
            bin_accuracy = correct_mask[bin_mask].mean()
            confidence_accuracy.append({
                'confidence_range': f'{confidence_bins[i]:.1f}-{confidence_bins[i+1]:.1f}',
                'accuracy': float(bin_accuracy),
                'count': int(bin_mask.sum())
            })
    
    analysis = {
        'confusion_matrix': cm.tolist(),
        'wrong_samples': wrong_samples,
        'confidence_accuracy': confidence_accuracy,
        'average_confidence': float(np.mean(confidences)),
        'correct_confidence': float(np.mean([confidences[i] for i in range(len(confidences)) if correct_mask[i]])),
        'wrong_confidence': float(np.mean([confidences[i] for i in range(len(confidences)) if not correct_mask[i]]))
    }
    
    # 분석 결과 저장
    with open(os.path.join(save_dir, 'analysis.json'), 'w') as f:
        json.dump(analysis, f, indent=2)
    
    return analysis


def plot_confusion_matrix(cm: np.ndarray, 
                         class_names: List[str],
                         save_path: str = './confusion_matrix.png'):
    """혼동 행렬 시각화"""
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - QNLI')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confidence_distribution(confidences: List[float],
                                correct_mask: List[bool],
                                save_path: str = './confidence_distribution.png'):
    """신뢰도 분포 시각화"""
    
    correct_confidences = [confidences[i] for i in range(len(confidences)) if correct_mask[i]]
    wrong_confidences = [confidences[i] for i in range(len(confidences)) if not correct_mask[i]]
    
    plt.figure(figsize=(10, 6))
    plt.hist(correct_confidences, bins=20, alpha=0.7, label='Correct', color='green')
    plt.hist(wrong_confidences, bins=20, alpha=0.7, label='Wrong', color='red')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Confidence Score Distribution - QNLI')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_checkpoint(checkpoint_path: str,
                       data_loader: DataLoader,
                       device: str = 'cuda',
                       save_dir: str = './evaluation') -> Dict:
    """체크포인트 모델 평가"""
    
    print(f"체크포인트 로드 중: {checkpoint_path}")
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 모델 상태 복원 (모델 구조는 별도로 생성해야 함)
    # 이 함수는 이미 생성된 모델과 함께 사용되어야 함
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("모델 평가 중...")
    # 평가 실행은 별도 함수에서 수행
    
    return checkpoint.get('metrics', {})


def compare_models(model_paths: List[str],
                  model_names: List[str],
                  data_loader: DataLoader,
                  device: str = 'cuda') -> Dict:
    """여러 모델 성능 비교"""
    
    results = {}
    
    for model_path, model_name in zip(model_paths, model_names):
        print(f"\n{model_name} 평가 중...")
        
        # 각 모델에 대해 평가 수행
        # 실제 구현에서는 모델을 로드하고 평가해야 함
        
        results[model_name] = {
            'path': model_path,
            'metrics': {}  # 실제 메트릭이 들어갈 자리
        }
    
    return results


def main():
    """메인 평가 함수"""
    
    # 설정
    config = {
        'checkpoint_path': './checkpoints/best_model.pt',
        'data_dir': './data/QNLI',
        'batch_size': 16,
        'max_length': 512,
        'save_dir': './evaluation',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"사용 디바이스: {config['device']}")
    
    # 데이터 로더 생성
    from dataset import create_data_loaders
    
    print("데이터 로더 생성 중...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        max_length=config['max_length']
    )
    
    # 체크포인트가 존재하는지 확인
    if not os.path.exists(config['checkpoint_path']):
        print(f"체크포인트를 찾을 수 없습니다: {config['checkpoint_path']}")
        print("먼저 모델을 훈련해주세요.")
        return
    
    # 모델 로드 및 평가
    from finetuning_model import create_classification_model
    
    print("모델 생성 중...")
    model = create_classification_model(
        model_path='../../../runs/L12_H768_A12_seed42/model_full.pth',
        num_labels=2
    )
    
    # 체크포인트에서 파인튜닝된 가중치 로드
    checkpoint = torch.load(config['checkpoint_path'], map_location=config['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("검증 데이터 평가 중...")
    val_results = evaluate_model(model, val_loader, config['device'])
    
    print("테스트 데이터 평가 중...")
    test_results = evaluate_model(model, test_loader, config['device'])
    
    # 결과 출력
    if 'metrics' in val_results:
        print(f"\n검증 데이터 결과:")
        metrics = val_results['metrics']
        print(f"  정확도: {metrics['accuracy']:.4f}")
        print(f"  정밀도: {metrics['precision']:.4f}")
        print(f"  재현율: {metrics['recall']:.4f}")
        print(f"  F1 점수: {metrics['f1']:.4f}")
    
    # 분석 및 시각화
    os.makedirs(config['save_dir'], exist_ok=True)
    
    if 'metrics' in val_results:
        # 혼동 행렬 시각화
        cm = confusion_matrix(val_results['labels'], val_results['predictions'])
        plot_confusion_matrix(
            cm, 
            get_label_names(),
            os.path.join(config['save_dir'], 'confusion_matrix.png')
        )
        
        # 신뢰도 분포 시각화
        confidences = [max(prob) for prob in val_results['probabilities']]
        correct_mask = np.array(val_results['predictions']) == np.array(val_results['labels'])
        
        plot_confidence_distribution(
            confidences,
            correct_mask,
            os.path.join(config['save_dir'], 'confidence_distribution.png')
        )
        
        # 상세 분석
        analysis = analyze_predictions(
            val_results['predictions'],
            val_results['labels'],
            val_results['probabilities'],
            save_dir=config['save_dir']
        )
        
        print(f"\n분석 결과:")
        print(f"  평균 신뢰도: {analysis['average_confidence']:.4f}")
        print(f"  정답 신뢰도: {analysis['correct_confidence']:.4f}")
        print(f"  오답 신뢰도: {analysis['wrong_confidence']:.4f}")
    
    # 결과 저장
    results_path = os.path.join(config['save_dir'], 'evaluation_results.json')
    with open(results_path, 'w') as f:
        # numpy 배열을 리스트로 변환하여 JSON 직렬화 가능하게 만듦
        serializable_results = {
            'val_results': {
                'metrics': val_results.get('metrics', {}),
                'loss': val_results.get('loss', 0.0)
            },
            'test_results': {
                'predictions_count': len(test_results['predictions'])
            },
            'config': config
        }
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n평가 완료! 결과가 {config['save_dir']}에 저장되었습니다.")


if __name__ == "__main__":
    main()