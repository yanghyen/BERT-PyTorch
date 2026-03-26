"""
BERT MNLI 모델 평가 모듈
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
    """분류 메트릭 계산 (MNLI 3-class)"""
    
    # 기본 메트릭
    accuracy = accuracy_score(labels, predictions)
    
    # 다중 클래스를 위한 macro/micro 평균
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average='macro'
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, predictions, average='micro'
    )
    
    # 클래스별 메트릭
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        labels, predictions, average=None
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision_macro,
        'recall': recall_macro,
        'f1': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
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
    
    for idx in wrong_indices[:30]:  # 상위 30개만
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
    
    # 클래스별 성능 분석
    label_names = get_label_names()
    class_analysis = {}
    
    for i, class_name in enumerate(label_names):
        class_mask = np.array(labels) == i
        if class_mask.sum() > 0:
            class_predictions = np.array(predictions)[class_mask]
            class_correct = (class_predictions == i).sum()
            class_total = class_mask.sum()
            
            class_analysis[class_name] = {
                'total_samples': int(class_total),
                'correct_predictions': int(class_correct),
                'accuracy': float(class_correct / class_total),
                'avg_confidence': float(np.mean([confidences[j] for j in range(len(confidences)) if labels[j] == i]))
            }
    
    # 신뢰도별 정확도
    confidence_bins = np.linspace(0.33, 1.0, 11)  # 3-class이므로 0.33부터 시작
    confidence_accuracy = []
    
    for i in range(len(confidence_bins) - 1):
        bin_mask = (np.array(confidences) >= confidence_bins[i]) & \
                   (np.array(confidences) < confidence_bins[i + 1])
        if bin_mask.sum() > 0:
            bin_accuracy = correct_mask[bin_mask].mean()
            confidence_accuracy.append({
                'confidence_range': f'{confidence_bins[i]:.2f}-{confidence_bins[i+1]:.2f}',
                'accuracy': float(bin_accuracy),
                'count': int(bin_mask.sum())
            })
    
    analysis = {
        'confusion_matrix': cm.tolist(),
        'wrong_samples': wrong_samples,
        'class_analysis': class_analysis,
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
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('MNLI Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_class_performance(class_analysis: Dict,
                          save_path: str = './class_performance.png'):
    """클래스별 성능 시각화"""
    
    class_names = list(class_analysis.keys())
    accuracies = [class_analysis[name]['accuracy'] for name in class_names]
    confidences = [class_analysis[name]['avg_confidence'] for name in class_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 클래스별 정확도
    bars1 = ax1.bar(class_names, accuracies, color=['red', 'orange', 'green'])
    ax1.set_title('Accuracy by Class')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    
    # 막대 위에 값 표시
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # 클래스별 평균 신뢰도
    bars2 = ax2.bar(class_names, confidences, color=['red', 'orange', 'green'])
    ax2.set_title('Average Confidence by Class')
    ax2.set_ylabel('Confidence')
    ax2.set_ylim(0, 1)
    
    # 막대 위에 값 표시
    for bar, conf in zip(bars2, confidences):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{conf:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confidence_distribution(confidences: List[float],
                                correct_mask: List[bool],
                                save_path: str = './confidence_distribution.png'):
    """신뢰도 분포 시각화"""
    
    correct_confidences = [confidences[i] for i in range(len(confidences)) if correct_mask[i]]
    wrong_confidences = [confidences[i] for i in range(len(confidences)) if not correct_mask[i]]
    
    plt.figure(figsize=(12, 6))
    plt.hist(correct_confidences, bins=30, alpha=0.7, label='Correct', color='green', density=True)
    plt.hist(wrong_confidences, bins=30, alpha=0.7, label='Wrong', color='red', density=True)
    plt.xlabel('Confidence Score')
    plt.ylabel('Density')
    plt.title('Confidence Score Distribution (MNLI)')
    plt.legend()
    plt.axvline(x=1/3, color='black', linestyle='--', alpha=0.5, label='Random Guess (0.33)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_matched_mismatched(model: BERTForSequenceClassification,
                               matched_loader: DataLoader,
                               mismatched_loader: DataLoader,
                               device: str = 'cuda',
                               save_dir: str = './evaluation') -> Dict:
    """Matched와 Mismatched 데이터셋 모두 평가"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("Matched 데이터셋 평가 중...")
    matched_results = evaluate_model(model, matched_loader, device)
    
    print("Mismatched 데이터셋 평가 중...")
    mismatched_results = evaluate_model(model, mismatched_loader, device)
    
    # 결과 비교
    comparison = {
        'matched': {
            'metrics': matched_results.get('metrics', {}),
            'loss': matched_results.get('loss', 0.0),
            'sample_count': len(matched_results['predictions'])
        },
        'mismatched': {
            'metrics': mismatched_results.get('metrics', {}),
            'loss': mismatched_results.get('loss', 0.0),
            'sample_count': len(mismatched_results['predictions'])
        }
    }
    
    # 성능 차이 계산
    if 'metrics' in matched_results and 'metrics' in mismatched_results:
        matched_acc = matched_results['metrics']['accuracy']
        mismatched_acc = mismatched_results['metrics']['accuracy']
        
        comparison['performance_gap'] = {
            'accuracy_difference': matched_acc - mismatched_acc,
            'matched_accuracy': matched_acc,
            'mismatched_accuracy': mismatched_acc
        }
    
    # 시각화
    if 'metrics' in matched_results and 'metrics' in mismatched_results:
        # 혼동 행렬 비교
        matched_cm = confusion_matrix(matched_results['labels'], matched_results['predictions'])
        mismatched_cm = confusion_matrix(mismatched_results['labels'], mismatched_results['predictions'])
        
        plot_confusion_matrix(
            matched_cm,
            get_label_names(),
            os.path.join(save_dir, 'confusion_matrix_matched.png')
        )
        
        plot_confusion_matrix(
            mismatched_cm,
            get_label_names(),
            os.path.join(save_dir, 'confusion_matrix_mismatched.png')
        )
        
        # 성능 비교 차트
        plot_matched_mismatched_comparison(
            matched_results['metrics'],
            mismatched_results['metrics'],
            os.path.join(save_dir, 'matched_vs_mismatched.png')
        )
    
    # 결과 저장
    with open(os.path.join(save_dir, 'matched_mismatched_comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2)
    
    return comparison


def plot_matched_mismatched_comparison(matched_metrics: Dict,
                                     mismatched_metrics: Dict,
                                     save_path: str):
    """Matched vs Mismatched 성능 비교 시각화"""
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    matched_values = [matched_metrics[metric] for metric in metrics]
    mismatched_values = [mismatched_metrics[metric] for metric in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, matched_values, width, label='Matched', color='skyblue')
    bars2 = ax.bar(x + width/2, mismatched_values, width, label='Mismatched', color='lightcoral')
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('MNLI: Matched vs Mismatched Performance')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend()
    ax.set_ylim(0, 1)
    
    # 막대 위에 값 표시
    for bars, values in [(bars1, matched_values), (bars2, mismatched_values)]:
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """메인 평가 함수"""
    
    # 설정
    config = {
        'checkpoint_path': './checkpoints/best_model.pt',
        'data_dir': './data/MNLI',
        'batch_size': 16,
        'max_length': 256,
        'save_dir': './evaluation',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'evaluate_both': True  # matched와 mismatched 모두 평가
    }
    
    print(f"사용 디바이스: {config['device']}")
    
    # 데이터 로더 생성
    from dataset import create_matched_mismatched_loaders
    
    print("데이터 로더 생성 중...")
    train_loader, val_matched_loader, val_mismatched_loader = create_matched_mismatched_loaders(
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
        num_labels=3  # MNLI: 3개 클래스
    )
    
    # 체크포인트에서 파인튜닝된 가중치 로드
    checkpoint = torch.load(config['checkpoint_path'], map_location=config['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if config['evaluate_both']:
        # Matched와 Mismatched 모두 평가
        comparison = evaluate_matched_mismatched(
            model, val_matched_loader, val_mismatched_loader,
            config['device'], config['save_dir']
        )
        
        print(f"\n평가 결과:")
        print(f"  Matched 정확도: {comparison['matched']['metrics']['accuracy']:.4f}")
        print(f"  Mismatched 정확도: {comparison['mismatched']['metrics']['accuracy']:.4f}")
        print(f"  성능 차이: {comparison['performance_gap']['accuracy_difference']:.4f}")
        
    else:
        # Matched만 평가
        print("Matched 데이터 평가 중...")
        results = evaluate_model(model, val_matched_loader, config['device'])
        
        if 'metrics' in results:
            print(f"\n검증 데이터 결과:")
            metrics = results['metrics']
            print(f"  정확도: {metrics['accuracy']:.4f}")
            print(f"  정밀도: {metrics['precision']:.4f}")
            print(f"  재현율: {metrics['recall']:.4f}")
            print(f"  F1 점수: {metrics['f1']:.4f}")
    
    print(f"\n평가 완료! 결과가 {config['save_dir']}에 저장되었습니다.")


if __name__ == "__main__":
    main()