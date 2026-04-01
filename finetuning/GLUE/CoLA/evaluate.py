"""
BERT CoLA 모델 평가 모듈
CoLA (Corpus of Linguistic Acceptability) 태스크 전용 평가 함수들
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, matthews_corrcoef
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
    """CoLA 분류 메트릭 계산 (Matthews 상관계수 포함)"""
    
    # 기본 메트릭
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', pos_label=1
    )
    
    # Matthews 상관계수 (CoLA의 주요 평가 지표)
    matthews_corr = matthews_corrcoef(labels, predictions)
    
    # 클래스별 메트릭
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        labels, predictions, average=None
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'matthews_corr': matthews_corr,  # CoLA의 주요 지표
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'support_per_class': support.tolist()
    }


def evaluate_model(model: BERTForSequenceClassification,
                  data_loader: DataLoader,
                  device: str = 'cuda') -> Dict:
    """CoLA 모델 평가"""
    
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating CoLA'):
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
    
    # 레이블이 있는 경우 메트릭 계산
    if all_labels:
        metrics = compute_metrics(all_predictions, all_labels)
        avg_loss = total_loss / len(data_loader)
        
        # CoLA 결과 출력
        print(f"\nCoLA 평가 결과:")
        print(f"  정확도: {metrics['accuracy']:.4f}")
        print(f"  정밀도: {metrics['precision']:.4f}")
        print(f"  재현율: {metrics['recall']:.4f}")
        print(f"  F1 점수: {metrics['f1']:.4f}")
        print(f"  Matthews 상관계수: {metrics['matthews_corr']:.4f}")  # 주요 지표
        print(f"  평균 손실: {avg_loss:.4f}")
        
        return {**metrics, 'loss': avg_loss}
    else:
        return {'predictions': all_predictions, 'probabilities': all_probabilities}


def analyze_predictions(model: BERTForSequenceClassification,
                       data_loader: DataLoader,
                       device: str = 'cuda') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """예측 분석을 위한 상세 결과 반환"""
    
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Analyzing predictions'):
            # 데이터를 디바이스로 이동
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # 순전파
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            logits = outputs['logits']
            
            # 예측 및 확률 계산
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            # 결과 수집
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # 레이블 수집 (test 데이터가 아닌 경우)
            if labels[0] != -1:
                all_labels.extend(labels.cpu().numpy())
    
    return (np.array(all_predictions), 
            np.array(all_labels) if all_labels else None,
            np.array(all_probabilities))


def plot_confusion_matrix(cm: np.ndarray,
                         class_names: List[str],
                         save_path: str = 'confusion_matrix.png',
                         title: str = 'CoLA Confusion Matrix'):
    """혼동 행렬 시각화"""
    
    plt.figure(figsize=(8, 6))
    
    # 정규화된 혼동 행렬
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 히트맵 그리기
    sns.heatmap(cm_normalized, 
                annot=True, 
                fmt='.3f',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'})
    
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    # 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"혼동 행렬이 저장되었습니다: {save_path}")


def analyze_error_cases(predictions: List[int],
                       labels: List[int],
                       probabilities: List[List[float]],
                       texts: Optional[List[str]] = None,
                       save_dir: str = './analysis') -> Dict:
    """오류 사례 분석"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 신뢰도 분석
    confidences = [max(prob) for prob in probabilities]
    correct_mask = np.array(predictions) == np.array(labels)
    
    # 잘못 분류된 샘플 분석
    wrong_indices = np.where(~correct_mask)[0]
    wrong_samples = []
    
    for idx in wrong_indices[:50]:  # 상위 50개 오류 사례
        sample_info = {
            'index': int(idx),
            'true_label': int(labels[idx]),
            'predicted_label': int(predictions[idx]),
            'confidence': float(confidences[idx]),
            'probabilities': [float(p) for p in probabilities[idx]],
            'true_label_name': get_label_names()[labels[idx]],
            'predicted_label_name': get_label_names()[predictions[idx]]
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
                'confidence_range': f'{confidence_bins[i]:.2f}-{confidence_bins[i+1]:.2f}',
                'accuracy': float(bin_accuracy),
                'count': int(bin_mask.sum())
            })
    
    # 클래스별 오류 분석
    class_errors = {}
    for true_label in [0, 1]:
        class_mask = np.array(labels) == true_label
        class_predictions = np.array(predictions)[class_mask]
        class_correct = (class_predictions == true_label).sum()
        class_total = class_mask.sum()
        
        class_errors[get_label_names()[true_label]] = {
            'total': int(class_total),
            'correct': int(class_correct),
            'accuracy': float(class_correct / class_total) if class_total > 0 else 0.0,
            'error_count': int(class_total - class_correct)
        }
    
    analysis_results = {
        'wrong_samples': wrong_samples,
        'confidence_accuracy': confidence_accuracy,
        'class_errors': class_errors,
        'overall_stats': {
            'total_samples': len(predictions),
            'correct_predictions': int(correct_mask.sum()),
            'wrong_predictions': int((~correct_mask).sum()),
            'accuracy': float(correct_mask.mean()),
            'average_confidence': float(np.mean(confidences))
        }
    }
    
    # 결과 저장
    analysis_file = os.path.join(save_dir, 'error_analysis.json')
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print(f"오류 분석 결과가 저장되었습니다: {analysis_file}")
    
    return analysis_results


def plot_confidence_distribution(probabilities: List[List[float]],
                                predictions: List[int],
                                labels: List[int],
                                save_path: str = 'confidence_distribution.png'):
    """신뢰도 분포 시각화"""
    
    confidences = [max(prob) for prob in probabilities]
    correct_mask = np.array(predictions) == np.array(labels)
    
    plt.figure(figsize=(12, 5))
    
    # 전체 신뢰도 분포
    plt.subplot(1, 2, 1)
    plt.hist([np.array(confidences)[correct_mask], 
              np.array(confidences)[~correct_mask]], 
             bins=20, alpha=0.7, 
             label=['Correct', 'Wrong'],
             color=['green', 'red'])
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Confidence Distribution')
    plt.legend()
    
    # 클래스별 신뢰도
    plt.subplot(1, 2, 2)
    for class_idx, class_name in enumerate(get_label_names()):
        class_mask = np.array(predictions) == class_idx
        if class_mask.sum() > 0:
            plt.hist(np.array(confidences)[class_mask], 
                    bins=15, alpha=0.7, 
                    label=f'Predicted {class_name}')
    
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Confidence by Predicted Class')
    plt.legend()
    
    plt.tight_layout()
    
    # 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"신뢰도 분포가 저장되었습니다: {save_path}")


if __name__ == "__main__":
    # 테스트 코드
    print("CoLA 평가 모듈 테스트...")
    
    # 더미 데이터로 메트릭 계산 테스트
    dummy_predictions = [0, 1, 1, 0, 1, 0, 0, 1]
    dummy_labels = [0, 1, 0, 0, 1, 1, 0, 1]
    
    metrics = compute_metrics(dummy_predictions, dummy_labels)
    print("테스트 메트릭:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # 혼동 행렬 테스트
    cm = confusion_matrix(dummy_labels, dummy_predictions)
    plot_confusion_matrix(cm, get_label_names(), 'test_confusion_matrix.png')
    
    print("테스트 완료!")