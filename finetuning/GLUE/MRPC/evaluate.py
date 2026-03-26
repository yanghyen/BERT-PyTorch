"""
BERT MRPC 모델 평가 모듈
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
    """분류 메트릭 계산 (MRPC 2-class)"""
    
    # 기본 메트릭
    accuracy = accuracy_score(labels, predictions)
    
    # 이진 분류를 위한 메트릭 (positive class = 1: paraphrase)
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
                       texts: Optional[List[Tuple[str, str]]] = None,
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
        if texts and idx < len(texts):
            sample_info['sentence1'] = texts[idx][0]
            sample_info['sentence2'] = texts[idx][1]
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
    confidence_bins = np.linspace(0.5, 1.0, 11)  # 2-class이므로 0.5부터 시작
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
    
    # Paraphrase 특화 분석
    paraphrase_analysis = analyze_paraphrase_patterns(predictions, labels, probabilities)
    
    analysis = {
        'confusion_matrix': cm.tolist(),
        'wrong_samples': wrong_samples,
        'class_analysis': class_analysis,
        'confidence_accuracy': confidence_accuracy,
        'paraphrase_analysis': paraphrase_analysis,
        'average_confidence': float(np.mean(confidences)),
        'correct_confidence': float(np.mean([confidences[i] for i in range(len(confidences)) if correct_mask[i]])),
        'wrong_confidence': float(np.mean([confidences[i] for i in range(len(confidences)) if not correct_mask[i]]))
    }
    
    # 분석 결과 저장
    with open(os.path.join(save_dir, 'analysis.json'), 'w') as f:
        json.dump(analysis, f, indent=2)
    
    return analysis


def analyze_paraphrase_patterns(predictions: List[int], 
                               labels: List[int], 
                               probabilities: List[List[float]]) -> Dict:
    """Paraphrase 패턴 분석"""
    
    # True Positive, False Positive, True Negative, False Negative
    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
    
    # Paraphrase 클래스 (1)에 대한 신뢰도 분석
    paraphrase_confidences = [prob[1] for prob in probabilities]
    
    # 실제 paraphrase vs 예측 paraphrase 신뢰도 비교
    true_paraphrase_conf = [paraphrase_confidences[i] for i in range(len(labels)) if labels[i] == 1]
    false_paraphrase_conf = [paraphrase_confidences[i] for i in range(len(labels)) if labels[i] == 0]
    
    return {
        'confusion_details': {
            'true_positive': tp,
            'false_positive': fp,
            'true_negative': tn,
            'false_negative': fn
        },
        'paraphrase_confidence_stats': {
            'true_paraphrase_avg_conf': float(np.mean(true_paraphrase_conf)) if true_paraphrase_conf else 0.0,
            'false_paraphrase_avg_conf': float(np.mean(false_paraphrase_conf)) if false_paraphrase_conf else 0.0,
            'true_paraphrase_std_conf': float(np.std(true_paraphrase_conf)) if true_paraphrase_conf else 0.0,
            'false_paraphrase_std_conf': float(np.std(false_paraphrase_conf)) if false_paraphrase_conf else 0.0
        }
    }


def plot_confusion_matrix(cm: np.ndarray, 
                         class_names: List[str],
                         save_path: str = './confusion_matrix.png'):
    """혼동 행렬 시각화"""
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('MRPC Confusion Matrix')
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
    colors = ['lightcoral', 'lightgreen']
    bars1 = ax1.bar(class_names, accuracies, color=colors)
    ax1.set_title('Accuracy by Class')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    
    # 막대 위에 값 표시
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # 클래스별 평균 신뢰도
    bars2 = ax2.bar(class_names, confidences, color=colors)
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
    plt.hist(correct_confidences, bins=20, alpha=0.7, label='Correct', color='green', density=True)
    plt.hist(wrong_confidences, bins=20, alpha=0.7, label='Wrong', color='red', density=True)
    plt.xlabel('Confidence Score')
    plt.ylabel('Density')
    plt.title('Confidence Score Distribution (MRPC)')
    plt.legend()
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Random Guess (0.5)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_paraphrase_analysis(paraphrase_analysis: Dict,
                            save_path: str = './paraphrase_analysis.png'):
    """Paraphrase 분석 시각화"""
    
    confusion_details = paraphrase_analysis['confusion_details']
    
    # 혼동 행렬 세부 분석
    labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    values = [confusion_details['true_negative'], confusion_details['false_positive'],
              confusion_details['false_negative'], confusion_details['true_positive']]
    colors = ['lightblue', 'lightcoral', 'orange', 'lightgreen']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 혼동 행렬 세부 분석 파이 차트
    ax1.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('MRPC Prediction Distribution')
    
    # Paraphrase 신뢰도 비교
    conf_stats = paraphrase_analysis['paraphrase_confidence_stats']
    categories = ['True Paraphrase', 'Not Paraphrase']
    avg_confs = [conf_stats['true_paraphrase_avg_conf'], conf_stats['false_paraphrase_avg_conf']]
    std_confs = [conf_stats['true_paraphrase_std_conf'], conf_stats['false_paraphrase_std_conf']]
    
    bars = ax2.bar(categories, avg_confs, yerr=std_confs, capsize=5, 
                   color=['lightgreen', 'lightcoral'], alpha=0.7)
    ax2.set_title('Average Paraphrase Confidence by True Class')
    ax2.set_ylabel('Confidence Score')
    ax2.set_ylim(0, 1)
    
    # 막대 위에 값 표시
    for bar, conf in zip(bars, avg_confs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{conf:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """메인 평가 함수"""
    
    # 설정
    config = {
        'checkpoint_path': './checkpoints/best_model.pt',
        'data_dir': './data/MRPC',
        'batch_size': 32,
        'max_length': 128,
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
        num_labels=2  # MRPC: 2개 클래스
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
        
        # 클래스별 성능 시각화
        analysis = analyze_predictions(
            val_results['predictions'],
            val_results['labels'],
            val_results['probabilities'],
            save_dir=config['save_dir']
        )
        
        plot_class_performance(
            analysis['class_analysis'],
            os.path.join(config['save_dir'], 'class_performance.png')
        )
        
        # 신뢰도 분포 시각화
        confidences = [max(prob) for prob in val_results['probabilities']]
        correct_mask = np.array(val_results['predictions']) == np.array(val_results['labels'])
        
        plot_confidence_distribution(
            confidences,
            correct_mask,
            os.path.join(config['save_dir'], 'confidence_distribution.png')
        )
        
        # Paraphrase 분석 시각화
        plot_paraphrase_analysis(
            analysis['paraphrase_analysis'],
            os.path.join(config['save_dir'], 'paraphrase_analysis.png')
        )
        
        print(f"\n분석 결과:")
        print(f"  평균 신뢰도: {analysis['average_confidence']:.4f}")
        print(f"  정답 신뢰도: {analysis['correct_confidence']:.4f}")
        print(f"  오답 신뢰도: {analysis['wrong_confidence']:.4f}")
        
        # Paraphrase 특화 분석 결과
        paraphrase_stats = analysis['paraphrase_analysis']['paraphrase_confidence_stats']
        print(f"\nParaphrase 분석:")
        print(f"  실제 Paraphrase 평균 신뢰도: {paraphrase_stats['true_paraphrase_avg_conf']:.4f}")
        print(f"  실제 Not Paraphrase 평균 신뢰도: {paraphrase_stats['false_paraphrase_avg_conf']:.4f}")
    
    # 결과 저장
    results_path = os.path.join(config['save_dir'], 'evaluation_results.json')
    with open(results_path, 'w') as f:
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