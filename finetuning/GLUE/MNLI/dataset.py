"""
MNLI 데이터셋 로딩 및 전처리 모듈
Multi-Genre Natural Language Inference - 3-class classification (entailment/contradiction/neutral)
"""

import os
import sys
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import urllib.request
import zipfile
from typing import List, Tuple, Dict, Optional


class MNLIDataset(Dataset):
    """MNLI 데이터셋 클래스"""
    
    def __init__(self, 
                 data_dir: str = "./data/MNLI",
                 split: str = "train",
                 max_length: int = 256,
                 tokenizer_name: str = "bert-base-uncased",
                 use_matched: bool = True):
        """
        Args:
            data_dir: 데이터가 저장될 디렉토리
            split: 'train', 'dev_matched', 'dev_mismatched', 'test_matched', 'test_mismatched' 중 하나
            max_length: 최대 시퀀스 길이
            tokenizer_name: 사용할 토크나이저 이름
            use_matched: dev/test에서 matched 버전 사용 여부 (False면 mismatched 사용)
        """
        self.data_dir = data_dir
        self.split = split
        self.max_length = max_length
        self.use_matched = use_matched
        
        # 토크나이저 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # 데이터 다운로드 및 로드
        self._download_data()
        self.premises, self.hypotheses, self.labels = self._load_data()
        
    def _download_data(self):
        """MNLI 데이터셋 다운로드"""
        os.makedirs(self.data_dir, exist_ok=True)
        
        # GLUE 데이터 URL
        base_url = "https://dl.fbaipublicfiles.com/glue/data/MNLI.zip"
        zip_path = os.path.join(self.data_dir, "MNLI.zip")
        
        # 이미 다운로드되어 있으면 스킵
        if os.path.exists(os.path.join(self.data_dir, "train.tsv")):
            return
            
        print(f"MNLI 데이터셋을 다운로드하는 중... {base_url}")
        try:
            urllib.request.urlretrieve(base_url, zip_path)
            
            # 압축 해제
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            # 중첩 폴더 정리 (MNLI/MNLI -> MNLI)
            nested_dir = os.path.join(self.data_dir, "MNLI")
            if os.path.exists(nested_dir):
                import shutil
                # 임시 디렉토리로 파일들 이동
                temp_dir = os.path.join(self.data_dir, "temp_mnli")
                shutil.move(nested_dir, temp_dir)
                
                # 파일들을 원래 위치로 이동
                for item in os.listdir(temp_dir):
                    shutil.move(os.path.join(temp_dir, item), self.data_dir)
                
                # 임시 디렉토리 삭제
                os.rmdir(temp_dir)
            
            # 압축 파일 삭제
            os.remove(zip_path)
            print("다운로드 완료!")
            
        except Exception as e:
            print(f"데이터 다운로드 실패: {e}")
            print("수동으로 데이터를 다운로드하여 다음 위치에 배치해주세요:")
            print(f"  - {self.data_dir}/train.tsv")
            print(f"  - {self.data_dir}/dev_matched.tsv")
            print(f"  - {self.data_dir}/dev_mismatched.tsv")
            print(f"  - {self.data_dir}/test_matched.tsv")
            print(f"  - {self.data_dir}/test_mismatched.tsv")
    
    def _get_file_path(self) -> str:
        """split에 따른 파일 경로 반환"""
        if self.split == "train":
            return os.path.join(self.data_dir, "train.tsv")
        elif self.split == "dev":
            if self.use_matched:
                return os.path.join(self.data_dir, "dev_matched.tsv")
            else:
                return os.path.join(self.data_dir, "dev_mismatched.tsv")
        elif self.split == "test":
            if self.use_matched:
                return os.path.join(self.data_dir, "test_matched.tsv")
            else:
                return os.path.join(self.data_dir, "test_mismatched.tsv")
        else:
            # 직접 파일명 지정한 경우
            return os.path.join(self.data_dir, f"{self.split}.tsv")
    
    def _load_data(self) -> Tuple[List[str], List[str], List[int]]:
        """데이터 파일 로드"""
        file_path = self._get_file_path()
        
        premises = []
        hypotheses = []
        labels = []
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")
        
        # 레이블 매핑
        label_map = {
            'entailment': 2,
            'contradiction': 0,
            'neutral': 1
        }
        
        # CSV field size limit 증가
        csv.field_size_limit(sys.maxsize)
        
        skipped_count = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row_idx, row in enumerate(reader):
                # MNLI 데이터 형식에 따라 컬럼명이 다를 수 있음
                premise = row.get('sentence1', row.get('premise', ''))
                hypothesis = row.get('sentence2', row.get('hypothesis', ''))
                label_str = row.get('gold_label', row.get('label', ''))
                
                # 데이터 품질 검사: 오염된 샘플 필터링
                if self._is_corrupted_sample(premise, hypothesis, label_str):
                    skipped_count += 1
                    continue
                
                premises.append(premise)
                hypotheses.append(hypothesis)
                
                # 레이블 처리 (test 데이터는 레이블이 없을 수 있음)
                if label_str in label_map:
                    labels.append(label_map[label_str])
                else:
                    # train/dev에서 라벨이 없으면 스킵 (test는 허용)
                    if self.split in ['train', 'dev', 'dev_matched', 'dev_mismatched']:
                        skipped_count += 1
                        premises.pop()  # 방금 추가한 premise 제거
                        hypotheses.pop()  # 방금 추가한 hypothesis 제거
                        continue
                    else:
                        labels.append(-1)  # test 데이터용 더미 레이벨
        
        if skipped_count > 0:
            print(f"{self.split} 데이터 로드 완료: {len(premises)}개 샘플 (오염된 샘플 {skipped_count}개 제외)")
        else:
            print(f"{self.split} 데이터 로드 완료: {len(premises)}개 샘플")
        return premises, hypotheses, labels
    
    def _is_corrupted_sample(self, premise: str, hypothesis: str, label_str: str) -> bool:
        """오염된 샘플인지 검사"""
        # 1. premise나 hypothesis에 탭 문자가 있으면 오염된 것으로 간주
        if '\t' in premise or '\t' in hypothesis:
            return True
        
        # 2. hypothesis가 라벨 값('contradiction', 'neutral', 'entailment')이면 오염된 것
        if hypothesis.strip().lower() in ['contradiction', 'neutral', 'entailment']:
            return True
        
        # 3. premise나 hypothesis가 비정상적으로 짧거나 길면 의심스러움
        if len(premise.strip()) < 5 or len(hypothesis.strip()) < 5:
            return True
        
        # 4. premise나 hypothesis에 개행 문자가 있으면 오염 가능성
        if '\n' in premise or '\n' in hypothesis:
            return True
        
        return False
    
    def __len__(self) -> int:
        return len(self.premises)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        premise = self.premises[idx]
        hypothesis = self.hypotheses[idx]
        label = self.labels[idx]
        
        # 두 문장을 [CLS] premise [SEP] hypothesis [SEP] 형태로 토크나이징
        encoding = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_data_loaders(data_dir: str = "./data/MNLI",
                       batch_size: int = 16,
                       max_length: int = 256,
                       tokenizer_name: str = "bert-base-uncased",
                       num_workers: int = 4,
                       use_matched: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """데이터 로더 생성"""
    
    # 데이터셋 생성
    train_dataset = MNLIDataset(
        data_dir=data_dir,
        split="train",
        max_length=max_length,
        tokenizer_name=tokenizer_name
    )
    
    dev_dataset = MNLIDataset(
        data_dir=data_dir,
        split="dev",
        max_length=max_length,
        tokenizer_name=tokenizer_name,
        use_matched=use_matched
    )
    
    test_dataset = MNLIDataset(
        data_dir=data_dir,
        split="test",
        max_length=max_length,
        tokenizer_name=tokenizer_name,
        use_matched=use_matched
    )
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, dev_loader, test_loader


def create_matched_mismatched_loaders(data_dir: str = "./data/MNLI",
                                     batch_size: int = 16,
                                     max_length: int = 256,
                                     tokenizer_name: str = "bert-base-uncased",
                                     num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Matched와 Mismatched 데이터 로더 모두 생성"""
    
    train_dataset = MNLIDataset(
        data_dir=data_dir,
        split="train",
        max_length=max_length,
        tokenizer_name=tokenizer_name
    )
    
    dev_matched_dataset = MNLIDataset(
        data_dir=data_dir,
        split="dev",
        max_length=max_length,
        tokenizer_name=tokenizer_name,
        use_matched=True
    )
    
    dev_mismatched_dataset = MNLIDataset(
        data_dir=data_dir,
        split="dev",
        max_length=max_length,
        tokenizer_name=tokenizer_name,
        use_matched=False
    )
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    dev_matched_loader = DataLoader(
        dev_matched_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    dev_mismatched_loader = DataLoader(
        dev_mismatched_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, dev_matched_loader, dev_mismatched_loader


def get_label_names() -> List[str]:
    """레이블 이름 반환"""
    return ["contradiction", "neutral", "entailment"]


def get_num_labels() -> int:
    """레이블 수 반환"""
    return 3


if __name__ == "__main__":
    # 테스트 코드
    print("MNLI 데이터셋 테스트 중...")
    
    # 데이터 로더 생성
    train_loader, dev_loader, test_loader = create_data_loaders(
        batch_size=8,
        max_length=128
    )
    
    # 첫 번째 배치 확인
    for batch in train_loader:
        print("배치 형태:")
        for key, value in batch.items():
            print(f"  {key}: {value.shape}")
        
        print("\n첫 번째 샘플:")
        print(f"  입력 텍스트 길이: {batch['input_ids'][0].sum()}")
        print(f"  레이블: {batch['labels'][0].item()}")
        
        # 토크나이저로 디코딩해서 실제 텍스트 확인
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        input_ids = batch['input_ids'][0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        # [SEP] 토큰 위치 찾기
        sep_indices = [i for i, token in enumerate(tokens) if token == '[SEP]']
        if len(sep_indices) >= 2:
            premise_tokens = tokens[1:sep_indices[0]]  # [CLS] 다음부터 첫 번째 [SEP]까지
            hypothesis_tokens = tokens[sep_indices[0]+1:sep_indices[1]]  # 첫 번째 [SEP] 다음부터 두 번째 [SEP]까지
            
            premise = tokenizer.convert_tokens_to_string(premise_tokens)
            hypothesis = tokenizer.convert_tokens_to_string(hypothesis_tokens)
            
            print(f"  Premise: {premise}")
            print(f"  Hypothesis: {hypothesis}")
            print(f"  Label: {get_label_names()[batch['labels'][0].item()]}")
        
        break
    
    print(f"\n데이터셋 크기:")
    print(f"  Train: {len(train_loader.dataset)}")
    print(f"  Dev: {len(dev_loader.dataset)}")
    print(f"  Test: {len(test_loader.dataset)}")
    
    # Matched/Mismatched 데이터 로더 테스트
    print("\nMatched/Mismatched 데이터 로더 테스트...")
    try:
        train_loader, dev_matched_loader, dev_mismatched_loader = create_matched_mismatched_loaders(
            batch_size=8,
            max_length=128
        )
        print(f"  Train: {len(train_loader.dataset)}")
        print(f"  Dev Matched: {len(dev_matched_loader.dataset)}")
        print(f"  Dev Mismatched: {len(dev_mismatched_loader.dataset)}")
    except Exception as e:
        print(f"  오류: {e}")