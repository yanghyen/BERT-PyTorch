"""
SST-2 데이터셋 로딩 및 전처리 모듈
Stanford Sentiment Treebank - Binary classification (positive/negative)
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


class SST2Dataset(Dataset):
    """SST-2 데이터셋 클래스"""
    
    def __init__(self, 
                 data_dir: str = "./data/SST-2",
                 split: str = "train",
                 max_length: int = 128,
                 tokenizer_name: str = "bert-base-uncased"):
        """
        Args:
            data_dir: 데이터가 저장될 디렉토리
            split: 'train', 'dev', 'test' 중 하나
            max_length: 최대 시퀀스 길이
            tokenizer_name: 사용할 토크나이저 이름
        """
        self.data_dir = data_dir
        self.split = split
        self.max_length = max_length
        
        # 토크나이저 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # 데이터 다운로드 및 로드
        self._download_data()
        self.texts, self.labels = self._load_data()
        
    def _download_data(self):
        """SST-2 데이터셋 다운로드"""
        os.makedirs(self.data_dir, exist_ok=True)
        
        # GLUE 데이터 URL
        base_url = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
        zip_path = os.path.join(self.data_dir, "SST-2.zip")
        
        # 이미 다운로드되어 있으면 스킵
        if os.path.exists(os.path.join(self.data_dir, "train.tsv")):
            return
            
        print(f"SST-2 데이터셋을 다운로드하는 중... {base_url}")
        try:
            urllib.request.urlretrieve(base_url, zip_path)
            
            # 압축 해제
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            # 중첩 폴더 정리 (SST-2/SST-2 -> SST-2)
            nested_dir = os.path.join(self.data_dir, "SST-2")
            if os.path.exists(nested_dir):
                import shutil
                # 임시 디렉토리로 파일들 이동
                temp_dir = os.path.join(self.data_dir, "temp_sst2")
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
            print(f"  - {self.data_dir}/dev.tsv")
            print(f"  - {self.data_dir}/test.tsv")
    
    def _load_data(self) -> Tuple[List[str], List[int]]:
        """데이터 파일 로드"""
        if self.split == "train":
            file_path = os.path.join(self.data_dir, "train.tsv")
        elif self.split == "dev":
            file_path = os.path.join(self.data_dir, "dev.tsv")
        elif self.split == "test":
            file_path = os.path.join(self.data_dir, "test.tsv")
        else:
            raise ValueError(f"지원하지 않는 split: {self.split}")
        
        texts = []
        labels = []
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")
        
        # CSV field size limit 증가
        csv.field_size_limit(sys.maxsize)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                texts.append(row['sentence'])
                # test 데이터는 레이블이 없을 수 있음
                if 'label' in row and row['label'] != '':
                    labels.append(int(row['label']))
                else:
                    labels.append(-1)  # test 데이터용 더미 레이블
        
        print(f"{self.split} 데이터 로드 완료: {len(texts)}개 샘플")
        return texts, labels
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 토크나이징
        encoding = self.tokenizer(
            text,
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


def create_data_loaders(data_dir: str = "./data/SST-2",
                       batch_size: int = 32,
                       max_length: int = 128,
                       tokenizer_name: str = "bert-base-uncased",
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """데이터 로더 생성"""
    
    # 데이터셋 생성
    train_dataset = SST2Dataset(
        data_dir=data_dir,
        split="train",
        max_length=max_length,
        tokenizer_name=tokenizer_name
    )
    
    dev_dataset = SST2Dataset(
        data_dir=data_dir,
        split="dev",
        max_length=max_length,
        tokenizer_name=tokenizer_name
    )
    
    test_dataset = SST2Dataset(
        data_dir=data_dir,
        split="test",
        max_length=max_length,
        tokenizer_name=tokenizer_name
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


def get_label_names() -> List[str]:
    """레이블 이름 반환"""
    return ["negative", "positive"]


def get_num_labels() -> int:
    """레이블 수 반환"""
    return 2


if __name__ == "__main__":
    # 테스트 코드
    print("SST-2 데이터셋 테스트 중...")
    
    # 데이터 로더 생성
    train_loader, dev_loader, test_loader = create_data_loaders(
        batch_size=8,
        max_length=64
    )
    
    # 첫 번째 배치 확인
    for batch in train_loader:
        print("배치 형태:")
        for key, value in batch.items():
            print(f"  {key}: {value.shape}")
        
        print("\n첫 번째 샘플:")
        print(f"  입력 텍스트 길이: {batch['input_ids'][0].sum()}")
        print(f"  레이블: {batch['labels'][0].item()}")
        break
    
    print(f"\n데이터셋 크기:")
    print(f"  Train: {len(train_loader.dataset)}")
    print(f"  Dev: {len(dev_loader.dataset)}")
    print(f"  Test: {len(test_loader.dataset)}")