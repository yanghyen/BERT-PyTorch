"""
MRPC 데이터셋 로딩 및 전처리 모듈
Microsoft Research Paraphrase Corpus - Binary classification (paraphrase/not_paraphrase)
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


class MRPCDataset(Dataset):
    """MRPC 데이터셋 클래스"""
    
    def __init__(self, 
                 data_dir: str = "./data/MRPC",
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
        self.sentence1s, self.sentence2s, self.labels = self._load_data()
        
    def _download_data(self):
        """MRPC 데이터셋 다운로드"""
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 이미 다운로드되어 있으면 스킵
        if os.path.exists(os.path.join(self.data_dir, "train.tsv")):
            return
        
        print("MRPC 데이터셋을 다운로드하는 중...")
        
        # 여러 다운로드 방법 시도
        success = False
        
        # 방법 1: HuggingFace datasets 사용
        success = self._download_via_huggingface()
        
        if not success:
            # 방법 2: 직접 GLUE 스크립트 사용
            success = self._download_via_glue_script()
        
        if not success:
            # 방법 3: 대안 URL들 시도
            success = self._download_via_alternative_urls()
        
        if not success:
            # 방법 4: 수동 데이터 생성 (개발/테스트용)
            success = self._create_sample_data()
        
        if not success:
            print("모든 자동 다운로드 방법이 실패했습니다.")
            print("수동으로 데이터를 다운로드하여 다음 위치에 배치해주세요:")
            print(f"  - {self.data_dir}/train.tsv")
            print(f"  - {self.data_dir}/dev.tsv")
            print(f"  - {self.data_dir}/test.tsv")
            print("\nMRPC 데이터는 다음 링크에서 다운로드할 수 있습니다:")
            print("https://www.microsoft.com/en-us/download/details.aspx?id=52398")
    
    def _download_via_huggingface(self) -> bool:
        """HuggingFace datasets를 통한 다운로드"""
        try:
            print("HuggingFace datasets를 통한 다운로드 시도 중...")
            
            # datasets 라이브러리 import 시도
            try:
                from datasets import load_dataset
            except ImportError:
                print("datasets 라이브러리가 설치되지 않음. pip install datasets로 설치하세요.")
                return False
            
            # GLUE MRPC 데이터셋 로드
            dataset = load_dataset("glue", "mrpc")
            
            # 각 split을 TSV 파일로 저장
            splits = [("train", "train.tsv"), ("validation", "dev.tsv"), ("test", "test.tsv")]
            
            for split_name, filename in splits:
                if split_name in dataset:
                    split_data = dataset[split_name]
                    filepath = os.path.join(self.data_dir, filename)
                    
                    with open(filepath, 'w', encoding='utf-8', newline='') as f:
                        if split_name == "test":
                            # 테스트 데이터는 레이블이 없을 수 있음
                            f.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
                            for i, example in enumerate(split_data):
                                sentence1 = example['sentence1'].replace('\t', ' ').replace('\n', ' ')
                                sentence2 = example['sentence2'].replace('\t', ' ').replace('\n', ' ')
                                f.write(f"{i}\t{i*2}\t{i*2+1}\t{sentence1}\t{sentence2}\n")
                        else:
                            # 훈련/검증 데이터
                            f.write("Quality\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
                            for i, example in enumerate(split_data):
                                label = example['label']
                                sentence1 = example['sentence1'].replace('\t', ' ').replace('\n', ' ')
                                sentence2 = example['sentence2'].replace('\t', ' ').replace('\n', ' ')
                                f.write(f"{label}\t{i*2}\t{i*2+1}\t{sentence1}\t{sentence2}\n")
            
            print("HuggingFace datasets를 통한 다운로드 완료!")
            return True
            
        except Exception as e:
            print(f"HuggingFace datasets 다운로드 실패: {e}")
            return False
    
    def _download_via_glue_script(self) -> bool:
        """GLUE 공식 스크립트를 통한 다운로드"""
        try:
            print("GLUE 공식 스크립트를 통한 다운로드 시도 중...")
            
            import subprocess
            import tempfile
            
            # GLUE 다운로드 스크립트 다운로드
            script_url = "https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py"
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                script_content = urllib.request.urlopen(script_url).read().decode('utf-8')
                f.write(script_content)
                script_path = f.name
            
            # 스크립트 실행
            result = subprocess.run([
                'python', script_path, 
                '--data_dir', self.data_dir, 
                '--tasks', 'MRPC'
            ], capture_output=True, text=True)
            
            # 임시 스크립트 파일 삭제
            os.unlink(script_path)
            
            if result.returncode == 0:
                print("GLUE 스크립트를 통한 다운로드 완료!")
                return True
            else:
                print(f"GLUE 스크립트 실행 실패: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"GLUE 스크립트 다운로드 실패: {e}")
            return False
    
    def _download_via_alternative_urls(self) -> bool:
        """대안 URL들을 통한 다운로드"""
        alternative_urls = [
            "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FMRPC.zip?alt=media&token=1d2c5c0d-b2b6-4b1a-9c4e-2e0c8b5d5e5e",
            "https://github.com/nyu-mll/GLUE-baselines/raw/master/glue_data/MRPC.zip",
            "https://dl.fbaipublicfiles.com/glue/data/MRPC.zip"
        ]
        
        for url in alternative_urls:
            try:
                print(f"대안 URL 시도 중: {url}")
                
                # User-Agent 헤더 추가
                req = urllib.request.Request(url)
                req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
                
                zip_path = os.path.join(self.data_dir, "MRPC.zip")
                
                with urllib.request.urlopen(req) as response:
                    with open(zip_path, 'wb') as f:
                        f.write(response.read())
                
                # 압축 해제
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
                
                # 중첩 폴더 정리
                self._organize_extracted_files()
                
                # 압축 파일 삭제
                os.remove(zip_path)
                
                print(f"대안 URL을 통한 다운로드 완료: {url}")
                return True
                
            except Exception as e:
                print(f"URL {url} 실패: {e}")
                continue
        
        return False
    
    def _organize_extracted_files(self):
        """압축 해제된 파일들 정리"""
        # 중첩 폴더 정리 (MRPC/MRPC -> MRPC)
        nested_dir = os.path.join(self.data_dir, "MRPC")
        if os.path.exists(nested_dir):
            import shutil
            # 임시 디렉토리로 파일들 이동
            temp_dir = os.path.join(self.data_dir, "temp_mrpc")
            shutil.move(nested_dir, temp_dir)
            
            # 파일들을 원래 위치로 이동
            for item in os.listdir(temp_dir):
                shutil.move(os.path.join(temp_dir, item), self.data_dir)
            
            # 임시 디렉토리 삭제
            os.rmdir(temp_dir)
    
    def _create_sample_data(self) -> bool:
        """개발/테스트용 샘플 데이터 생성"""
        try:
            print("개발용 샘플 데이터 생성 중...")
            
            # 샘플 데이터 (실제 MRPC 형식)
            sample_data = {
                "train.tsv": [
                    ["Quality", "#1 ID", "#2 ID", "#1 String", "#2 String"],
                    ["1", "1", "2", "Amrozi accused his brother, whom he called \"the witness\", of deliberately distorting his evidence.", "Referring to him as only \"the witness\", Amrozi accused his brother of deliberately distorting his evidence."],
                    ["0", "3", "4", "Yucaipa owned Dominick's before selling the chain to Safeway in 1998 for $2.5 billion.", "Yucaipa bought Dominick's in 1995 for $693 million and sold it to Safeway for $1.8 billion in 1998."],
                    ["1", "5", "6", "They had published an advertisement on the Internet on June 10, offering the cargo for sale.", "On June 10, the ship's owners had published an advertisement on the Internet, offering the explosives for sale."],
                    ["0", "7", "8", "Around 0335 GMT, Tab shares were up 19 cents, or 4.4%, at A$4.56.", "Tab shares jumped 20 cents, or 4.6%, to set a record closing high at A$4.57."]
                ],
                "dev.tsv": [
                    ["Quality", "#1 ID", "#2 ID", "#1 String", "#2 String"],
                    ["1", "1", "2", "The DVD-CCA then appealed to the state Supreme Court.", "The DVD CCA appealed that decision to the U.S. Supreme Court."],
                    ["0", "3", "4", "The stock rose $2.11, or about 11 percent, to close Friday at $21.51 on the New York Stock Exchange.", "PG&E Corp. shares jumped $1.63 or 8 percent to $21.03 on the New York Stock Exchange on Friday."]
                ],
                "test.tsv": [
                    ["index", "#1 ID", "#2 ID", "#1 String", "#2 String"],
                    ["0", "1", "2", "The bird is bathing in the sink.", "Birdie is washing itself in the water basin."],
                    ["1", "3", "4", "The old man the boat.", "The elderly are steering the vessel."]
                ]
            }
            
            for filename, data in sample_data.items():
                filepath = os.path.join(self.data_dir, filename)
                with open(filepath, 'w', encoding='utf-8', newline='') as f:
                    for row in data:
                        f.write('\t'.join(row) + '\n')
            
            print("샘플 데이터 생성 완료! (개발/테스트용)")
            print("실제 사용을 위해서는 공식 MRPC 데이터셋을 다운로드하세요.")
            return True
            
        except Exception as e:
            print(f"샘플 데이터 생성 실패: {e}")
            return False
    
    def _load_data(self) -> Tuple[List[str], List[str], List[int]]:
        """데이터 파일 로드"""
        if self.split == "train":
            file_path = os.path.join(self.data_dir, "train.tsv")
        elif self.split == "dev":
            file_path = os.path.join(self.data_dir, "dev.tsv")
        elif self.split == "test":
            file_path = os.path.join(self.data_dir, "test.tsv")
        else:
            raise ValueError(f"지원하지 않는 split: {self.split}")
        
        sentence1s = []
        sentence2s = []
        labels = []
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")
        
        # CSV field size limit 증가
        csv.field_size_limit(sys.maxsize)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                # MRPC 데이터 형식: Quality, #1 ID, #2 ID, #1 String, #2 String
                sentence1 = row.get('#1 String', row.get('sentence1', ''))
                sentence2 = row.get('#2 String', row.get('sentence2', ''))
                label_str = row.get('Quality', row.get('label', ''))
                
                sentence1s.append(sentence1)
                sentence2s.append(sentence2)
                
                # 레이블 처리 (test 데이터는 레이블이 없을 수 있음)
                if label_str and label_str.isdigit():
                    labels.append(int(label_str))
                else:
                    labels.append(-1)  # test 데이터용 더미 레이블
        
        print(f"{self.split} 데이터 로드 완료: {len(sentence1s)}개 샘플")
        
        # 클래스 분포 출력 (train/dev인 경우)
        if self.split in ['train', 'dev'] and labels and labels[0] != -1:
            label_counts = {0: 0, 1: 0}
            for label in labels:
                if label in label_counts:
                    label_counts[label] += 1
            print(f"  클래스 분포 - Not paraphrase: {label_counts[0]}, Paraphrase: {label_counts[1]}")
        
        return sentence1s, sentence2s, labels
    
    def __len__(self) -> int:
        return len(self.sentence1s)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sentence1 = self.sentence1s[idx]
        sentence2 = self.sentence2s[idx]
        label = self.labels[idx]
        
        # 두 문장을 [CLS] sentence1 [SEP] sentence2 [SEP] 형태로 토크나이징
        encoding = self.tokenizer(
            sentence1,
            sentence2,
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


def create_data_loaders(data_dir: str = "./data/MRPC",
                       batch_size: int = 32,
                       max_length: int = 128,
                       tokenizer_name: str = "bert-base-uncased",
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """데이터 로더 생성"""
    
    # 데이터셋 생성
    train_dataset = MRPCDataset(
        data_dir=data_dir,
        split="train",
        max_length=max_length,
        tokenizer_name=tokenizer_name
    )
    
    dev_dataset = MRPCDataset(
        data_dir=data_dir,
        split="dev",
        max_length=max_length,
        tokenizer_name=tokenizer_name
    )
    
    test_dataset = MRPCDataset(
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
    return ["not_paraphrase", "paraphrase"]


def get_num_labels() -> int:
    """레이블 수 반환"""
    return 2


def analyze_dataset_statistics(data_dir: str = "./data/MRPC"):
    """데이터셋 통계 분석"""
    
    print("MRPC 데이터셋 통계 분석")
    print("=" * 40)
    
    for split in ['train', 'dev', 'test']:
        try:
            dataset = MRPCDataset(data_dir=data_dir, split=split)
            
            print(f"\n{split.upper()} 데이터:")
            print(f"  샘플 수: {len(dataset)}")
            
            if dataset.labels and dataset.labels[0] != -1:
                # 레이블 분포
                label_counts = {0: 0, 1: 0}
                for label in dataset.labels:
                    if label in label_counts:
                        label_counts[label] += 1
                
                total = sum(label_counts.values())
                print(f"  Not paraphrase: {label_counts[0]} ({label_counts[0]/total*100:.1f}%)")
                print(f"  Paraphrase: {label_counts[1]} ({label_counts[1]/total*100:.1f}%)")
                
                # 문장 길이 통계
                sentence1_lengths = [len(s.split()) if s else 0 for s in dataset.sentence1s]
                sentence2_lengths = [len(s.split()) if s else 0 for s in dataset.sentence2s]
                
                print(f"  문장1 평균 길이: {sum(sentence1_lengths)/len(sentence1_lengths):.1f} 단어")
                print(f"  문장2 평균 길이: {sum(sentence2_lengths)/len(sentence2_lengths):.1f} 단어")
                print(f"  문장1 최대 길이: {max(sentence1_lengths)} 단어")
                print(f"  문장2 최대 길이: {max(sentence2_lengths)} 단어")
            
        except Exception as e:
            print(f"  {split} 데이터 로드 실패: {e}")


if __name__ == "__main__":
    # 테스트 코드
    print("MRPC 데이터셋 테스트 중...")
    
    # 데이터셋 통계 분석
    try:
        analyze_dataset_statistics()
    except Exception as e:
        print(f"통계 분석 실패: {e}")
    
    print("\n" + "=" * 40)
    print("데이터 로더 테스트")
    
    # 데이터 로더 생성
    try:
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
            
            # 토크나이저로 디코딩해서 실제 텍스트 확인
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
            input_ids = batch['input_ids'][0]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            
            # [SEP] 토큰 위치 찾기
            sep_indices = [i for i, token in enumerate(tokens) if token == '[SEP]']
            if len(sep_indices) >= 2:
                sentence1_tokens = tokens[1:sep_indices[0]]  # [CLS] 다음부터 첫 번째 [SEP]까지
                sentence2_tokens = tokens[sep_indices[0]+1:sep_indices[1]]  # 첫 번째 [SEP] 다음부터 두 번째 [SEP]까지
                
                sentence1 = tokenizer.convert_tokens_to_string(sentence1_tokens)
                sentence2 = tokenizer.convert_tokens_to_string(sentence2_tokens)
                
                print(f"  문장1: {sentence1}")
                print(f"  문장2: {sentence2}")
                print(f"  레이블: {get_label_names()[batch['labels'][0].item()]}")
            
            break
        
        print(f"\n데이터셋 크기:")
        print(f"  Train: {len(train_loader.dataset)}")
        print(f"  Dev: {len(dev_loader.dataset)}")
        print(f"  Test: {len(test_loader.dataset)}")
        
    except Exception as e:
        print(f"데이터 로더 테스트 실패: {e}")
        import traceback
        traceback.print_exc()