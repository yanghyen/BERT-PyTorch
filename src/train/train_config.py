"""
train.py 전용 설정 모듈.

학습 설정은 이 파일(`TRAIN_CONFIG`)에서 관리합니다.
"""

from __future__ import annotations

TRAIN_CONFIG = {
    # HF dataset (create_pretraining_instance.py 산출) 경로 (repo root 기준 상대경로)
    "dataset_dir": "data/pretraining_instances_spmask",
    # Hugging Face tokenizer 이름
    "tokenizer": "bert-base-uncased",
    # DataLoader / 학습 스펙
    "batch_size": 32,
    # step 기준 학습(요청: 1,000,000 step)
    "max_steps": 1_000_000,
    # 체크포인트 저장 주기(step). 100_000이면 100k마다 저장.
    "checkpoint_every_steps": 100_000,
    # LR scheduler: warmup 후 linear decay
    "warmup_steps": 10_000,
    "lr": 1e-4,
    # AdamW 하이퍼파라미터 (BERT pretraining 권장값)
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "weight_decay": 0.01,
    # Ampere+ GPU에서 matmul/convolution TF32 사용으로 속도 향상 가능
    "allow_tf32": True,
    # grad clipping은 안정성에 도움되지만 약간의 연산 오버헤드가 있습니다.
    # 속도 우선이면 None으로 비활성화할 수 있습니다.
    "grad_clip_norm": 1.0,
    # 데이터 로딩 병렬화: CPU 코어 여유가 있으면 4~8까지 올려보세요.
    "num_workers": 12,
    # curriculum: 대부분은 짧게(128), 일부는 길게(512)
    # 주의: 이 전략을 쓰려면 pretraining_instances가 최소 512 길이로 생성돼 있어야 합니다.
    "use_curriculum": False,
    "seq_len_short": 128,
    "seq_len_long": 512,
    "short_seq_prob": 0.9,  # 90% short, 10% long
    # 긴 시퀀스는 메모리/속도 부담이 커서 배치 크기를 별도로 두는 게 실무에서 흔합니다.
    # None이면 short 배치의 1/4로 자동 설정합니다.
    "batch_size_long": None,

    # 재현성(reproducibility)을 위한 시드 고정
    "seed": 42,
    # 연산/커널이 완전히 결정적이지 않은 경우가 있어도 학습이 멈추지 않도록
    # 일부 환경에서는 warn-only 형태로 동작하도록 처리합니다(구현은 train.py에서).
    # 속도 우선이면 False 권장 (deterministic=True는 커널 선택 폭이 줄어 느려질 수 있음)
    "deterministic": False,
    # 디버그 옵션
    "debug_masking": False,
    "debug_masking_batches": 3,
}

