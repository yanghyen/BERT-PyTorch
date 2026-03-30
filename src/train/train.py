from __future__ import annotations

import csv
import os
import random
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
from datasets import load_from_disk
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizerFast
from transformers import get_linear_schedule_with_warmup

# numpy는 시드 고정에만 선택적으로 사용(없어도 동작)
try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None

# 스크립트 실행 위치에 상관없이 src/를 import 가능하게
import sys

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# `train.py` 바로 옆에 있는 `train_config.py`를 import 하기 위한 경로 추가
TRAIN_DIR = Path(__file__).resolve().parent
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

import train_config

from model.bert import BERT, BERTLM, BERTLM_NoNSP  # noqa: E402


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """학습 재현성을 위한 시드 고정."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # CUDA 환경에서 cublas 재현성 요구사항을 만족하도록 기본 workspace를 설정합니다.
        # (이미 설정된 값이 있으면 존중)
        if torch.cuda.is_available():
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # 일부 연산은 결정적 구현이 없어도 학습이 멈추지 않도록 warn-only로 동작시킵니다.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover
            pass
    else:
        torch.backends.cudnn.benchmark = True


# --- 3) Masked Language Modeling 마스킹 함수 ---
def mask_tokens(
    inputs: torch.Tensor,
    tokenizer,
    *,
    special_tokens_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = inputs.clone().long()
    device = labels.device

    # Special tokens은 mask하지 않음
    probability_matrix = torch.full(labels.shape, 0.15, device=device)

    if special_tokens_mask is None:
        # 한 문장(1D list) 전체를 한번에 넘겨야 함
        special_tokens_mask_list = tokenizer.get_special_tokens_mask(
            labels.tolist(), already_has_special_tokens=True
        )
        special_tokens_mask = torch.tensor(
            special_tokens_mask_list, dtype=torch.bool, device=device
        )
    else:
        # dataset에 저장된 마스크(0/1)를 bool로 변환
        special_tokens_mask = special_tokens_mask.to(device=device, dtype=torch.bool)

    # padding id가 special token으로 잡히지 않은 경우에 대비(정확도 안전장치)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        special_tokens_mask = special_tokens_mask | (labels == int(pad_token_id))

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Loss 계산에서 무시할 토큰

    inputs = inputs.clone().long()

    # masked 토큰들 기준으로 정확히 80/10/10으로 분기합니다.
    #
    # 1) masked 토큰 중 80% -> [MASK]
    # 2) 나머지 20% 중 절반(=10%/10%) -> random, 나머지 -> keep
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError("tokenizer.mask_token_id가 None입니다.")

    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8, device=device)).bool() & masked_indices
    )
    inputs[indices_replaced] = mask_token_id

    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5, device=device)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(
        len(tokenizer), labels.shape, dtype=torch.long, device=device
    )
    inputs[indices_random] = random_words[indices_random]

    # 나머지 (masked 토큰의 10%)는 원래 토큰 유지 (변경 없음)

    return inputs, labels


# --- 4) DataLoader용 Dataset 클래스 ---
class BertPretrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenized_dataset,
        tokenizer,
        *,
        seq_len: int | None = None,
        return_original_input_ids: bool = True,
    ):
        self.dataset = tokenized_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.return_original_input_ids = return_original_input_ids

        # option b를 통해 저장된 컬럼이 있으면 그걸 사용(=tokenizer.get_special_tokens_mask 호출 제거)
        self.has_special_tokens_mask = "special_tokens_mask" in getattr(
            tokenized_dataset, "column_names", []
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        if self.seq_len is None:
            input_ids_list = row["input_ids"]
            token_type_ids_list = row["token_type_ids"]
            attention_mask_list = row["attention_mask"]
            special_tokens_mask_list = (
                row["special_tokens_mask"] if self.has_special_tokens_mask else None
            )
        else:
            input_ids_list = row["input_ids"][: self.seq_len]
            token_type_ids_list = row["token_type_ids"][: self.seq_len]
            attention_mask_list = row["attention_mask"][: self.seq_len]
            special_tokens_mask_list = (
                row["special_tokens_mask"][: self.seq_len]
                if self.has_special_tokens_mask
                else None
            )

        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        original_input_ids = input_ids.clone() if self.return_original_input_ids else None
        token_type_ids = torch.tensor(token_type_ids_list, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask_list, dtype=torch.long)
        nsp_label = torch.tensor(row["nsp_label"], dtype=torch.long)

        special_tokens_mask_tensor = (
            torch.tensor(special_tokens_mask_list, dtype=torch.bool)
            if special_tokens_mask_list is not None
            else None
        )

        out = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "special_tokens_mask": special_tokens_mask_tensor,
            "nsp_labels": nsp_label,
        }
        if self.return_original_input_ids:
            out["original_input_ids"] = original_input_ids
        return out


class BertPretrainDatasetNoNSP(torch.utils.data.Dataset):
    """NSP 없이 MLM만 사용하는 BERT 사전학습용 데이터셋"""
    def __init__(
        self,
        tokenized_dataset,
        tokenizer,
        *,
        seq_len: int | None = None,
        return_original_input_ids: bool = True,
    ):
        self.dataset = tokenized_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.return_original_input_ids = return_original_input_ids

        # option b를 통해 저장된 컬럼이 있으면 그걸 사용(=tokenizer.get_special_tokens_mask 호출 제거)
        self.has_special_tokens_mask = "special_tokens_mask" in getattr(
            tokenized_dataset, "column_names", []
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        if self.seq_len is None:
            input_ids_list = row["input_ids"]
            attention_mask_list = row["attention_mask"]
            special_tokens_mask_list = (
                row["special_tokens_mask"] if self.has_special_tokens_mask else None
            )
        else:
            input_ids_list = row["input_ids"][: self.seq_len]
            attention_mask_list = row["attention_mask"][: self.seq_len]
            special_tokens_mask_list = (
                row["special_tokens_mask"][: self.seq_len]
                if self.has_special_tokens_mask
                else None
            )

        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        original_input_ids = input_ids.clone() if self.return_original_input_ids else None
        # NSP 없으므로 token_type_ids는 모두 0으로 설정 (단일 문장)
        token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask_list, dtype=torch.long)

        special_tokens_mask_tensor = (
            torch.tensor(special_tokens_mask_list, dtype=torch.bool)
            if special_tokens_mask_list is not None
            else None
        )

        out = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "special_tokens_mask": special_tokens_mask_tensor,
        }
        if self.return_original_input_ids:
            out["original_input_ids"] = original_input_ids
        return out


# --- 5) 학습 루프 ---
if __name__ == "__main__":
    config = train_config.TRAIN_CONFIG
    seed = int(config.get("seed", 42))
    deterministic = bool(config.get("deterministic", True))
    seed_everything(seed, deterministic=deterministic)

    debug_masking = bool(config.get("debug_masking", False))
    debug_masking_batches = int(config.get("debug_masking_batches", 3))
    dataset_dir = config.get("dataset_dir", "data/pretraining_instances")
    tokenizer_name = config.get("tokenizer", "bert-base-uncased")
    batch_size = int(config.get("batch_size", 256))
    max_steps = int(config.get("max_steps", 1_000_000))
    checkpoint_every_steps = int(config.get("checkpoint_every_steps", 100_000))
    warmup_steps = int(config.get("warmup_steps", 10_000))
    lr = float(config.get("lr", 1e-4))
    adam_beta1 = float(config.get("adam_beta1", 0.9))
    adam_beta2 = float(config.get("adam_beta2", 0.999))
    weight_decay = float(config.get("weight_decay", 0.0))
    allow_tf32 = bool(config.get("allow_tf32", True))
    grad_clip_norm = config.get("grad_clip_norm", 1.0)
    grad_clip_norm = float(grad_clip_norm) if grad_clip_norm is not None else None
    num_workers = int(config.get("num_workers", 8))
    use_curriculum = bool(config.get("use_curriculum", False))
    seq_len_short = int(config.get("seq_len_short", 128))
    seq_len_long = int(config.get("seq_len_long", 512))
    short_seq_prob = float(config.get("short_seq_prob", 0.9))
    batch_size_long_cfg = config.get("batch_size_long", None)

    print("train process start")
    repo_root = Path(__file__).resolve().parent.parent.parent
    dataset_path = (repo_root / dataset_dir).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset dir을 찾을 수 없습니다: {dataset_path}")

    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id != 0:
        raise ValueError(
            f"현재 tokenizer.pad_token_id={tokenizer.pad_token_id} 입니다. "
            f"이 레포의 BERT 구현은 padding id=0을 가정합니다."
        )

    tokenized_datasets = load_from_disk(str(dataset_path))

    vocab_size = tokenizer.vocab_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and allow_tf32:
        # A100/RTX30+ 계열에서 TF32는 정확도 손실을 거의 늘리지 않으면서 속도 향상에 유리합니다.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    model = BERTLM(BERT(vocab_size=vocab_size)).to(device)
    # 모델이 실제로 기대한 BERT-Base 스펙인지 빠르게 검증
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model params: total={total_params/1e6:.2f}M, trainable={trainable_params/1e6:.2f}M",
        flush=True,
    )
    # BERTLM 내부에 bert(백본) 보유
    if hasattr(model, "bert"):
        print(
            f"BERT config: hidden={model.bert.hidden}, layers={model.bert.n_layers}, heads={model.bert.attn_heads}",
            flush=True,
        )
    bert_spec_dir = (
        repo_root
        / "runs"
        / f"L{model.bert.n_layers}_H{model.bert.hidden}_A{model.bert.attn_heads}_seed{seed}"
    )
    bert_spec_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv_path = bert_spec_dir / "metrics.csv"
    metrics_fieldnames = [
        "step",
        "epoch",
        "lr",
        "train_loss",
        "mlm_loss",
        "nsp_loss",
        "train_mlm_acc",
        "train_nsp_acc",
        "step_time",
        "gpu_mem_gb",
    ]
    with metrics_csv_path.open("w", newline="", encoding="utf-8") as metrics_f:
        metrics_writer = csv.DictWriter(metrics_f, fieldnames=metrics_fieldnames)
        metrics_writer.writeheader()

    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(adam_beta1, adam_beta2),
            weight_decay=weight_decay,
            fused=torch.cuda.is_available(),
        )
    except TypeError:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(adam_beta1, adam_beta2),
            weight_decay=weight_decay,
        )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )
    criterion_mlm = nn.CrossEntropyLoss(ignore_index=-100)
    criterion_nsp = nn.CrossEntropyLoss()

    # curriculum을 쓰려면 기본 데이터가 long 길이(>=512)로 만들어져 있어야 합니다.
    # (128로 생성된 instances로는 512를 만들 수 없음)
    if use_curriculum:
        example_len = len(tokenized_datasets[0]["input_ids"])
        if example_len < seq_len_long:
            raise ValueError(
                f"curriculum(use_curriculum=True)에는 dataset이 최소 seq_len_long={seq_len_long} 길이여야 합니다. "
                f"현재 instance 길이={example_len} 입니다. "
                f"create_pretraining_instance.py를 --max-seq-length {seq_len_long}로 다시 생성하세요."
            )

    # DataLoader 샘플링 재현성을 위한 generator / worker 시드
    g_train = torch.Generator()
    g_train.manual_seed(seed + 3)
    g_short = torch.Generator()
    g_short.manual_seed(seed + 1)
    g_long = torch.Generator()
    g_long.manual_seed(seed + 2)

    def make_seed_worker(offset: int):
        def _seed_worker(worker_id: int) -> None:
            worker_seed = seed + offset + worker_id
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)
            if np is not None:
                np.random.seed(worker_seed)

        return _seed_worker

    if use_curriculum:
        # short/long 데이터셋을 분리해 DataLoader를 따로 만들면
        # 128-step에서는 실제로 텐서가 128 길이라 속도/메모리 이득이 납니다.
        short_ds = BertPretrainDataset(
            tokenized_datasets,
            tokenizer,
            seq_len=seq_len_short,
            return_original_input_ids=debug_masking,
        )
        long_ds = BertPretrainDataset(
            tokenized_datasets,
            tokenizer,
            seq_len=seq_len_long,
            return_original_input_ids=debug_masking,
        )

        if batch_size_long_cfg is None:
            batch_size_long = max(1, batch_size // 4)
        else:
            batch_size_long = int(batch_size_long_cfg)

        short_loader = DataLoader(
            short_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            worker_init_fn=make_seed_worker(offset=1),
            generator=g_short,
            persistent_workers=(num_workers > 0),
            prefetch_factor=(2 if num_workers > 0 else None),
        )
        long_loader = DataLoader(
            long_ds,
            batch_size=batch_size_long,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            worker_init_fn=make_seed_worker(offset=2),
            generator=g_long,
            persistent_workers=(num_workers > 0),
            prefetch_factor=(2 if num_workers > 0 else None),
        )
        short_iter = iter(short_loader)
        long_iter = iter(long_loader)
    else:
        train_dataset = BertPretrainDataset(
            tokenized_datasets,
            tokenizer,
            return_original_input_ids=debug_masking,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            worker_init_fn=make_seed_worker(offset=3),
            generator=g_train,
            persistent_workers=(num_workers > 0),
            prefetch_factor=(2 if num_workers > 0 else None),
        )
        train_iter = iter(train_loader)

    scaler = GradScaler()

    def save_checkpoint(step: int) -> None:
        ckpt_path = bert_spec_dir / f"checkpoint_step_{step}.pth"
        ckpt = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": config,
        }
        torch.save(ckpt, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}", flush=True)

    model.train()
    global_step = 0
    if use_curriculum:
        steps_per_epoch = max(1, len(short_loader))
    else:
        steps_per_epoch = max(1, len(train_loader))

    pbar = tqdm(
        total=max_steps,
        desc="Training (steps)",
        dynamic_ncols=True,
        mininterval=0.5,
    )
    with metrics_csv_path.open("a", newline="", encoding="utf-8") as metrics_f:
        metrics_writer = csv.DictWriter(metrics_f, fieldnames=metrics_fieldnames)

        while global_step < max_steps:
            step_start = time.perf_counter()
            # 90% short(128), 10% long(512)
            if use_curriculum:
                # 첫 스텝은 워밍업/초기 지연 완화를 위해 항상 short를 사용
                use_short = True if global_step == 0 else (torch.rand(()) < short_seq_prob)
                if use_short:
                    try:
                        batch = next(short_iter)
                    except StopIteration:
                        short_iter = iter(short_loader)
                        batch = next(short_iter)
                else:
                    try:
                        batch = next(long_iter)
                    except StopIteration:
                        long_iter = iter(long_loader)
                        batch = next(long_iter)
            else:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

            optimizer.zero_grad(set_to_none=True)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            original_input_ids = None
            if debug_masking and global_step < debug_masking_batches:
                original_input_ids = batch["original_input_ids"].to(
                    device, non_blocking=True
                )
            token_type_ids = batch["token_type_ids"].to(device, non_blocking=True)
            special_tokens_mask = batch["special_tokens_mask"].to(
                device, non_blocking=True
            )
            input_ids, mlm_labels = mask_tokens(
                input_ids, tokenizer, special_tokens_mask=special_tokens_mask
            )
            nsp_labels = batch["nsp_labels"].to(device, non_blocking=True)

            if debug_masking and global_step < debug_masking_batches:
                # mlm_labels != -100  => masked_indices
                masked_indices = mlm_labels != -100
                masked_count = int(masked_indices.sum().item())

                mask_token_id = tokenizer.mask_token_id
                if mask_token_id is None:
                    raise ValueError("tokenizer.mask_token_id가 None입니다.")

                indices_replaced = masked_indices & (input_ids == mask_token_id)
                replaced_count = int(indices_replaced.sum().item())

                # indices_random은 'random으로 바뀐 경우'를 엄밀히 추적하려면 mask_tokens에서 분기 마스크를 같이 반환해야 합니다.
                # 여기서는 (replaced 아님) & (원본과 다름)으로 근사합니다. (random 토큰이 우연히 원본과 같으면 keep으로 잡힐 수 있습니다.)
                indices_random = (
                    masked_indices & ~indices_replaced & (input_ids != original_input_ids)
                )
                random_count = int(indices_random.sum().item())

                indices_keep = masked_indices & ~indices_replaced & ~indices_random
                keep_count = int(indices_keep.sum().item())

                # padding id가 masked로 포함되었는지 점검
                pad_token_id = tokenizer.pad_token_id
                pad_masked_count = int(
                    (masked_indices & (original_input_ids == pad_token_id)).sum().item()
                )

                if masked_count > 0:
                    replaced_ratio = replaced_count / masked_count
                    random_ratio = random_count / masked_count
                    keep_ratio = keep_count / masked_count
                else:
                    replaced_ratio = random_ratio = keep_ratio = 0.0

                print(
                    f"[debug-masking] global_step={global_step} "
                    f"masked={masked_count} "
                    f"replaced={replaced_count}({replaced_ratio:.4f}) "
                    f"random={random_count}({random_ratio:.4f}) "
                    f"keep={keep_count}({keep_ratio:.4f}) "
                    f"pad_masked={pad_masked_count}",
                    flush=True,
                )

            with autocast():
                pred_nsp, pred_mlm = model(input_ids, token_type_ids)

                mlm_loss = criterion_mlm(
                    pred_mlm.view(-1, vocab_size), mlm_labels.view(-1)
                )
                nsp_loss = criterion_nsp(pred_nsp, nsp_labels)

                loss = mlm_loss + nsp_loss

            scaler.scale(loss).backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            epoch = (global_step - 1) // steps_per_epoch + 1
            step_time = time.perf_counter() - step_start
            gpu_mem_gb = (
                float(torch.cuda.memory_allocated(device) / (1024**3))
                if torch.cuda.is_available()
                else 0.0
            )

            should_log_metrics = (global_step % 10000 == 0) or (global_step == max_steps)
            if should_log_metrics:
                with torch.no_grad():
                    # MLM 정확도는 masked 토큰 위치(mlm_labels != -100)에서만 계산
                    mlm_mask = mlm_labels != -100
                    mlm_mask_count = int(mlm_mask.sum().item())
                    if mlm_mask_count > 0:
                        mlm_pred_ids = pred_mlm.argmax(dim=-1)
                        mlm_correct = int(
                            (mlm_pred_ids[mlm_mask] == mlm_labels[mlm_mask]).sum().item()
                        )
                        train_mlm_acc = float(mlm_correct / mlm_mask_count)
                    else:
                        train_mlm_acc = 0.0

                    nsp_pred = pred_nsp.argmax(dim=-1)
                    nsp_correct = int((nsp_pred == nsp_labels).sum().item())
                    train_nsp_acc = float(nsp_correct / nsp_labels.numel())

                metrics_writer.writerow(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "train_loss": float(loss.item()),
                        "mlm_loss": float(mlm_loss.item()),
                        "nsp_loss": float(nsp_loss.item()),
                        "train_mlm_acc": train_mlm_acc,
                        "train_nsp_acc": train_nsp_acc,
                        "step_time": float(step_time),
                        "gpu_mem_gb": gpu_mem_gb,
                    }
                )
                metrics_f.flush()

            should_save_checkpoint = (
                checkpoint_every_steps > 0
                and (global_step % checkpoint_every_steps == 0)
            )
            if should_save_checkpoint:
                save_checkpoint(global_step)

            pbar.update(1)
            if global_step % 20 == 0:
                pbar.set_postfix({"L": f"{loss.item():.3f}"})

    pbar.close()
    print(f"Training finished at global_step={global_step}/{max_steps}")
    # 저장 경로는 사용자 인자 없이, 모델 스펙 + seed 기준으로만 고정합니다.
    if not hasattr(model, "bert"):
        raise AttributeError("현재 model 객체에 bert 속성이 없습니다. (저장 규칙 확인 필요)")

    bert_spec_dir = (
        repo_root
        / "runs"
        / f"L{model.bert.n_layers}_H{model.bert.hidden}_A{model.bert.attn_heads}_seed{seed}"
    )
    save_path = bert_spec_dir / "model_full.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, save_path)