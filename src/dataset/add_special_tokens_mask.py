"""
기존 HF datasets(`pretraining_instances`)에 special token 위치 마스크를 컬럼으로 추가합니다.

목표:
- train.py에서 on-the-fly로 tokenizer.get_special_tokens_mask()를 호출하지 않아도 되게 함
- MLM의 "special token은 마스킹하지 않는다" 규칙은 정확히 유지

출력 dataset에는 컬럼 `special_tokens_mask` (0/1)가 추가됩니다.
"""

from __future__ import annotations

import argparse
import shutil
from typing import List

from datasets import load_from_disk
from transformers import BertTokenizerFast


def _compute_special_tokens_mask(
    input_ids: List[int],
    tokenizer: BertTokenizerFast,
) -> List[int]:
    # already_has_special_tokens=True:
    # - input_ids에 이미 [CLS]/[SEP]/[PAD] 같은 special token이 포함되어 있다는 전제
    mask = tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)

    # 안전장치: padding id 위치도 special로 처리
    pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        mask = [int(m or (t == int(pad_id))) for m, t in zip(mask, input_ids)]
    else:
        mask = [int(m) for m in mask]

    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        type=str,
        default="data/pretraining_instances",
        help="기존 HF dataset 디렉토리 경로",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/pretraining_instances_spmask",
        help="special_tokens_mask 컬럼이 추가된 새 HF dataset 저장 경로",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bert-base-uncased",
        help="토크나이저 이름(캐시/특수 토큰 위치 계산용)",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=1,
        help="datasets.map 병렬 프로세스 수",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="out-dir이 있으면 덮어씀",
    )
    args = parser.parse_args()

    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer)

    if tokenizer.pad_token_id != 0:
        raise ValueError(
            f"현재 tokenizer.pad_token_id={tokenizer.pad_token_id} 입니다. "
            "이 레포 모델 마스크가 padding id=0을 가정합니다."
        )

    ds = load_from_disk(args.in_dir)
    if "input_ids" not in ds.column_names:
        raise KeyError("dataset에 'input_ids' 컬럼이 없습니다.")

    if "special_tokens_mask" in ds.column_names and not args.overwrite:
        raise FileExistsError(
            "out-dir로 덮어쓰지 않고 special_tokens_mask 컬럼이 이미 존재합니다. "
            "--overwrite 옵션을 사용하세요."
        )

    def add_col(batch):
        # batched=True 형태에서 batch는 dict[str, list[...]]로 옴
        input_ids_batch = batch["input_ids"]
        masks = [_compute_special_tokens_mask(x, tokenizer) for x in input_ids_batch]
        batch["special_tokens_mask"] = masks
        return batch

    # batched=True로 전체 배치를 받고 리스트로 변환합니다(토크나이저 호출만 반복).
    ds2 = ds.map(
        add_col,
        batched=True,
        num_proc=args.num_proc,
        desc="special_tokens_mask 컬럼 추가",
    )

    if args.overwrite and args.out_dir:
        shutil.rmtree(args.out_dir, ignore_errors=True)

    ds2.save_to_disk(args.out_dir)

    print(f"[OK] Saved: {args.out_dir}")


if __name__ == "__main__":
    main()

