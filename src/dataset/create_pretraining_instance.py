"""
create_pretraining_documents.py가 만든 pretraining_documents.pkl(List[List[str]])을 입력으로 받아
BERT pretraining용 인스턴스(문장쌍/NSP/청킹/토크나이즈/패딩)를 생성해 저장합니다.

기본 산출물 (HF datasets 컬럼):
  - input_ids: List[int] (len=max_seq_length)
  - token_type_ids: List[int] (len=max_seq_length)
  - attention_mask: List[int] (len=max_seq_length)
  - special_tokens_mask: List[int] (len=max_seq_length, 0/1)
  - nsp_label: int (0/1)

MLM은 현재 train.py에서 on-the-fly로 마스킹하므로 여기서는 고정 MLM을 생성하지 않습니다.
special_tokens_mask도 이 스크립트에서 함께 생성합니다.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from datasets import Dataset, Features, Sequence, Value
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm
from transformers import BertTokenizerFast


@dataclass(frozen=True)
class PretrainInstance:
    input_ids: List[int]
    token_type_ids: List[int]
    attention_mask: List[int]
    nsp_label: int


def _set_seed(seed: int) -> None:
    random.seed(seed)


def _compute_special_tokens_mask(
    input_ids: List[int],
    tokenizer: BertTokenizerFast,
) -> List[int]:
    # already_has_special_tokens=True:
    # - input_ids에 [CLS]/[SEP]/[PAD] 등 special token이 이미 포함되어 있다는 전제
    mask = tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)

    # 안전장치: padding id 위치도 special로 처리
    pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        return [int(m or (t == int(pad_id))) for m, t in zip(mask, input_ids)]
    return [int(m) for m in mask]


def _flatten_docs_lengths(docs_token_ids: Sequence[Sequence[Sequence[int]]]) -> int:
    return sum(len(s) for d in docs_token_ids for s in d)


def _truncate_seq_pair(tokens_a: List[int], tokens_b: List[int], max_num_tokens: int) -> None:
    """
    max_num_tokens 이내가 되도록 (A+B) 길이를 줄입니다.
    BERT 원 구현처럼 긴 쪽을 랜덤하게 앞/뒤에서 하나씩 제거합니다.
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            return
        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        if not trunc_tokens:
            return
        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def _build_bert_input(
    tokens_a: List[int],
    tokens_b: List[int],
    *,
    tokenizer: BertTokenizerFast,
    max_seq_length: int,
    nsp_label: int,
) -> PretrainInstance:
    # [CLS] A [SEP] B [SEP]
    input_ids = [tokenizer.cls_token_id] + tokens_a + [tokenizer.sep_token_id]
    token_type_ids = [0] * len(input_ids)

    input_ids += tokens_b + [tokenizer.sep_token_id]
    token_type_ids += [1] * (len(tokens_b) + 1)

    attention_mask = [1] * len(input_ids)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer.pad_token_id가 None입니다. pad token이 설정된 tokenizer를 사용하세요.")

    if len(input_ids) > max_seq_length:
        raise ValueError(
            f"내부 오류: sequence가 max_seq_length를 초과했습니다. "
            f"len={len(input_ids)} max_seq_length={max_seq_length}"
        )

    pad_len = max_seq_length - len(input_ids)
    if pad_len:
        input_ids = input_ids + [pad_id] * pad_len
        token_type_ids = token_type_ids + [0] * pad_len
        attention_mask = attention_mask + [0] * pad_len

    return PretrainInstance(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        nsp_label=int(nsp_label),
    )


def _create_instances_from_document(
    docs_token_ids: Sequence[Sequence[Sequence[int]]],
    doc_index: int,
    *,
    tokenizer: BertTokenizerFast,
    max_seq_length: int,
    short_seq_prob: float,
    nsp_random_prob: float,
) -> List[PretrainInstance]:
    """
    하나의 document(=문장 토큰 id 리스트들의 리스트)로부터 여러 pretraining instance를 생성합니다.
    - 청킹: 문장들을 current_chunk로 쌓다가 target length에 도달하면 instance 생성
    - NSP: 확률적으로 random next를 선택
    """
    max_num_tokens = max_seq_length - 3  # [CLS], [SEP], [SEP]
    document = docs_token_ids[doc_index]

    # 문서가 너무 짧으면 스킵
    if len(document) < 2:
        return []

    # 목표 길이 설정 (가끔 짧게)
    target_seq_length = max_num_tokens
    if random.random() < short_seq_prob:
        target_seq_length = random.randint(2, max_num_tokens)

    instances: List[PretrainInstance] = []
    current_chunk: List[List[int]] = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = list(document[i])
        current_chunk.append(segment)
        current_length += len(segment)

        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # A는 1..k-1 문장, B는 나머지
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = random.randint(1, len(current_chunk) - 1)

                tokens_a: List[int] = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b: List[int] = []
                # BERT/HF convention:
                #   0 -> IsNext
                #   1 -> NotNext
                nsp_label = 0  # is_next

                # random next
                if len(current_chunk) == 1 or random.random() < nsp_random_prob:
                    nsp_label = 1  # not_next
                    target_b_length = target_seq_length - len(tokens_a)

                    # 다른 문서에서 랜덤 시작점 선택
                    rand_doc_index = doc_index
                    for _ in range(10):
                        rand_doc_index = random.randint(0, len(docs_token_ids) - 1)
                        if rand_doc_index != doc_index and len(docs_token_ids[rand_doc_index]) >= 2:
                            break

                    rand_doc = docs_token_ids[rand_doc_index]
                    rand_start = random.randint(0, len(rand_doc) - 1)
                    for j in range(rand_start, len(rand_doc)):
                        tokens_b.extend(rand_doc[j])
                        if len(tokens_b) >= target_b_length:
                            break

                    # i를 되돌려서 남은 current_chunk를 다음 instance에 사용
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                else:
                    # 실제 다음 문장들 사용
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                if not tokens_a or not tokens_b:
                    current_chunk = []
                    current_length = 0
                    i += 1
                    continue

                _truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                instance = _build_bert_input(
                    tokens_a,
                    tokens_b,
                    tokenizer=tokenizer,
                    max_seq_length=max_seq_length,
                    nsp_label=nsp_label,
                )
                instances.append(instance)

            current_chunk = []
            current_length = 0

        i += 1

    return instances


def iter_pretraining_instances(
    documents: Sequence[Sequence[str]],
    *,
    tokenizer: BertTokenizerFast,
    max_seq_length: int = 128,
    dupe_factor: int = 5,
    short_seq_prob: float = 0.1,
    nsp_random_prob: float = 0.5,
    seed: int = 1234,
    max_docs: int | None = None,
    max_instances: int | None = None,
    show_progress: bool = False,
) -> Iterable[PretrainInstance]:
    """
    documents(List[List[str]]) -> List[PretrainInstance]
    - WordPiece 토크나이즈(문장 단위, special token 없이)
    - dupe_factor만큼 서로 다른 샘플링으로 인스턴스 증식
    """
    if max_seq_length < 8:
        raise ValueError("max_seq_length는 최소 8 이상을 권장합니다.")
    if dupe_factor < 1:
        dupe_factor = 1
    if not (0.0 <= short_seq_prob <= 1.0):
        raise ValueError("short_seq_prob는 0~1 범위여야 합니다.")
    if not (0.0 <= nsp_random_prob <= 1.0):
        raise ValueError("nsp_random_prob는 0~1 범위여야 합니다.")

    if max_docs is not None and max_docs > 0:
        documents = documents[:max_docs]

    # 문장 -> token ids (special token 제외)
    #
    # 주의: 일부 문장(특히 위키/책 말뭉치의 이상한 줄바꿈/문장분리 실패)은
    # WordPiece 기준으로 tokenizer.model_max_length(보통 512)를 초과할 수 있습니다.
    # 이런 케이스는 경고를 유발하고 처리도 비효율적이므로 문장 단위에서 상한으로 잘라 둡니다.
    docs_token_ids: List[List[List[int]]] = []
    sent_max_len = tokenizer.model_max_length
    if not isinstance(sent_max_len, int) or sent_max_len <= 0 or sent_max_len > 1_000_000:
        sent_max_len = 512
    doc_iter = documents
    if show_progress:
        try:
            n_docs = len(documents)  # type: ignore[arg-type]
        except TypeError:
            n_docs = None
        doc_iter = tqdm(
            documents,
            total=n_docs,
            desc="문서 토크나이즈",
            unit="doc",
            mininterval=0.5,
        )
    for doc in doc_iter:
        sents_token_ids: List[List[int]] = []
        for sent in doc:
            ids = tokenizer(
                sent,
                add_special_tokens=False,
                return_attention_mask=False,
                truncation=True,
                max_length=sent_max_len,
            )["input_ids"]
            if ids:
                sents_token_ids.append(ids)
        if len(sents_token_ids) >= 2:
            docs_token_ids.append(sents_token_ids)

    if not docs_token_ids:
        raise ValueError("유효한 문서가 없습니다. documents 전처리/토크나이즈 결과를 확인하세요.")

    if show_progress:
        print(
            f"  → 유효 문서 {len(docs_token_ids):,}개. "
            f"인스턴스 생성·기록 중 (dupe_factor={dupe_factor}회 패스)…",
            flush=True,
        )

    yielded = 0
    base_seed = seed
    for dup in range(dupe_factor):
        _set_seed(base_seed + dup)

        # 문서 순서를 섞어서 NSP random next 다양화
        doc_indices = list(range(len(docs_token_ids)))
        random.shuffle(doc_indices)

        for doc_index in doc_indices:
            doc_instances = _create_instances_from_document(
                docs_token_ids,
                doc_index,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                short_seq_prob=short_seq_prob,
                nsp_random_prob=nsp_random_prob,
            )
            for ins in doc_instances:
                yield ins
                yielded += 1
                if max_instances is not None and max_instances > 0 and yielded >= max_instances:
                    return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bert-base-uncased",
        help="HuggingFace tokenizer name/path (예: bert-base-uncased)",
    )
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--dupe-factor", type=int, default=5)
    parser.add_argument("--short-seq-prob", type=float, default=0.1)
    parser.add_argument("--nsp-random-prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--documents-pkl",
        type=str,
        default="data/pretraining_documents.pkl",
        help="create_pretraining_documents.py 산출물 경로",
    )
    parser.add_argument(
        "--out-dataset-dir",
        type=str,
        default="data/pretraining_instances_spmask",
        help="생성된 instance dataset 저장 디렉토리",
    )
    parser.add_argument(
        "--out-meta-json",
        type=str,
        default="data/pretraining_instances_spmask_meta.json",
        help="생성 통계 JSON 저장 경로",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="진행바·중간 로그 끔",
    )
    args = parser.parse_args()
    verbose = not args.quiet

    root = Path(__file__).resolve().parent.parent.parent
    docs_path = (root / args.documents_pkl).resolve()
    out_dir = (root / args.out_dataset_dir).resolve()
    meta_path = (root / args.out_meta_json).resolve()

    if not docs_path.exists():
        raise FileNotFoundError(f"documents pkl을 찾을 수 없습니다: {docs_path}")

    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer)

    # bert.py는 (x != 0)을 padding 마스크로 쓰므로 pad_token_id=0이 가장 안전
    if tokenizer.pad_token_id != 0:
        # tokenizer에 따라 pad id가 다를 수 있어도 동작은 하지만, 이 레포 모델 마스크와 불일치 가능성이 큼
        raise ValueError(
            f"현재 tokenizer.pad_token_id={tokenizer.pad_token_id} 입니다. "
            f"이 레포의 BERT 구현은 padding id=0을 가정합니다."
        )

    with open(docs_path, "rb") as f:
        documents = pickle.load(f)

    # ArrowWriter를 사용해 스트리밍으로 작성
    out_dir.mkdir(parents=True, exist_ok=True)

    features = Features(
        {
            "input_ids": Sequence(Value("int32")),
            "token_type_ids": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int8")),
            "special_tokens_mask": Sequence(Value("int8")),
            "nsp_label": Value("int64"),
        }
    )

    arrow_path = out_dir / "data.arrow"
    writer = ArrowWriter(features=features, path=str(arrow_path))

    n = 0
    nsp_is_next = 0
    total_nonpad = 0

    for ins in iter_pretraining_instances(
        documents,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        dupe_factor=args.dupe_factor,
        short_seq_prob=args.short_seq_prob,
        nsp_random_prob=args.nsp_random_prob,
        seed=args.seed,
        # 전체 데이터셋 사용 (부분 샘플링 비활성화)
        max_docs=None,
        max_instances=None,
    ):
        example = {
            "input_ids": ins.input_ids,
            "token_type_ids": ins.token_type_ids,
            "attention_mask": ins.attention_mask,
            "special_tokens_mask": _compute_special_tokens_mask(ins.input_ids, tokenizer),
            "nsp_label": int(ins.nsp_label),
        }
        writer.write(example)

        n += 1
        if int(ins.nsp_label) == 0:
            nsp_is_next += 1
        total_nonpad += sum(1 for t in ins.attention_mask if t)

    writer.finalize()

    ds = Dataset.from_file(str(arrow_path))
    ds.save_to_disk(str(out_dir))

    nsp_not_next = n - nsp_is_next
    avg_nonpad = float(total_nonpad / max(1, n))

    meta = {
        "documents_pkl": str(docs_path),
        "tokenizer": args.tokenizer,
        "max_seq_length": args.max_seq_length,
        "dupe_factor": args.dupe_factor,
        "short_seq_prob": args.short_seq_prob,
        "nsp_random_prob": args.nsp_random_prob,
        "seed": args.seed,
        "num_instances": n,
        "nsp_is_next": nsp_is_next,
        "nsp_not_next": nsp_not_next,
        "avg_nonpad_tokens": avg_nonpad,
        "special_tokens_mask_included": True,
        "out_dataset_dir": str(out_dir),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"instances: {n:,}")
    print(f"NSP is_next(0): {nsp_is_next:,}, not_next(1): {nsp_not_next:,}")
    print(f"avg non-pad tokens: {avg_nonpad:.2f}")
    print(f"saved dataset: {out_dir}")
    print(f"saved meta: {meta_path}")


if __name__ == "__main__":
    main()