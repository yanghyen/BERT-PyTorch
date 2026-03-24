"""
download_dataset.py에서 저장한 raw dataset을 불러와
텍스트를 정제한 뒤 문장 단위로 나누어 pretraining용 문장 리스트를 만듭니다.
"""

import argparse
import os
import pickle
import re
from pathlib import Path
from typing import List, Optional

from datasets import load_from_disk
from nltk.tokenize import sent_tokenize


def load_raw_dataset(data_dir: Path):
    """download_dataset.py에서 저장한 dataset을 불러옵니다."""
    path = data_dir / "raw_dataset"
    if not path.exists():
        raise FileNotFoundError(
            f"'{path}'를 찾을 수 없습니다. 먼저 download_dataset.py를 실행해 주세요."
        )
    return load_from_disk(str(path))


_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")
_RE_NON_WORD_DENSITY = re.compile(r"[A-Za-z]")


def _clean_line(line: str) -> Optional[str]:
    line = line.strip()
    if not line:
        return None

    # URL 제거
    line = _RE_URL.sub("", line).strip()
    if not line:
        return None

    # 마크다운 헤딩/구분선 같은 잡음 제거
    if _RE_MD_HEADING.match(line):
        line = _RE_MD_HEADING.sub("", line).strip()
        if not line:
            return None
    if set(line) <= {"-", "_", "*", "="}:
        return None

    # 너무 짧은 라인 제거 (예: "1 + 2", "02212018" 같은 잡음)
    if len(line) < 20:
        # 알파벳이 거의 없으면(숫자/기호 위주) 제거
        if len(_RE_NON_WORD_DENSITY.findall(line)) < 5:
            return None

    # 저작권/에디션/재현 금지 등 흔한 서지/권리 문구 제거 (너무 공격적이지 않게 핵심 키워드만)
    lower = line.lower()
    blacklist = (
        "copyright",
        "all rights reserved",
        "smashwords",
        "ebook edition",
        "no part of this",
        "may be reproduced",
        "information storage and retrieval",
        "cover art",
        "isbn",
    )
    if any(k in lower for k in blacklist):
        return None

    # 공백 정리
    line = re.sub(r"\s+", " ", line).strip()
    return line or None


def normalize_text(text: str) -> str:
    """
    BookCorpus/Wikipedia에 섞인 줄바꿈/헤더/URL/저작권 등을 완화한 텍스트로 정규화.
    문장 토크나이즈 전에 먼저 호출.
    """
    if not text or not text.strip():
        return ""

    # 줄 단위로 클린업 후 다시 합치기
    lines = text.splitlines()
    cleaned: List[str] = []
    for ln in lines:
        cl = _clean_line(ln)
        if cl:
            cleaned.append(cl)

    # 너무 많이 날아가면 원문 기반으로 최소 정규화만 적용
    if not cleaned:
        text = _RE_URL.sub("", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    return " ".join(cleaned).strip()


def text_to_sentences(text: str) -> List[str]:
    """텍스트를 문장 단위로 나눕니다. 빈 문장/너무 짧은 문장은 제외합니다."""
    text = normalize_text(text)
    if not text:
        return []

    sents = sent_tokenize(text)
    out: List[str] = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        # 극단적으로 짧은 문장 제거
        if len(s) < 15:
            continue
        out.append(s)
    return out


def create_pretraining_documents(
    dataset,
    min_sentences_per_doc: int = 2,
    num_proc: int = 1,
    batch_size: int = 512,
    max_chars_per_doc: int = 200_000,
    skip_long_docs: bool = True,
) -> List[List[str]]:
    """
    dataset의 각 row(=document)의 'text'를 문장 리스트로 변환해
    document boundary를 유지한 구조로 반환합니다.

    반환 형태:
      [
        ["doc1_sentence1", "doc1_sentence2", ...],
        ["doc2_sentence1", "doc2_sentence2", ...],
        ...
      ]
    """
    # datasets.map을 사용하면 multiprocess 병렬 처리가 가능함
    # (참고: sent_tokenize는 파이썬 레벨 호출이라 단건 루프보다 map+num_proc가 훨씬 빠름)
    if num_proc < 1:
        num_proc = 1
    if batch_size < 1:
        batch_size = 1

    def _batch_to_sentences(batch):
        texts = batch.get("text", [])
        out = []
        for t in texts:
            t = t or ""
            if max_chars_per_doc and len(t) > max_chars_per_doc:
                if skip_long_docs:
                    out.append([])
                    continue
                t = t[:max_chars_per_doc]
            try:
                out.append(text_to_sentences(t))
            except Exception:
                out.append([])
        return {"sentences": out}

    processed = dataset.map(
        _batch_to_sentences,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc=f"문서별 문장 분리 (num_proc={num_proc}, batch_size={batch_size})",
    )

    processed = processed.filter(
        lambda ex: len(ex["sentences"]) >= min_sentences_per_doc,
        desc=f"짧은 문서 제거 (min_sentences_per_doc={min_sentences_per_doc})",
        num_proc=num_proc,
    )

    # pickle로 저장할 리스트 형태로 추출
    return processed["sentences"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-sentences-per-doc", type=int, default=2)
    parser.add_argument(
        "--num-proc",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="문장 분리 병렬 프로세스 수 (기본: CPU-1)",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--max-chars-per-doc",
        type=int,
        default=200_000,
        help="문서 길이 상한(문자). 초과 시 기본은 제외(스킵). 0이면 제한 없음.",
    )
    parser.add_argument(
        "--no-skip-long-docs",
        action="store_true",
        help="긴 문서를 제외하지 않고, --max-chars-per-doc까지만 잘라서 처리",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_raw_dataset(data_dir)
    pretraining_documents = create_pretraining_documents(
        dataset,
        min_sentences_per_doc=args.min_sentences_per_doc,
        num_proc=args.num_proc,
        batch_size=args.batch_size,
        max_chars_per_doc=args.max_chars_per_doc,
        skip_long_docs=not args.no_skip_long_docs,
    )

    out_path = data_dir / "pretraining_documents.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(pretraining_documents, f)

    num_docs = len(pretraining_documents)
    num_sents = sum(len(d) for d in pretraining_documents)
    print(f"총 문서 수: {num_docs:,}")
    print(f"총 문장 수: {num_sents:,}")
    print(f"저장 경로: {out_path}")


if __name__ == "__main__":
    main()