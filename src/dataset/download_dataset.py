from pathlib import Path

from datasets import load_dataset, concatenate_datasets

bookcorpus = load_dataset(
    "lucadiliello/bookcorpusopen",
    split="train"
)

wiki = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train"
)

wiki = wiki.remove_columns(
    [c for c in wiki.column_names if c != "text"]
)

dataset = concatenate_datasets([bookcorpus, wiki])

# create_pretraining_data.py에서 불러올 수 있도록 저장
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
dataset.save_to_disk(str(DATA_DIR / "raw_dataset"))
print(dataset)
print(f"저장 경로: {DATA_DIR / 'raw_dataset'}")