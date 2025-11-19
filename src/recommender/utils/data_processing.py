import os
import re
import csv
import json
import pandas as pd
from loguru import logger
from pathlib import Path
import shutil
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import hf_hub_download


def extract_data_from_general_file(path: str) -> list:
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".json":
        return json.load(open(path, "r", encoding="utf-8"))
    if ext == ".jsonl":
        return [json.loads(l) for l in open(path, "r", encoding="utf-8")]
    if ext == ".csv":
        return list(csv.DictReader(open(path, "r", encoding="utf-8")))
    if ext == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    if ext == ".arrow":
        try:
            ds = load_dataset("arrow", data_files=path)
            key = next(iter(ds.keys()))
            return [dict(x) for x in ds[key]]
        except Exception as e:
            logger.error(f"Failed to load Arrow file: {e}")
            raise
    raise ValueError(f"Unsupported file format: {ext}")


def load_training_data(path: str) -> list:

    if os.path.isfile(path):
        data = extract_data_from_general_file(path)
        if not data:
            raise ValueError(f"Local file '{path}' contains no data.")
        return data

    if os.path.isdir(path):
        raise ValueError(
            f"Local folder '{path}' is not supported. "
            f"Pass a file or HF dataset ID (org/name)."
        )

    try:
        ds = load_dataset(path)
    except Exception as e:
        raise ValueError(
            f"Failed to load HF dataset '{path}': {e}\n"
            f"Ensure ID is correct/public or specify a config (e.g. repo/config)."
        )

    if isinstance(ds, Dataset):
        return [dict(x) for x in ds]
    if isinstance(ds, DatasetDict):
        split = "train" if "train" in ds else next(iter(ds.keys()), None)
        if not split:
            raise ValueError(f"Dataset '{path}' has no splits.")
        return [dict(x) for x in ds[split]]
    raise ValueError(f"Unrecognized dataset format for '{path}'.")


def load_model_file_from_hf(repo_id: str, file_name: str) -> dict:
    try:
        config_path = hf_hub_download(repo_id=repo_id, filename=file_name)
        return json.load(open(config_path, "r", encoding="utf-8"))
    except Exception:
        return {}


def escape_newlines_in_strings(s: str) -> str:
    pattern = r"""(['"])(.*?)(?<!\\)\1"""

    def repl(m):
        q, content = m.group(1), m.group(2)
        return f"{q}{content.replace('\n', '\\n')}{q}"

    return re.sub(pattern, repl, s, flags=re.DOTALL)


def get_model_path(model: str, unique_tag: str) -> str:
    model_path = Path(model)

    if model_path.is_dir():
        return str(model_path)

    base = Path(__file__).parent.parent
    cache_dir = base / "cached_files" / "models" / model / unique_tag
    cache_dir.mkdir(parents=True, exist_ok=True)

    for name in ("config.json", "tokenizer_config.json"):
        src = hf_hub_download(model, filename=name)
        shutil.copy(src, cache_dir / name)

    return str(cache_dir)

