"""Phase 7 data prep: WikiText-103 (raw) -> GPT-2 BPE token bins.

Downloads the parquet shards of Salesforce/wikitext (wikitext-103-raw-v1)
via huggingface_hub, tokenizes with tiktoken's GPT-2 encoding, and writes
uint16 bins + meta.json under data/lm/wt103/. One-time, CPU-only (~2 GB RAM,
a few minutes). Run inside the container:

    python experiments/lm/prepare_data.py
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import tiktoken
from huggingface_hub import HfApi, hf_hub_download

REPO = "Salesforce/wikitext"
CONFIG = "wikitext-103-raw-v1"
EOT = 50256  # GPT-2 <|endoftext|>


def shard_files(split: str) -> list[str]:
    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    out = sorted(
        f for f in files
        if f.startswith(f"{CONFIG}/") and f"{split}-" in f and f.endswith(".parquet")
    )
    if not out:
        raise RuntimeError(f"no parquet shards found for split {split!r} in {REPO}")
    return out


def tokenize_split(split: str, enc) -> np.ndarray:
    ids: list[int] = []
    for fname in shard_files(split):
        path = hf_hub_download(REPO, fname, repo_type="dataset")
        rows = pd.read_parquet(path)["text"].tolist()
        print(f"  {fname}: {len(rows)} rows")
        # Batch-encode in chunks; wikitext rows are lines, so no per-row EOT.
        for i in range(0, len(rows), 20000):
            for toks in enc.encode_ordinary_batch(rows[i:i + 20000]):
                ids.extend(toks)
    ids.append(EOT)
    arr = np.array(ids, dtype=np.uint16)
    assert int(arr.max()) < 2 ** 16
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/lm/wt103")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    meta = {"dataset": f"{REPO}:{CONFIG}", "tokenizer": "gpt2", "vocab_size": enc.n_vocab}
    for split, binname in [("validation", "val.bin"), ("train", "train.bin")]:
        path = os.path.join(args.out, binname)
        if os.path.exists(path):
            n = os.path.getsize(path) // 2
            print(f"{binname} exists ({n:,} tokens), skipping")
            meta[f"{binname[:-4]}_tokens"] = n
            continue
        print(f"tokenizing {split} ...")
        arr = tokenize_split(split, enc)
        arr.tofile(path)
        meta[f"{binname[:-4]}_tokens"] = len(arr)
        print(f"  -> {path}: {len(arr):,} tokens")

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("done:", json.dumps(meta))


if __name__ == "__main__":
    main()
