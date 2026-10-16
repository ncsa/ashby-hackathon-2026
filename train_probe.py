#!/usr/bin/env python3
"""Train a linear probe (logistic regression) on frozen embeddings.

Reads the CSV that run_embeddings.py / submit.sbatch write (columns
`image_path`, `embedding`) and reports held-out accuracy. CPU only; runs in
seconds for tens of thousands of tiles.

Labels come from each tile's parent folder name by default (the layout of
NCT-CRC-HE-100K: one folder per class). Pass --labels with a CSV of
`image_path,label` to use your own, and add a `group` column (patient or
slide ID) to split by group so tiles from one patient never land in both
train and test.

    python train_probe.py embeddings.csv
    python train_probe.py embeddings.csv --labels labels.csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def load_dataset(embeddings_csv: str, labels_csv: Optional[str] = None) -> pd.DataFrame:
    """Return a frame with `embedding` (np.ndarray), `label` and optional `group`."""
    df = pd.read_csv(embeddings_csv)
    if "embedding" not in df.columns:
        raise SystemExit(
            f"{embeddings_csv} has no 'embedding' column. Was it written with --text-prompts? "
            "That mode writes predicted labels, not embeddings; rerun without it."
        )
    df["embedding"] = df["embedding"].map(lambda s: np.asarray(json.loads(s), dtype=np.float32))

    if labels_csv:
        labels = pd.read_csv(labels_csv)
        missing = {"image_path", "label"} - set(labels.columns)
        if missing:
            raise SystemExit(f"{labels_csv} is missing column(s): {', '.join(sorted(missing))}")
        df = df.merge(labels, on="image_path", how="inner")
        if df.empty:
            raise SystemExit(f"No image_path values in {labels_csv} match {embeddings_csv}.")
    else:
        df["label"] = df["image_path"].map(lambda p: Path(p).parent.name)
    return df


def split(df: pd.DataFrame, test_size: float, seed: int):
    if "group" in df.columns:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(df, groups=df["group"]))
        return df.iloc[train_idx], df.iloc[test_idx]
    return train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("embeddings_csv")
    parser.add_argument("--labels", help="CSV with image_path,label[,group]; default: parent folder name")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    df = load_dataset(args.embeddings_csv, args.labels)
    if df["label"].nunique() < 2:
        raise SystemExit("Need at least two distinct labels to train a classifier.")
    train, test = split(df, args.test_size, args.seed)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(np.stack(train["embedding"]), train["label"])
    pred = clf.predict(np.stack(test["embedding"]))

    by = "group" if "group" in df.columns else "tile (no group column: accuracy may be optimistic)"
    print(f"{len(train)} train / {len(test)} test tiles, split by {by}")
    print(f"accuracy: {np.mean(pred == test['label'].to_numpy()):.3f}")
    print(classification_report(test["label"], pred, zero_division=0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
