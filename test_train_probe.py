"""Tests for train_probe.py on small synthetic embeddings CSVs."""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_probe  # noqa: E402


def _write_embeddings(path, n_per_class=20, classes=("TUM", "STR"), dim=8, seed=0):
    """Two well-separated clusters, one folder per class, like NCT-CRC-HE-100K."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, cls in enumerate(classes):
        center = np.zeros(dim)
        center[i] = 5.0
        for j in range(n_per_class):
            emb = center + rng.normal(scale=0.1, size=dim)
            rows.append({"image_path": f"/data/{cls}/{cls}-{j}.tif", "embedding": json.dumps(emb.tolist())})
    pd.DataFrame(rows).to_csv(path, index=False)
    return rows


class TestTrainProbe(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.emb_csv = os.path.join(self.tmp.name, "embeddings.csv")

    def _run(self, *argv):
        out = io.StringIO()
        with redirect_stdout(out):
            code = train_probe.main(list(argv))
        return code, out.getvalue()

    def test_labels_default_to_parent_folder(self):
        _write_embeddings(self.emb_csv)
        df = train_probe.load_dataset(self.emb_csv)
        self.assertEqual(set(df["label"]), {"TUM", "STR"})
        self.assertEqual(df["embedding"].iloc[0].shape, (8,))

    def test_separable_data_reaches_perfect_accuracy(self):
        _write_embeddings(self.emb_csv)
        code, out = self._run(self.emb_csv)
        self.assertEqual(code, 0)
        self.assertIn("accuracy: 1.000", out)
        self.assertIn("split by tile", out)

    def test_labels_csv_with_group_splits_by_group(self):
        rows = _write_embeddings(self.emb_csv)
        labels_csv = os.path.join(self.tmp.name, "labels.csv")
        pd.DataFrame(
            {
                "image_path": [r["image_path"] for r in rows],
                "label": [r["image_path"].split("/")[2] for r in rows],
                "group": [f"patient{i % 10}" for i in range(len(rows))],
            }
        ).to_csv(labels_csv, index=False)

        df = train_probe.load_dataset(self.emb_csv, labels_csv)
        train, test = train_probe.split(df, test_size=0.3, seed=0)
        self.assertFalse(set(train["group"]) & set(test["group"]))

        code, out = self._run(self.emb_csv, "--labels", labels_csv, "--test-size", "0.3")
        self.assertEqual(code, 0)
        self.assertIn("split by group", out)

    def test_classify_mode_csv_gives_clear_error(self):
        pd.DataFrame({"image_path": ["/a/TUM/x.tif"], "predicted_label": ["TUM"], "score": [0.9]}).to_csv(
            self.emb_csv, index=False
        )
        with self.assertRaises(SystemExit) as ctx:
            train_probe.load_dataset(self.emb_csv)
        self.assertIn("--text-prompts", str(ctx.exception))

    def test_labels_csv_missing_column(self):
        _write_embeddings(self.emb_csv)
        labels_csv = os.path.join(self.tmp.name, "labels.csv")
        pd.DataFrame({"image_path": ["/data/TUM/TUM-0.tif"]}).to_csv(labels_csv, index=False)
        with self.assertRaises(SystemExit) as ctx:
            train_probe.load_dataset(self.emb_csv, labels_csv)
        self.assertIn("label", str(ctx.exception))

    def test_single_label_rejected(self):
        _write_embeddings(self.emb_csv, classes=("TUM",))
        with self.assertRaises(SystemExit):
            self._run(self.emb_csv)


if __name__ == "__main__":
    unittest.main()
