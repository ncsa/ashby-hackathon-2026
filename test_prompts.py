"""The shipped prompt files must load the way run_embeddings.py reads them."""

import json
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
NCT_CLASSES = {"ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"}


class TestPromptFiles(unittest.TestCase):
    def _load(self, name):
        with open(os.path.join(HERE, name)) as f:
            prompts = json.load(f)
        self.assertIsInstance(prompts, dict)
        for label, text in prompts.items():
            self.assertTrue(label and isinstance(text, str) and text.strip(), label)
        return prompts

    def test_prompts_json(self):
        self._load("prompts.json")

    def test_nct_labels_match_dataset_folders(self):
        # Labels must equal the NCT-CRC-HE-100K folder names so predictions can be
        # scored against each tile's folder.
        self.assertEqual(set(self._load("prompts_nct.json")), NCT_CLASSES)


if __name__ == "__main__":
    unittest.main()
