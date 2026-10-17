"""Guard the environment setup in the sbatch scripts.

Delta's pytorch-conda/2.8 module sets PYTHONNOUSERSITE=1, which hides the
`pip install --user` packages (open_clip, conch, musk) the jobs import. Both
scripts must unset it after loading the module, or every job dies with
ModuleNotFoundError.
"""

import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


class TestSbatchEnv(unittest.TestCase):
    def _lines(self, name):
        with open(os.path.join(HERE, name)) as f:
            return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    def _check(self, name):
        lines = self._lines(name)
        load = lines.index("module load pytorch-conda/2.8")
        self.assertIn("unset PYTHONNOUSERSITE", lines[load + 1 :], f"{name}: unset must follow module load")
        unset = lines.index("unset PYTHONNOUSERSITE")
        launches = ("python", "uvicorn", "PORT=$(python")
        first_python = next(i for i, line in enumerate(lines) if line.startswith(launches))
        self.assertLess(unset, first_python, f"{name}: unset must come before Python runs")

    def test_submit(self):
        self._check("submit.sbatch")

    def test_submit_serve(self):
        self._check("submit_serve.sbatch")

    def test_serve_port_honours_serve_port(self):
        self.assertIn('PORT=$(find_free_port "${SERVE_PORT:-8000}")', self._lines("submit_serve.sbatch"))


if __name__ == "__main__":
    unittest.main()
