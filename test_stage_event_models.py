"""Tests for stage_event_models.sh, with a fake sbatch on PATH."""

import json
import os
import stat
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "stage_event_models.sh")


class TestStageEventModels(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.bin = os.path.join(self.tmp.name, "bin")
        self.serve = os.path.join(self.tmp.name, "serve")
        self.calls = os.path.join(self.tmp.name, "sbatch_calls")
        os.makedirs(self.bin)
        fake = os.path.join(self.bin, "sbatch")
        with open(fake, "w") as f:
            f.write(
                f'#!/bin/bash\necho "$*" >> {self.calls}\n'
                f'echo "HF_HOME=$HF_HOME HF_HUB_OFFLINE=$HF_HUB_OFFLINE" >> {self.calls}.env\n'
                'echo "Submitted batch job 1"\n'
            )
        os.chmod(fake, os.stat(fake).st_mode | stat.S_IEXEC)

    def _run(self, *args):
        env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}", SERVE_DIR=self.serve)
        return subprocess.run(["bash", SCRIPT, *args], env=env, capture_output=True, text=True)

    def _sbatch_calls(self):
        with open(self.calls) as f:
            return f.read().splitlines()

    def test_submit_sends_one_job_per_model_to_the_reservation(self):
        result = self._run("submit")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self._sbatch_calls()
        self.assertEqual(len(calls), 4)
        for call in calls:
            self.assertIn("--account=bfqi-delta-gpu", call)
            self.assertIn("--partition=gpuA40x4", call)
            self.assertIn("--reservation=sup-30434", call)
            self.assertIn("--time=34:25:00", call)
            self.assertIn("submit_serve.sbatch", call)
        self.assertTrue(calls[2].endswith("--job-name=quiltnet --export=ALL,SERVE_PORT=8002 submit_serve.sbatch openclip:wisdomik/QuiltNet-B-32"))

    def test_each_server_gets_its_own_starting_port(self):
        # Regression: four jobs starting together on one node all chose 8000.
        self._run("submit")
        ports = [c.split("SERVE_PORT=")[1].split()[0] for c in self._sbatch_calls()]
        self.assertEqual(ports, ["8000", "8001", "8002", "8003"])
        for call in self._sbatch_calls():
            self.assertIn("--export=ALL,SERVE_PORT=", call)

    def test_submit_disables_resubmit_chain_for_each_job(self):
        self._run("submit")
        for name in ("conch", "musk", "quiltnet", "plip"):
            self.assertTrue(os.path.exists(os.path.join(self.serve, f"STOP_CHAIN-{name}")), name)

    def test_jobs_get_tested_cache_offline_regardless_of_caller_hf_home(self):
        env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}", SERVE_DIR=self.serve,
                   HF_HOME="/work/nvme/elsewhere", HOME=self.tmp.name)
        subprocess.run(["bash", SCRIPT, "submit"], env=env, check=True, capture_output=True)
        with open(self.calls + ".env") as f:
            lines = set(f.read().splitlines())
        self.assertEqual(lines, {f"HF_HOME={self.tmp.name}/.cache/huggingface HF_HUB_OFFLINE=1"})

    def test_dry_run_submits_nothing(self):
        result = self._run("--dry-run", "submit")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(os.path.exists(self.calls))
        self.assertEqual(result.stdout.count("DRY RUN: sbatch"), 4)

    def test_status_prints_up_and_pending_servers(self):
        os.makedirs(os.path.join(self.serve, "current-conch"))
        with open(os.path.join(self.serve, "current-conch", "connection.json"), "w") as f:
            json.dump({"node": "gpub080", "port": 8000, "api_key": "pathology-abc"}, f)
        result = self._run("status")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("- conch: http://gpub080:8000  key: pathology-abc", result.stdout)
        self.assertIn("- musk (musk): not up yet", result.stdout)

    def test_unknown_action_is_usage_error(self):
        self.assertEqual(self._run("launch").returncode, 2)


if __name__ == "__main__":
    unittest.main()
