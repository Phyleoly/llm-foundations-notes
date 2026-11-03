from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smoke import run_smoke


class SmokeCommandTests(unittest.TestCase):
    def test_smallest_supported_config_has_an_effective_unmasked_control(self):
        for seed in [1, 2, 4, 5]:
            with self.subTest(seed=seed):
                report = run_smoke(Namespace(
                    device='cpu', dtype='float32', seed=seed, batch_size=1,
                    sequence_length=2, hidden_dim=2, num_heads=1,
                ))
                self.assertTrue(all(report['checks'].values()))

    def test_standalone_command_reports_config_evidence_and_scope(self):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "smoke.py"),
                "--sequence-length", "8",
                "--hidden-dim", "16",
                "--num-heads", "4",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        report = json.loads(result.stdout)
        self.assertEqual(report["scope"], "operator_smoke_only")
        self.assertEqual(report["config"]["device"], "cpu")
        self.assertEqual(report["input"]["shape"], [2, 8, 16])
        self.assertTrue(all(report["checks"].values()))
        self.assertLessEqual(report["metrics"]["causal_prefix_max_abs_diff"], 1e-6)
        self.assertGreater(report["metrics"]["unmasked_prefix_max_abs_diff"], 1e-5)
        self.assertIn("torch", report["environment"])
        self.assertEqual(len(report["source_sha256"]["smoke.py"]), 64)
        self.assertEqual(
            len(report["source_sha256"]["chi_2023_variance/src/model.py"]), 64
        )


if __name__ == "__main__":
    unittest.main()
