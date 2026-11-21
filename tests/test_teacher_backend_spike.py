from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from jorjor.teacher_backend_spike import run_teacher_backend_spike


class TeacherBackendSpikeTest(unittest.TestCase):
    """Test k.0.1 teacher backend availability and local CFR fallback."""

    def test_teacher_backend_spike_writes_recommendation_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "teacher_backend_spike.json"
            payload = run_teacher_backend_spike(output_path=output_path)
            saved = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(saved["report_type"], "k.0.1_teacher_backend_spike")
        self.assertEqual(saved["checksum"], payload["checksum"])
        self.assertIn("pyspiel", saved["external_backends"])
        self.assertIn("local_leduc_cfr", saved["local_backends"])
        self.assertIn("local_limit_abstract_cfr", saved["local_backends"])
        self.assertEqual(saved["local_backends"]["local_leduc_cfr"]["illegal_action_count"], 0)
        self.assertEqual(saved["local_backends"]["local_limit_abstract_cfr"]["illegal_action_count"], 0)
        self.assertEqual(saved["local_backend_pass_count"], 2)
        self.assertEqual(saved["recommended_backend"], "local_limit_abstract_cfr_for_k_0_4")
        self.assertIn("does_not_prove_belief_tracking", saved["claim_boundary"])

    def test_teacher_backend_spike_script_runs_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_teacher_backend_spike.py",
                    "--output-path",
                    str(Path(temp_dir) / "teacher_backend_spike.json"),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("wrote k.0.1 teacher backend spike", result.stdout)


if __name__ == "__main__":
    unittest.main()
