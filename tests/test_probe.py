import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

HAS_TORCH = importlib.util.find_spec('torch') is not None
REPO = Path(__file__).resolve().parents[1]

@unittest.skipUnless(HAS_TORCH, 'Install the probe extra to validate model compatibility and training.')
class ProbeTests(unittest.TestCase):
    def test_checkpoint_compatibility_with_preserved_model(self):
        import torch
        from suffice.model import SufficiencyNet
        sys.path.insert(0, str(REPO / 'research/src'))
        from sufficiency_model_v7 import SufficiencyNet as HistoricalNet
        torch.manual_seed(17)
        original = HistoricalNet(6).eval()
        current = SufficiencyNet(6).eval()
        current.load_state_dict(original.state_dict(), strict=True)
        features = torch.randn(4, 6)
        for expected, actual in zip(original(features), current(features)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_training_checkpoint_roundtrip_and_auxiliary_independence(self):
        import torch
        from suffice.model import ProbeEnsemble
        from suffice.policy import select_ratios
        rng = np.random.default_rng(17)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = [f'synthetic-{i}' for i in range(24)]
            features = rng.normal(size=(24, 6)).astype(np.float32)
            np.savez(root / 'features.npz', ids=ids, features=features)
            rows = [{'id': qid, 'split': 'train' if i < 16 else 'dev',
                     'candidate_correctness': [0, 0, 1, 0, 1, 1, 1, 1],
                     'candidate_tokens': [12, 14, 18, 16, 22, 25, 28, 32],
                     'actual_recoverability': [.125, .375, .625, .375, .625, .875, .875, .875]}
                    for i, qid in enumerate(ids)]
            (root / 'rows.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            proc = subprocess.run([sys.executable, str(REPO / 'scripts/train_probe.py'),
                    '--rows', str(root / 'rows.jsonl'), '--features', str(root / 'features.npz'),
                    '--output', str(root / 'trained'), '--device', 'cpu', '--max-epochs', '2',
                    '--minimum-epochs', '0'], text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            probe = ProbeEnsemble(root / 'trained/soft_boundary')
            first = probe.predict(features)
            self.assertEqual(first['prob'].shape, (24, 8))
            self.assertTrue(np.isfinite(first['cost']).all())
            np.testing.assert_allclose(probe.mean, features[:16].mean(0))
            for net, _, _ in probe.models:
                with torch.no_grad():
                    net.soft.weight.fill_(100)
                    net.soft.bias.fill_(100)
            second = probe.predict(features)
            np.testing.assert_array_equal(first['prob'], second['prob'])
            np.testing.assert_array_equal(first['cost'], second['cost'])
            np.testing.assert_array_equal(select_ratios(first['prob'], first['cost'])['actions'],
                                          select_ratios(second['prob'], second['cost'])['actions'])

if __name__ == '__main__':
    unittest.main()
