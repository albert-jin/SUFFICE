import importlib.util
from pathlib import Path
import unittest
from suffice.policy import RATIOS

spec = importlib.util.spec_from_file_location('assemble_outcome_bank', Path(__file__).resolve().parents[1] / 'scripts/assemble_outcome_bank.py')
bank = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bank)

class BankTests(unittest.TestCase):
    def setUp(self):
        self.inputs = [{'id': 'synthetic', 'split': 'train'}]
        self.retention = [{'id': 'synthetic', 'ratio': float(r), 'correct': 1, 'output_tokens': 20} for r in RATIOS]
        self.recovery = [{'id': 'synthetic', 'ratio': float(r), 'correct': int(seed != 17), 'seed': seed, 'condition': 'true'} for r in RATIOS for seed in (17, 43, 79)]

    def test_jeffreys_smoothed_targets(self):
        result = bank.assemble(self.inputs, self.retention, self.recovery)
        self.assertEqual(result[0]['actual_recoverability'], [.625] * 8)

    def test_test_split_is_rejected(self):
        with self.assertRaises(ValueError):
            bank.assemble([{'id': 'synthetic', 'split': 'test'}], self.retention, self.recovery)

    def test_incomplete_recovery_is_rejected(self):
        with self.assertRaises(ValueError):
            bank.assemble(self.inputs, self.retention, self.recovery[:-1])

    def test_scoring_errors_are_not_silently_labels(self):
        self.retention[0]['scoring_error'] = 'TimeoutError'
        with self.assertRaises(ValueError):
            bank.assemble(self.inputs, self.retention, self.recovery)
