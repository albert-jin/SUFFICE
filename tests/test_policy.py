import unittest
import numpy as np
from suffice import RATIOS, admissible, select_ratios
from suffice.prompts import token_plan

class PolicyTests(unittest.TestCase):
    def test_fallback_when_all_compressed_options_fail(self):
        p = np.array([[.1] * 7 + [.9]])
        result = select_ratios(p, np.ones_like(p))
        self.assertEqual(result['ratios'].tolist(), [1.])

    def test_select_actual_cost_not_smallest_ratio(self):
        p = np.full((1, 8), .8)
        c = np.array([[1, 1, 90, 80, 30, 70, 60, 120]])
        result = select_ratios(p, c)
        self.assertEqual(result['ratios'].tolist(), [.7])
        self.assertFalse(result['allowed'][0, 0])
        self.assertFalse(result['allowed'][0, 1])

    def test_gate_is_invariant_to_per_query_monotone_transform(self):
        rng = np.random.default_rng(5)
        p = rng.random((100, 8))
        transformed = np.exp(p * rng.uniform(.1, 3, (100, 1)))
        np.testing.assert_array_equal(admissible(p), admissible(transformed))

    def test_nonzero_slack_changes_admission(self):
        p = np.array([[.1, .1, .77, .1, .1, .1, .1, .8]])
        self.assertFalse(admissible(p)[0, 2])
        self.assertTrue(admissible(p, slack=.04)[0, 2])

    def test_ties_match_original_increasing_ratio_order(self):
        self.assertEqual(select_ratios(np.ones((1, 8)), np.ones((1, 8)))['ratios'].tolist(), [.5])

    def test_invalid_predictions_do_not_silently_route(self):
        for p, c in [(np.full((1, 8), np.nan), np.ones((1, 8))),
                     (np.ones((1, 8)), np.zeros((1, 8))),
                     (np.ones((1, 8)), np.ones((1, 7)))]:
            with self.assertRaises(ValueError):
                select_ratios(p, c)

    def test_exact_prefix_and_suffix_reconstruct_all_prompts(self):
        class CharacterTokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(text.encode())
        prefix, suffix, full = token_plan(CharacterTokenizer(), 'A synthetic question?')
        for r in RATIOS:
            self.assertEqual(prefix + suffix[r], full[r])
        self.assertNotIn(b'<|eot_id|>', bytes(prefix))

if __name__ == '__main__':
    unittest.main()
