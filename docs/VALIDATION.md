# Release validation

- **13 tests passed**, including full-retention fallback, cost-based selection, monotone-score invariance, ratio-independent prompt construction, incomplete-bank rejection, and test-split exclusion.
- The portable probe loads the preserved model's state dictionary and produces identical outputs in the compatibility test.
- A small synthetic CPU run trains all three probe seeds, saves and reloads checkpoints, verifies train-only normalization, and confirms that changing the auxiliary recovery head cannot change online routing.
- Tests ran on Python 3.12, PyTorch 2.4.1, and NumPy 1.26.4. The historical GPU environment is described separately in the README.
- All 29 V13 method aggregate means and three V14 latency means match the local question-level records.
- Downloaded dataset revisions and checksums are recorded outside the repository. Verified labeled evaluation counts: GSM8K 1,319; MATH-500 500; CommonsenseQA 1,221; MMLU-STEM 3,018. Reconstructed train/dev/calibration/reserve counts are 6,143 / 900 / 899 / 901, with no duplicate normalized questions across those four splits.
- Five displayed PNG figures have PDF companions. The framework is 3840 by 1640 pixels. PNG metadata and PDF author metadata were removed from release copies.
- The release guard checks Python syntax, excluded payload extensions, private paths, potential credentials, README hyperlinks, and image references.

No full GPU training, new benchmark generation, or GPU execution of the portable inference wrapper was performed during packaging. Historical results retain their original development/screening scope. The synthetic CPU training test is software validation, not an experiment result.
