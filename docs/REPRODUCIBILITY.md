# Reproduction guide

## Three distinct uses

1. **Use the policy:** `src/suffice/` provides the zero-slack relative gate, minimum predicted-token selection, checkpoint-compatible probe, and routing CLI. NumPy alone is enough to route precomputed predictions.
2. **Train on new data:** prepare external datasets, generate and grade an offline retention/recovery bank, extract prefix features, then train only the probe. New runs must be reported under their own configurations.
3. **Audit historical results:** original scripts and aggregate tables are retained. Local question-level records support numerical replay. Exact GPU replay additionally needs the matching models, adapters, manifests, numerical settings, and historical split IDs, which are not supplied by this repository.

## Build a new outcome bank

The historical bank and scoring scripts require Linux and CUDA. Create a fresh external workspace, install the package plus inference/data dependencies and a vLLM environment compatible with the recorded bank-generation runtime, and set `AT_ROOT` to that workspace:

```bash
python scripts/stage_workspace.py --output ../SUFFICE-WORKSPACE
export AT_ROOT="$(cd ../SUFFICE-WORKSPACE && pwd)"
```

Place the matching frozen Qwen2.5-3B base model/tokenizer in `models/base/` and its merged TokenSkip source generator in `models/merged/`. The historical feature extractor reads the tokenizer from `models/base/` and model weights from `models/merged/`; using a different encoder changes the features and requires a new probe. Model weights must be acquired separately under their source terms. The source generator and recovery model stay frozen.

`prepare_datasets.py` combines reconstructed `train.jsonl` and `dev.jsonl` in that order into the external `development_pool.jsonl`. It preserves IDs and explicit `split` fields. The schema includes `id`, `dataset`, `question`, `answer`, `task`, and, for MCQ, `n_choices`. All questions in this stage must be training or development questions.

```bash
python research/src/generate_bank.py --input ../SUFFICE-DATASETS/development_pool.jsonl --run retention --mode retention
python research/src/score_bank.py --input ../SUFFICE-DATASETS/development_pool.jsonl --bank ../SUFFICE-WORKSPACE/runs/retention/bank.jsonl
python research/src/generate_bank.py --input ../SUFFICE-DATASETS/development_pool.jsonl --run recovery --mode recovery --source ../SUFFICE-WORKSPACE/runs/retention/bank.jsonl
python research/src/score_bank.py --input ../SUFFICE-DATASETS/development_pool.jsonl --bank ../SUFFICE-WORKSPACE/runs/recovery/bank.jsonl
python scripts/assemble_outcome_bank.py --input ../SUFFICE-DATASETS/development_pool.jsonl --retention ../SUFFICE-WORKSPACE/runs/retention/scores.jsonl --recovery ../SUFFICE-WORKSPACE/runs/recovery/scores.jsonl --output ../SUFFICE-WORKSPACE/training_rows.jsonl
```

`assemble_outcome_bank.py` rejects missing/duplicate per-ratio scores, grading failures, missing recovery seeds, and non-training/development inputs. Recovery uses three seeds (17, 43, 79), removes explicit final-answer text, and constructs Jeffreys-smoothed targets. It is offline supervision; recovery is never an online routing input.

The historical feature extractor additionally requires a `manifests/MERGED_READY.json` provenance record for the merged model. Record the model identifiers, immutable revisions, adapter, merge procedure, and checksums before extraction. It hashes this record; it does not construct model assets for you.

```bash
python research/src/embed_shared_prefix.py --input ../SUFFICE-DATASETS/development_pool.jsonl --out ../SUFFICE-WORKSPACE/features.npz
python scripts/train_probe.py --rows ../SUFFICE-WORKSPACE/training_rows.jsonl --features ../SUFFICE-WORKSPACE/features.npz --output ../SUFFICE-WORKSPACE/probe_run --device cuda
```

The default recipe is `soft_boundary`. `soft_cost` jointly removes boundary and soft-regret terms; it is not a separate one-loss ablation. The other retained recipes add monotonicity and asymmetric BCE. If asymmetric BCE is used, inference corrects logits by adding the log negative-class weight before sigmoid. Recoverability predictions do not enter action selection. The cost head uses train-only ratio-wise smearing after exponentiation.

## Historical runtime and numerical controls

- V13 uses a Qwen2.5-7B-Instruct generator with the upstream GSM8K LoRA and a transferred frozen 3B probe. This is not checkpoint-aligned to the upstream paper's CSQA adapter.
- `qwen7b_adaptive_v12.py` retains validation and manual merging for that particular standard rank-8, alpha-16 adapter. It is not a general LoRA merger.
- `inference_ablations_v13.py` uses a physical input width fixed across all candidate ratios and policies for each question batch. Changing padding can change BF16 features, routing, and generated outputs.
- The original CPU probe uses three seeds. `shared_decode.py` includes an earlier joint-controller implementation as a dependency and same-model cache-reuse experiments; it is not the final SUFFICE policy.
- Cross-model 3B-to-7B transfer cannot share a KV cache. Its encoder cost is charged separately.
- `freeze_inference_ablations_v13.py` depends on preceding V12 artifacts. Preserved hashes identify historical files; relocation and code changes produce new hashes. Do not rewrite a historical freeze to pass an integrity check. Construct a new run and freeze under a new identity.
- `research/src/train_sufficiency_v7.py` retains historical sample-count and layout assumptions. `scripts/train_probe.py` is the configurable entry point and preserves the training mathematics and state keys.
- `scripts/generate.py` is a new Qwen-compatible convenience interface. GPU execution of this wrapper was not rerun during release preparation; measured latency tables come from the historical runtime.

## Data completeness and provenance

The download manifest pins five source repositories covering four task families. GSM8K main, all CommonsenseQA splits, all MMLU splits, all seven MATH subjects (train and test), and MATH-500 test are stored outside the repository. MMLU-STEM uses the 18-subject list in `research/src/prepare_data.py`; its labeled test subset has 3,018 examples.

CommonsenseQA validation has 1,221 labeled examples. Its separate official test set has 1,140 examples with no answer key. The historical filename `csqa_test.parquet` refers to labeled validation, not the hidden-label test split.

The split constructor uses seed 20261009 and removes question duplicates against evaluation questions. MATH training order and other public-source differences can change reconstructed IDs and split membership. Keep `reconstructed_splits/` separate from `historical/`. Download/preparation manifests record checksums and counts; they are local artifacts, not code-repository data payloads.

## Reported evidence and limits

V13's development and screening questions were reused across earlier experiments. V14 repeats 32 development questions across three trials, including one earlier time block. Neither is a fresh independent held-out confirmation. Reported cross-dataset snapshots in `four_datasets_current_snapshot.csv` mix generators, adapters, splits, and sample sizes; use their explicit comparison flags rather than attributing differences to the method.

The release verification script checks all 29 V13 method means and three V14 latency means against local per-question records. It does not recalculate every historical confidence interval or rerun generation. The paired intervals and tests in the CSVs remain attributed to the recorded analyses. No additional GPU training or evaluation was conducted during packaging.
