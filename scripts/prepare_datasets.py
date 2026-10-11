"""Normalize downloaded benchmarks and run the preserved split constructor.

New splits reconstructed from pinned public sources are explicitly separate
from the historical experiment records. No exact historical split claim is made.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'research/upstream'))
sys.path.insert(0, str(REPO / 'research/src'))

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPO.parent / 'SUFFICE-DATASETS')
    args = parser.parse_args()
    root = args.root.resolve()
    if root == REPO or REPO in root.parents:
        parser.error('Dataset root must be outside the repository.')
    import pyarrow.parquet as pq
    from data_processing.answer_extraction import extract_math_answer
    raw = root / 'raw'
    prepared = root / 'prepared_raw'
    prepared.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ['train', 'test']:
        source = raw / 'openai/gsm8k/main' / f'{split}-00000-of-00001.parquet'
        rows = pq.read_table(source).to_pylist()
        normalized = []
        for row in rows:
            if '####' not in row['answer']:
                raise ValueError('GSM8K answer delimiter missing.')
            normalized.append({'question': row['question'], 'answer': row['answer'].rsplit('####', 1)[1].strip()})
        write_jsonl(prepared / 'gsm8k' / f'{split}.jsonl', normalized)
        counts['gsm8k_' + split] = len(rows)
    math_train = []
    normalization_notes = []
    math_test_count = 0
    for source in sorted((raw / 'EleutherAI/hendrycks_math').glob('*/train-*.parquet')):
        for index, row in enumerate(pq.read_table(source).to_pylist()):
            answer = extract_math_answer(row['problem'], row['solution'], 'cot')
            if not answer:
                # Two source solutions use valid single-token TeX (e.g. \\boxed 2),
                # which the retained brace-based evaluation helper does not parse.
                match = re.search(r'\\boxed\s+([0-9])\s*\$', row['solution'])
                if not match:
                    raise ValueError('Cannot extract MATH reference answer.')
                answer = [match.group(1)]
                normalization_notes.append({'file': source.relative_to(raw).as_posix(), 'row': index,
                                            'operation': 'Read unbraced single-digit boxed reference from source solution.'})
            math_train.append({**row, 'question': row['problem'], 'answer': answer})
    for source in sorted((raw / 'EleutherAI/hendrycks_math').glob('*/test-*.parquet')):
        math_test_count += pq.ParquetFile(source).metadata.num_rows
    write_jsonl(prepared / 'math-500/train.jsonl', math_train)
    rows = [json.loads(line) for line in (raw / 'HuggingFaceH4/MATH-500/test.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    write_jsonl(prepared / 'math-500/test.jsonl', rows)
    counts.update(math_train=len(math_train), math_original_test=math_test_count, math500_test=len(rows))
    for split, output in [('train', 'csqa_train'), ('validation', 'csqa_test'), ('test', 'csqa_unlabeled_test')]:
        table = pq.read_table(raw / 'tau/commonsense_qa/data' / f'{split}-00000-of-00001.parquet')
        pq.write_table(table, prepared / (output + '.parquet'))
        counts['commonsenseqa_' + split] = table.num_rows
    for split in ['auxiliary_train', 'dev', 'validation', 'test']:
        table = pq.read_table(raw / 'cais/mmlu/all' / f'{split}-00000-of-00001.parquet')
        pq.write_table(table, prepared / ('mmlu_' + split + '.parquet'))
        counts['mmlu_all_' + split] = table.num_rows
    expected = {'gsm8k_train': 7473, 'gsm8k_test': 1319, 'math_train': 7500,
                'math_original_test': 5000, 'math500_test': 500, 'commonsenseqa_train': 9741,
                'commonsenseqa_validation': 1221, 'commonsenseqa_test': 1140}
    for key, value in expected.items():
        if counts[key] != value:
            raise ValueError(f'Unexpected source count for {key}: {counts[key]} != {value}')
    subprocess.run([sys.executable, str(REPO / 'research/src/prepare_data.py'),
                    '--raw', str(prepared), '--out', str(root / 'reconstructed_splits')], check=True)
    development = []
    for split in ['train', 'dev']:
        path = root / 'reconstructed_splits' / (split + '.jsonl')
        development.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip())
    write_jsonl(root / 'development_pool.jsonl', development)
    manifest = {'counts': counts, 'normalization_notes': normalization_notes, 'evaluation_split': {'gsm8k': 'test', 'math500': 'test',
                'commonsenseqa': 'validation (labeled)', 'mmlu_stem': 'test, 18 STEM subjects'},
                'provenance': 'Reconstructed from pinned public revisions; historical records are separate.',
                'normalized_files': {p.relative_to(prepared).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in prepared.rglob('*') if p.is_file()}}
    (root / 'PREPARATION_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(counts, indent=2))

if __name__ == '__main__':
    main()
