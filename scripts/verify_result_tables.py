"""Recompute released aggregate means from local-only per-question records."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]

def read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=REPO.parent / 'SUFFICE-DATASETS')
    args = parser.parse_args()
    rows = read_jsonl(args.data_root / 'historical/inference_ablations_v13_complete/runs/inference_ablations_v13/policy_rows_screen.jsonl')
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['method']].append(row)
    with (REPO / 'results/v13/all_29_methods_with_paired_statistics.csv').open(newline='', encoding='utf-8-sig') as stream:
        table = list(csv.DictReader(stream))
    for record in table:
        group = grouped[record['method']]
        assert len(group) == len({x['id'] for x in group}) == int(record['n'])
        actual = [sum(x['correct'] for x in group), 100 * np.mean([x['correct'] for x in group]),
                  np.mean([x['output_tokens'] for x in group]), np.mean([x['paper_cot_tokens'] for x in group])]
        expected = [float(record[k]) for k in ['correct', 'accuracy_percent', 'mean_output_tokens', 'mean_paper_cot_tokens']]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
    timing = read_jsonl(args.data_root / 'historical/v14/latency_three_trials.jsonl')
    with (REPO / 'results/v14/latency_summary.csv').open(newline='', encoding='utf-8-sig') as stream:
        latency_table = list(csv.DictReader(stream))
    for record in latency_table:
        group = [x for x in timing if x['method'] == record['method']]
        assert len(group) == int(record['observations'])
        assert len({x['id'] for x in group}) == int(record['n_questions'])
        np.testing.assert_allclose(np.mean([x['total_request_seconds'] for x in group]), float(record['mean_seconds']), rtol=0, atol=1e-10)
    print(json.dumps({'v13_methods_verified': len(table), 'v14_methods_verified': len(latency_table),
                      'new_model_inference_performed': False}))

if __name__ == '__main__':
    main()
