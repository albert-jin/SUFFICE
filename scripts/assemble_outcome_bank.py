"""Join complete retention and K=3 recovery scores for probe-only training."""
import argparse
from collections import defaultdict
from suffice.io import read_jsonl, write_jsonl
from suffice.policy import RATIOS

def assemble(inputs, retention, recovery):
    if not inputs or any(x.get('split') not in {'train', 'dev'} for x in inputs):
        raise ValueError('Only explicitly marked train/dev questions may form a training bank.')
    ids = {x['id'] for x in inputs}
    if len(ids) != len(inputs):
        raise ValueError('Duplicate question IDs.')
    candidates = {}
    recoveries = defaultdict(dict)
    for row in retention:
        if row['id'] not in ids:
            continue
        key = (row['id'], float(row['ratio']))
        if key in candidates or row.get('correct') not in (0, 1) or row.get('scoring_error') is not None or row['output_tokens'] <= 0:
            raise ValueError('Duplicate, failed, or invalid retention score: ' + str(key))
        candidates[key] = row
    for row in recovery:
        if row['id'] not in ids or row.get('condition') != 'true':
            continue
        key = (row['id'], float(row['ratio']))
        if row['seed'] in recoveries[key] or row.get('correct') not in (0, 1) or row.get('scoring_error') is not None:
            raise ValueError('Duplicate or failed recovery score: ' + str(key))
        recoveries[key][row['seed']] = row['correct']
    result = []
    for question in inputs:
        y, c, q = [], [], []
        for ratio in RATIOS:
            key = (question['id'], float(ratio))
            if key not in candidates or set(recoveries[key]) != {17, 43, 79}:
                raise ValueError('Incomplete outcome/recovery bank: ' + str(key))
            y.append(candidates[key]['correct'])
            c.append(candidates[key]['output_tokens'])
            q.append((sum(recoveries[key].values()) + .5) / 4)
        result.append({'id': question['id'], 'split': question['split'],
                       'candidate_correctness': y, 'candidate_tokens': c, 'actual_recoverability': q})
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--retention', required=True)
    parser.add_argument('--recovery', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = assemble(read_jsonl(args.input), read_jsonl(args.retention), read_jsonl(args.recovery))
    write_jsonl(args.output, rows)
    print('Assembled {} training/development rows in input order.'.format(len(rows)))

if __name__ == '__main__':
    main()
