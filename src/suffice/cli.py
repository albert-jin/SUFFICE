"""Route precomputed predictions or frozen prefix features, without generation."""
import argparse
import numpy as np
from .io import write_jsonl
from .policy import select_ratios

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--predictions', help='NPZ with ids, prob, cost arrays.')
    inputs.add_argument('--features', help='NPZ with ids and features arrays.')
    parser.add_argument('--probe', help='External trained probe directory for feature input.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--min-ratio', type=float, default=.5)
    parser.add_argument('--slack', type=float, default=0.)
    args = parser.parse_args()
    with np.load(args.predictions or args.features, allow_pickle=False) as arrays:
        ids = arrays['ids']
        if args.features:
            if not args.probe:
                parser.error('--features requires --probe')
            from .model import ProbeEnsemble
            prediction = ProbeEnsemble(args.probe).predict(arrays['features'])
            p, c = prediction['prob'], prediction['cost']
        else:
            p, c = arrays['prob'], arrays['cost']
        result = select_ratios(p, c, min_ratio=args.min_ratio, slack=args.slack)
        if len(ids) != len(result['actions']) or len(set(map(str, ids))) != len(ids):
            raise ValueError('Prediction IDs must be unique and match the array rows.')
        write_jsonl(args.output, [{'id': str(qid), 'ratio': float(r), 'action_index': int(a),
                                   'predicted_sufficiency': float(p[i, a]), 'predicted_tokens': float(c[i, a])}
                                  for i, (qid, r, a) in enumerate(zip(ids, result['ratios'], result['actions']))])

if __name__ == '__main__':
    main()
