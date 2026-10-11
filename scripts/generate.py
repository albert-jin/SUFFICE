"""Qwen-compatible reference inference with one selected generator call per query.

The merged generator, frozen feature encoder, and trained probe are local assets.
This portable entry point has no cross-model KV reuse. Historical measured timing
belongs to research/src, not to this new convenience entry point.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from suffice.io import external_output, read_jsonl, write_jsonl
from suffice.prompts import make_prompt, token_plan
from suffice.policy import RATIOS, select_ratios

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Question-only JSONL with unique id and question.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--generator', required=True, help='Local merged ratio-controllable Qwen model.')
    parser.add_argument('--tokenizer', help='Local tokenizer; defaults to generator directory.')
    parser.add_argument('--encoder', help='Exact frozen feature model used to train the probe.')
    parser.add_argument('--encoder-tokenizer', help='Defaults to the encoder directory.')
    parser.add_argument('--probe', help='External probe checkpoint directory.')
    parser.add_argument('--fixed-ratio', type=float, choices=list(RATIOS))
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--max-context', type=int, default=8192)
    parser.add_argument('--min-ratio', type=float, default=.5)
    parser.add_argument('--slack', type=float, default=0.)
    args = parser.parse_args()
    if args.fixed_ratio is None and not (args.encoder and args.probe):
        parser.error('Adaptive inference needs --encoder and --probe.')
    if args.max_new_tokens <= 0:
        parser.error('--max-new-tokens must be positive')
    output = external_output(args.output)
    if output.exists():
        parser.error('Output exists; use a new run filename.')
    rows = read_jsonl(args.input)
    if not rows or len({x['id'] for x in rows}) != len(rows) or any('answer' in x or 'target' in x for x in rows):
        raise ValueError('Provide unique question IDs and no answer/target fields.')
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    from suffice.model import ProbeEnsemble
    if not torch.cuda.is_available():
        raise RuntimeError('This BF16 inference entry point requires CUDA.')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.generator, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    generator = AutoModelForCausalLM.from_pretrained(args.generator, torch_dtype=torch.bfloat16,
                 attn_implementation='sdpa', local_files_only=True).cuda().eval().requires_grad_(False)
    if args.fixed_ratio is None:
        encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder_tokenizer or args.encoder, local_files_only=True)
        encoder = AutoModel.from_pretrained(args.encoder, torch_dtype=torch.bfloat16,
                  attn_implementation='sdpa', local_files_only=True).cuda().eval().requires_grad_(False)
        probe = ProbeEnsemble(args.probe)
    outputs = []
    for row in rows:
        torch.cuda.synchronize()
        start = time.perf_counter()
        if args.fixed_ratio is None:
            prefix = token_plan(encoder_tokenizer, row['question'])[0]
            if len(prefix) > args.max_context:
                raise ValueError('Feature prefix exceeds context guard.')
            ids = torch.tensor([prefix], device='cuda')
            with torch.inference_mode():
                hidden = encoder(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state[:, -1]
            prediction = probe.predict(hidden.float().cpu().numpy())
            ratio = float(select_ratios(prediction['prob'], prediction['cost'], min_ratio=args.min_ratio, slack=args.slack)['ratios'][0])
        else:
            ratio = args.fixed_ratio
        sequences = {float(r): tokenizer.encode(make_prompt(row['question'], r), add_special_tokens=False) for r in RATIOS}
        width = max(map(len, sequences.values()))
        if width + args.max_new_tokens > args.max_context:
            raise ValueError('Prompt plus generation cap exceeds context guard.')
        sequence = sequences[ratio]
        ids = torch.full((1, width), tokenizer.pad_token_id, dtype=torch.long, device='cuda')
        mask = torch.zeros_like(ids)
        ids[0, -len(sequence):] = torch.tensor(sequence, device='cuda')
        mask[0, -len(sequence):] = 1
        with torch.inference_mode():
            generated = generator.generate(input_ids=ids, attention_mask=mask, max_new_tokens=args.max_new_tokens,
                         do_sample=False, temperature=None, top_p=None, top_k=None, use_cache=True,
                         pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        tokens = generated[0, width:].tolist()
        stopped = tokenizer.eos_token_id in tokens
        if stopped:
            tokens = tokens[:tokens.index(tokenizer.eos_token_id) + 1]
        torch.cuda.synchronize()
        outputs.append({'id': row['id'], 'ratio': ratio, 'text': tokenizer.decode(tokens, skip_special_tokens=True),
                        'token_ids': tokens, 'output_tokens': len(tokens), 'generation_requests': 1,
                        'truncated': not stopped, 'physical_input_width': width,
                        'elapsed_seconds': time.perf_counter() - start})
    write_jsonl(output, outputs)
    print(json.dumps({'queries': len(outputs), 'generation_calls': len(outputs), 'output_tokens': sum(x['output_tokens'] for x in outputs)}))

if __name__ == '__main__':
    main()
