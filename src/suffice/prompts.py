"""The recorded Qwen/TokenSkip prompt and its exact common token prefix."""
from .policy import RATIOS

def make_prompt(question, ratio):
    if float(ratio) not in RATIOS:
        raise ValueError('Unsupported retention ratio.')
    suffix = f'<|eot_id|>{ratio:.1f}<|eot_id|>' if ratio < 1 else ''
    return ('<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
            '<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{}.\n'
            + question + suffix + '<|im_end|>\n<|im_start|>assistant\n')

def token_plan(tokenizer, question):
    full = {float(r): tokenizer.encode(make_prompt(question, r), add_special_tokens=False) for r in RATIOS}
    sequences = list(full.values())
    n = 0
    for tokens in zip(*sequences):
        if len(set(tokens)) != 1:
            break
        n += 1
    if not 0 < n < min(map(len, sequences)):
        raise ValueError('Need a nonempty shared prefix and a ratio-dependent suffix.')
    prefix = sequences[0][:n]
    suffixes = {r: ids[n:] for r, ids in full.items()}
    return prefix, suffixes, full
