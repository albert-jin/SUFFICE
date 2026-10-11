"""Exact token plans for within-request prefix reuse; no generated probes."""
from common import RATIOS,make_prompt
def token_plan(tokenizer,question):
    full={r:tokenizer.encode(make_prompt(question,r),add_special_tokens=False) for r in RATIOS}
    sequences=list(full.values());n=0
    for tokens in zip(*sequences):
        if len(set(tokens))!=1:break
        n+=1
    assert 0<n<min(map(len,sequences)),'Need a nonempty common prefix and action suffix'
    prefix=sequences[0][:n];suffix={r:ids[n:] for r,ids in full.items()}
    assert all(prefix+suffix[r]==full[r] for r in RATIOS)
    return prefix,suffix,full

