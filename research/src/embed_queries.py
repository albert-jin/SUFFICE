"""Frozen same-model query features; record separate prefill cost, no answers."""
import argparse,time,os
from common import *
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    import torch
    from transformers import AutoTokenizer,AutoModel
    rows=readl(args.input);path=Path(args.out)
    if path.exists():print('EXISTS',str(path));return
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/base'),local_files_only=True);tok.padding_side='right'
    model=AutoModel.from_pretrained(str(ROOT/'models/base'),torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
    feats=[];tokens=0;durations=[];start=time.time()
    with torch.inference_mode():
        for offset in range(0,len(rows),8):
            batch=rows[offset:offset+8];enc=tok([feature_prompt(x['question']) for x in batch],padding=True,return_tensors='pt')
            if enc.input_ids.shape[1]>3072:raise ValueError('Query exceeds frozen feature context; do not silently truncate')
            enc={k:v.cuda() for k,v in enc.items()};torch.cuda.synchronize();t=time.perf_counter();h=model(**enc,use_cache=False).last_hidden_state
            lengths=enc['attention_mask'].sum(-1);v=h[torch.arange(len(batch),device='cuda'),lengths-1]
            torch.cuda.synchronize();durations.append(time.perf_counter()-t);feats.append(v.float().cpu().numpy());tokens+=int(lengths.sum())
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,ids=np.array([x['id'] for x in rows]),features=np.concatenate(feats))
    dump(path.with_suffix('.json'),{'input_sha256':digest(args.input),'n':len(rows),'feature_shape':list(np.concatenate(feats).shape),'prefill_tokens':tokens,'batch_size':8,'forward_seconds':sum(durations),'wall_seconds':time.time()-start,'online_note':'This is an additional base-model prefill, must be measured and charged in online end-to-end latency; no free reuse assumed.'})
if __name__=='__main__':main()
