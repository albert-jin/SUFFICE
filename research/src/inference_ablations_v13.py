"""Matched-width inference-only experiments. No optimizer or training calls."""
import time
import torch
from common import *
from qwen7b_adaptive_v12 import Qwen7BGenerator,BASE,ADAPTER,prompt
RUN=ROOT/'runs/inference_ablations_v13'
FREEZE=ROOT/'manifests/INFERENCE_ABLATIONS_V13_FREEZE.json'
CONDITIONS=[f'fixed_{r:.1f}' for r in RATIOS]+['adaptive_relative']

def settings():
    f=json.loads(FREEZE.read_text())
    for p,h in f['source_files'].items():assert digest(ROOT/p)==h
    for p,h in f['model_files'].items():assert digest(ROOT/p)==h
    return f

def actions(p,c,variant='relative'):
    rs=np.array(RATIOS);allow=(rs[None,:]>=.5)&(p>=p[:,-1,None]);allow[:,-1]=True
    if variant=='absolute':allow&=(p>=.97);allow[:,-1]=True
    if variant=='no_quality':allow=np.broadcast_to(rs[None,:]>=.5,p.shape)
    criterion=np.broadcast_to(rs[None,:],p.shape) if variant=='no_cost' else c
    return np.argmin(np.where(allow,criterion,np.inf),axis=1)

class MatchedWidthGenerator(Qwen7BGenerator):
    def generate_width(self,questions,ratios,width,cap=512):
        t=time.perf_counter();tok=self.tokenizer
        seqs=[tok.encode(prompt(q,r,tok),add_special_tokens=False) for q,r in zip(questions,ratios)]
        assert max(map(len,seqs))<=width and width+cap<=8192
        ids=torch.full((len(seqs),width),tok.pad_token_id,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
        for i,s in enumerate(seqs):ids[i,-len(s):]=torch.tensor(s,device='cuda');mask[i,-len(s):]=1
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():
            out=self.model.generate(input_ids=ids,attention_mask=mask,max_new_tokens=cap,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize();decode=time.perf_counter()-start;records=[]
        for tokens,s,r in zip(out[:,width:].tolist(),seqs,ratios):
            stopped=tok.eos_token_id in tokens
            if stopped:tokens=tokens[:tokens.index(tok.eos_token_id)+1]
            records.append({'text':tok.decode(tokens,skip_special_tokens=True),'token_ids':tokens,'output_tokens':len(tokens),'output_tokens_excluding_eos':len(tokens)-int(stopped),'input_tokens':len(s),'ratio':float(r),'truncated':not stopped,'generation_requests':1,'max_new_tokens':cap})
        return records,{'total_seconds':time.perf_counter()-t,'generator_prefill_decode_seconds':decode,'batch_size':len(seqs),'physical_input_width':width,'padding_rule':'same width for this question batch across ALL ratios and policies','max_new_tokens':cap}
