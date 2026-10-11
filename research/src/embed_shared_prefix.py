"""Features available from the generator's required common-prompt prefill."""
import argparse,time
from common import *
from shared_prefix import token_plan
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();inp=Path(a.input);out=Path(a.out);rows=readl(inp)
    assert all(x.get('split') in ['train','dev'] for x in rows),'This development extractor may not open reserve/final data'
    if out.exists():
        assert json.loads(out.with_suffix('.json').read_text())['input_sha256']==digest(inp);return
    import torch
    from transformers import AutoTokenizer,AutoModel
    torch.set_num_threads(4);tok=AutoTokenizer.from_pretrained(str(ROOT/'models/base'),local_files_only=True);model=AutoModel.from_pretrained(str(ROOT/'models/merged'),torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
    plans=[token_plan(tok,x['question'])[0] for x in rows];assert max(map(len,plans))<=3072;features=[];seconds=[];start=time.time()
    with torch.inference_mode():
        for k in range(0,len(rows),8):
            batch=plans[k:k+8];n=max(map(len,batch));ids=torch.full((len(batch),n),tok.pad_token_id,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
            for i,v in enumerate(batch):ids[i,:len(v)]=torch.tensor(v,device='cuda');mask[i,:len(v)]=1
            pos=(mask.cumsum(-1)-1).clamp_min(0);torch.cuda.synchronize();t=time.perf_counter();h=model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False).last_hidden_state;v=h[torch.arange(len(batch),device='cuda'),mask.sum(-1)-1];torch.cuda.synchronize();seconds.append(time.perf_counter()-t);features.append(v.float().cpu().numpy())
    out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,ids=np.array([x['id'] for x in rows]),features=np.concatenate(features),prefix_token_counts=np.array(list(map(len,plans))))
    dump(out.with_suffix('.json'),{'input_sha256':digest(inp),'generator_merge_manifest_sha256':digest(ROOT/'manifests/MERGED_READY.json'),'n':len(rows),'shape':list(np.concatenate(features).shape),'prefill_tokens':sum(map(len,plans)),'offline_forward_seconds':sum(seconds),'offline_wall_seconds':time.time()-start,'prefix_rule':'Longest common token prefix across the frozen candidate retention prompts','deployment_note':'Online use must reuse this generator prefix cache; separate extraction here is training data construction, not evidence of online speed.'})
if __name__=='__main__':main()
