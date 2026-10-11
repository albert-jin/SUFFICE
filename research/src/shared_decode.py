"""One generation branch after a query-only common-prefix prefill.

Matched cached baselines use the same question batches and fixed suffix width.
Ordinary full-prompt generation remains a separately measured control.
"""
import time,copy
import torch
from common import *
from shared_prefix import token_plan
from controller_runtime import Controller
from dual_view_policy import choose_dual_view

class DualRouter(Controller):
    def __init__(self,path,config):super().__init__(path);self.config=config
    def route(self,features):
        x=torch.tensor((np.asarray(features,dtype=np.float32)-self.mean)/self.std);joints=[];recover=[];cost=[]
        with torch.inference_mode():
            for net,smear in self.models:
                j,q,c=net(x);joints.append(j.softmax(-1).numpy());recover.append(q.sigmoid().numpy());cost.append((np.exp(c.numpy().clip(0,9))*smear-1).clip(1))
        sj=np.stack(joints);sq=np.stack(recover);sc=np.stack(cost)
        return choose_dual_view(sj,sq,sc,**self.config)

class SharedDecoder:
    def __init__(self):
        from transformers import AutoModelForCausalLM,AutoTokenizer
        torch.set_num_threads(4);self.tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'models/base'),local_files_only=True);self.tokenizer.padding_side='left'
        t=time.perf_counter();self.model=AutoModelForCausalLM.from_pretrained(str(ROOT/'models/merged'),torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval();torch.cuda.synchronize();self.load_seconds=time.perf_counter()-t
        self.opts=dict(do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,eos_token_id=self.tokenizer.eos_token_id,pad_token_id=self.tokenizer.pad_token_id)
    def _pad(self,sequences,width=None):
        width=max(map(len,sequences)) if width is None else width
        ids=torch.full((len(sequences),width),self.tokenizer.pad_token_id,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
        for i,s in enumerate(sequences):ids[i,width-len(s):]=torch.tensor(s,device='cuda');mask[i,width-len(s):]=1
        return ids,mask
    def prefill(self,questions,features=True):
        start=time.perf_counter();plans=[token_plan(self.tokenizer,q) for q in questions];plan_seconds=time.perf_counter()-start
        prefixes=[p[0] for p in plans];ids,mask=self._pad(prefixes);positions=(mask.cumsum(-1)-1).clamp_min(0)
        if ids.shape[1]>7168:raise ValueError('Question prefix exceeds the declared context guard; do not silently truncate')
        torch.cuda.synchronize();t=time.perf_counter()
        with torch.inference_mode():pre=self.model.model(input_ids=ids,attention_mask=mask,position_ids=positions,use_cache=True,return_dict=True)
        h=pre.last_hidden_state[:,-1].float().cpu().numpy() if features else None
        torch.cuda.synchronize();seconds=time.perf_counter()-t
        return {'plans':plans,'ids':ids,'mask':mask,'cache':pre.past_key_values,'features':h,'prefix_tokens':list(map(len,prefixes)),'prefix_width':ids.shape[1],'suffix_width':max(len(s) for p in plans for s in p[1].values()),'plan_seconds':plan_seconds,'prefill_seconds':seconds,'consumed':False}
    def _generate(self,ids,mask,cap,cache=None):
        widths=[]
        def hook(module,args,kwargs):
            x=kwargs.get('input_ids',args[0] if args else None)
            if x is not None:widths.append(int(x.shape[1]))
        handle=self.model.register_forward_pre_hook(hook,with_kwargs=True)
        torch.cuda.synchronize();t=time.perf_counter()
        try:
            with torch.inference_mode():out=self.model.generate(input_ids=ids,attention_mask=mask,past_key_values=cache,max_new_tokens=cap,**self.opts)
            torch.cuda.synchronize();seconds=time.perf_counter()-t
        finally:handle.remove()
        records=[]
        for seq in out[:,ids.shape[1]:].tolist():
            if self.tokenizer.eos_token_id in seq:
                seq=seq[:seq.index(self.tokenizer.eos_token_id)+1];finish='stop'
            else:finish='length'
            non_stop=[v for v in seq if v!=self.tokenizer.eos_token_id]
            records.append({'text':self.tokenizer.decode(seq,skip_special_tokens=True),'token_ids':seq,'output_tokens':len(seq),'output_tokens_excluding_eos':len(non_stop),'finish_reason':finish,'truncated':finish=='length'})
        return records,seconds,widths
    def continue_one_branch(self,batch,ratios,cap,technical_copy=False):
        assert len(ratios)==len(batch['plans'])
        if batch['consumed'] and not technical_copy:raise RuntimeError('Prefix already consumed; online method may generate only one branch')
        tails=[p[1][float(r)] for p,r in zip(batch['plans'],ratios)];tail,tm=self._pad(tails,batch['suffix_width']);ids=torch.cat([batch['ids'],tail],dim=1);mask=torch.cat([batch['mask'],tm],dim=1)
        cache=copy.deepcopy(batch['cache']) if technical_copy else batch['cache']
        if not technical_copy:batch['consumed']=True
        if ids.shape[1]+cap>8192:raise ValueError('Input plus output exceeds frozen 8192 context guard')
        records,seconds,widths=self._generate(ids,mask,cap,cache)
        assert widths and widths[0]==batch['suffix_width'] and all(w==1 for w in widths[1:]),'A cached generation must process only the suffix, then single new tokens'
        for i,(row,p,r) in enumerate(zip(records,batch['plans'],ratios)):
            row.update(ratio=float(r),input_tokens=len(p[2][float(r)]),prefix_tokens=batch['prefix_tokens'][i],suffix_tokens=len(tails[i]),generation_requests=1)
        return records,{'suffix_decode_seconds':seconds,'forward_input_widths':widths,'prefix_processed_once':True,'physical_prefix_width':batch['prefix_width'],'physical_suffix_width':batch['suffix_width'],'batch_size':len(ratios),'technical_cache_copy':technical_copy}
    def adaptive(self,questions,router,cap):
        start=time.perf_counter();batch=self.prefill(questions,features=True);t=time.perf_counter();actions=router.route(batch['features']);routing=time.perf_counter()-t;ratios=[RATIOS[j] for j in actions]
        rows,times=self.continue_one_branch(batch,ratios,cap);times.update(plan_seconds=batch['plan_seconds'],prefix_seconds=batch['prefill_seconds'],router_seconds=routing,total_seconds=time.perf_counter()-start)
        return rows,times
    def fixed_cached(self,questions,ratios,cap):
        start=time.perf_counter();batch=self.prefill(questions,features=False);rows,times=self.continue_one_branch(batch,ratios,cap);times.update(plan_seconds=batch['plan_seconds'],prefix_seconds=batch['prefill_seconds'],router_seconds=0.,total_seconds=time.perf_counter()-start);return rows,times
    def ordinary(self,questions,ratio,cap):
        start=time.perf_counter();t=time.perf_counter();sequences=[self.tokenizer.encode(make_prompt(q,ratio),add_special_tokens=False) for q in questions];ids,mask=self._pad(sequences);planning=time.perf_counter()-t
        rows,seconds,widths=self._generate(ids,mask,cap)
        for row,seq in zip(rows,sequences):row.update(ratio=ratio,input_tokens=len(seq),generation_requests=1)
        return rows,{'ordinary_prefill_decode_seconds':seconds,'plan_seconds':planning,'router_seconds':0.,'total_seconds':time.perf_counter()-start,'batch_size':len(questions),'forward_input_widths':widths}
