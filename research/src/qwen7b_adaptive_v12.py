"""Inference-only transfer: frozen Qwen question router, Qwen7B TokenSkip generator.

No shared KV cache between models. The Qwen prefill is charged to every query.
The public 7B GSM LoRA is from the paper author, not its CSQA-trained adapter.
"""
import time, struct
import torch
from common import *

RUN=ROOT/'runs/qwen7b_adaptive_v12'
FREEZE=ROOT/'manifests/QWEN7B_ADAPTIVE_V12_FREEZE.json'
BASE=ROOT/'models/v8/qwen7b'
ADAPTER=ROOT/'models/v11/qwen7b_gsm_author'
COUNTS={'commonsenseqa':256}

def check_adapter():
    meta=json.loads((ADAPTER/'ASSETS_MANIFEST.json').read_text())
    for x in meta['files']:
        p=ADAPTER/x['name'];assert p.stat().st_size==x['bytes'] and digest(p)==x['sha256']
    cfg=json.loads((ADAPTER/'adapter_config.json').read_text())
    assert cfg['peft_type']=='LORA' and cfg['r']==8 and cfg['lora_alpha']==16
    assert cfg['bias']=='none' and not cfg['fan_in_fan_out']
    assert set(cfg['target_modules'])=={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}
    disabled=['use_dora','use_rslora','use_qalora','lora_bias','ensure_weight_tying','modules_to_save','trainable_token_indices','target_parameters','alora_invocation_tokens','arrow_config','corda_config','eva_config','exclude_modules','layer_replication','layers_pattern','layers_to_transform','loftq_config','rank_pattern','alpha_pattern','megatron_config']
    assert all(not cfg.get(k) for k in disabled), 'Unsupported active adapter variant'
    with (ADAPTER/'adapter_model.safetensors').open('rb') as file:
        head=json.loads(file.read(struct.unpack('<Q',file.read(8))[0]))
    head.pop('__metadata__',None);assert len(head)==392
    expected={}
    for layer in range(28):
        for module,(outdim,indim) in {'self_attn.q_proj':(3584,3584),'self_attn.k_proj':(512,3584),'self_attn.v_proj':(512,3584),'self_attn.o_proj':(3584,3584),'mlp.gate_proj':(18944,3584),'mlp.up_proj':(18944,3584),'mlp.down_proj':(3584,18944)}.items():
            stem=f'base_model.model.model.layers.{layer}.{module}'
            expected[stem+'.lora_A.weight']=[8,indim];expected[stem+'.lora_B.weight']=[outdim,8]
    assert set(head)==set(expected) and all(head[k]['shape']==s and head[k]['dtype']=='F32' for k,s in expected.items())
    a=json.loads((BASE/'tokenizer.json').read_text());b=json.loads((ADAPTER/'tokenizer.json').read_text())
    norm=lambda xs:[v if isinstance(v,list) else v.split(' ') for v in xs]
    assert a['model']['vocab']==b['model']['vocab'] and norm(a['model']['merges'])==norm(b['model']['merges'])
    assert {x['content']:x['id'] for x in a['added_tokens']}=={x['content']:x['id'] for x in b['added_tokens']}
    return {'status':'static_compatible','tensor_count':392,'vocab_merges_and_added_token_ids_equal':True,'generation_tokenizer':'official_author_adapter','adapter_manifest':meta,'merge':'W=float32(W)+2*(float32(B)@float32(A)); cast BF16; no fitting','training_ids_disclosed':False,'prompt_training_recipe_unverified':True}

def load_freeze():
    f=json.loads(FREEZE.read_text())
    assert all(digest(ROOT/p)==h for p,h in f['source_files'].items())
    assert all(digest(ROOT/p)==h for p,h in f['controller_files'].items())
    assert all(digest(ROOT/p)==h for p,h in f['input_files'].items())
    return f

def prompt(question,ratio,tok):
    return make_prompt(question,ratio)

class Qwen7BGenerator:
    def __init__(self):
        from transformers import AutoTokenizer,AutoModelForCausalLM
        from safetensors.torch import load_file
        torch.set_num_threads(4);t=time.perf_counter()
        self.tokenizer=AutoTokenizer.from_pretrained(ADAPTER,local_files_only=True)
        self.tokenizer.padding_side='left'
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
        assert self.tokenizer.eos_token_id==151645 and self.tokenizer.pad_token_id==151643
        model=AutoModelForCausalLM.from_pretrained(BASE,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',low_cpu_mem_usage=True).eval()
        assert model.config.hidden_size==3584
        weights=load_file(ADAPTER/'adapter_model.safetensors',device='cpu');used=set()
        # Standard rank-8 LoRA only; newer PEFT metadata is not executed.
        with torch.no_grad():
            for key in sorted(weights):
                if not key.endswith('.lora_A.weight'):continue
                bkey=key.replace('.lora_A.weight','.lora_B.weight');a=weights[key];b=weights[bkey]
                path=key.removeprefix('base_model.model.').removesuffix('.lora_A.weight');module=model.get_submodule(path)
                assert (b.shape[0],a.shape[1])==tuple(module.weight.shape)
                assert torch.isfinite(a).all() and torch.isfinite(b).all()
                module.weight.copy_((module.weight.float()+2.0*(b@a)).to(torch.bfloat16));used.update([key,bkey])
        assert used==set(weights) and len(used)==392
        del weights
        self.model=model.requires_grad_(False).cuda().eval();torch.cuda.synchronize();self.load_seconds=time.perf_counter()-t
    def generate(self,questions,ratios,cap):
        t=time.perf_counter();tok=self.tokenizer
        texts=[prompt(q,r,tok) for q,r in zip(questions,ratios)]
        enc=tok(texts,add_special_tokens=False,padding=True,return_tensors='pt').to('cuda')
        assert enc.input_ids.shape[1]+cap<=8192,'Do not silently truncate inputs'
        counts=enc.attention_mask.sum(1).tolist();width=enc.input_ids.shape[1]
        torch.cuda.synchronize();pre=time.perf_counter()
        with torch.inference_mode():
            out=self.model.generate(**enc,max_new_tokens=cap,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize();decode=time.perf_counter()-pre;records=[]
        for seq,ratio,n in zip(out[:,width:].tolist(),ratios,counts):
            stopped=tok.eos_token_id in seq
            if stopped:seq=seq[:seq.index(tok.eos_token_id)+1]
            records.append({'text':tok.decode(seq,skip_special_tokens=True),'token_ids':seq,'output_tokens':len(seq),'output_tokens_excluding_eos':len(seq)-int(stopped),'input_tokens':n,'ratio':ratio,'truncated':not stopped,'finish_reason':'stop' if stopped else 'length','generation_requests':1,'max_new_tokens':cap})
        return records,{'total_seconds':time.perf_counter()-t,'generator_prefill_decode_seconds':decode,'physical_input_width':width,'batch_size':len(records),'max_new_tokens':cap,'qwen3b_prefix_reused_by_qwen7b':False}
