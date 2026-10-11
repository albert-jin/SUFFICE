"""Shared immutable definitions for paired-retention experiments."""
import hashlib,json,re,os
from pathlib import Path
import numpy as np
ROOT=Path(os.environ.get('AT_ROOT',str(Path(__file__).resolve().parents[3] / 'SUFFICE-WORKSPACE')))
RATIOS=(.3,.4,.5,.6,.7,.8,.9,1.0)
DATASETS=('gsm8k','math500','commonsenseqa','mmlu_stem')
SEEDS=(17,43,79)
def readl(path):
    with Path(path).open(encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]
def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8');temp.replace(path)
def writel(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def question_hash(text):return hashlib.sha256(re.sub(r'\s+',' ',text).strip().lower().encode()).hexdigest()
def make_prompt(question,ratio):
    suffix=f'<|eot_id|>{ratio:.1f}<|eot_id|>' if ratio<1 else ''
    return '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{}.\n'+question+suffix+'<|im_end|>\n<|im_start|>assistant\n'
def feature_prompt(question):
    return '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{}.\n'+question
def strip_final(text):
    """Remove explicit answer block and trailing conclusion without using gold."""
    cuts=[m.start() for m in re.finditer(r'\\boxed\s*\{|\\fbox\s*\{|(?im:^\s*(?:final answer|answer\s*:|therefore\b|thus\b|hence\b))',text)]
    if not cuts:return text.strip(),False
    cut=min(cuts)
    # A boxed expression on its own line is removed with its line prefix.
    line_start=text.rfind('\n',0,cut)+1
    if cut-line_start<70:cut=line_start
    return text[:cut].rstrip(),True
def recovery_prompt(question,cot):
    return '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nAnswer the question using the supplied partial reasoning when it is useful. Return only the final answer in \\boxed{}, without further explanation.\nQuestion:\n'+question+'\nPartial reasoning:\n'+(cot or '[No reasoning supplied]')+'<|im_end|>\n<|im_start|>assistant\n'
def paired_targets(y):
    y=np.asarray(y,dtype=np.int64)
    assert y.ndim==2 and np.isin(y,[0,1]).all()
    return 2*y[:,-1,None]+y
def select_actions(outcome,cost,threshold=0,uncertainty=None,beta=0):
    """Label-free action selection; full-retention is an exact fallback."""
    p=np.asarray(outcome);cost=np.asarray(cost)
    assert p.shape[:-1]==cost.shape and p.shape[-1]==4
    loss=p[...,2]-p[...,1]
    if uncertainty is not None:loss=loss+beta*np.asarray(uncertainty)
    loss=loss.copy();loss[:,-1]=0
    feasible=loss<=threshold;feasible[:,-1]=True
    return np.argmin(np.where(feasible,cost,np.inf),axis=1)

