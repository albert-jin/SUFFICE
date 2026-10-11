"""Build train/dev/calibration/reserve and sealed tests from fixed raw sources."""
import argparse,sys,json,hashlib,random,re
from pathlib import Path
from common import *
STEM=['abstract_algebra','astronomy','college_biology','college_chemistry','college_computer_science','college_mathematics','college_physics','computer_security','conceptual_physics','electrical_engineering','elementary_mathematics','high_school_biology','high_school_chemistry','high_school_computer_science','high_school_mathematics','high_school_physics','high_school_statistics','machine_learning']
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',required=True);ap.add_argument('--out',required=True);args=ap.parse_args();raw=Path(args.raw);out=Path(args.out)
    import pyarrow.parquet as pq
    def mc(row,ds,idx):
        if 'train' in row and isinstance(row['train'],dict):row=row['train']
        choices=row['choices'];letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        if isinstance(choices,dict):labs=choices['label'];txt=choices['text']
        else:txt=choices;labs=list(letters[:len(txt)])
        answer=row.get('answerKey',row.get('answer'))
        if isinstance(answer,(int,np.integer)):answer=letters[int(answer)]
        return {'id':f'{ds}:{idx}','dataset':ds,'question_stem':row['question'],'question':row['question']+'\n'+'\n'.join(f'{l}. {t}' for l,t in zip(labs,txt))+'\nChoose the correct option letter.','answer':answer,'task':'mcq','n_choices':len(txt),'subject':row.get('subject','auxiliary_unspecified'),'source_id':row.get('id',idx)}
    pools={};tests={};separate_dev={}
    for ds,folder in [('gsm8k','gsm8k'),('math500','math-500')]:
        def convert(rows,split):
            return [{'id':f'{ds}:{split}:{i}','dataset':ds,'question':x.get('question',x.get('problem')),'answer':x['answer'],'task':ds,'source_id':x.get('unique_id',x.get('id',i)),'level':x.get('level'),'subject':x.get('type','')} for i,x in enumerate(rows)]
        pools[ds]=convert(readl(raw/folder/'train.jsonl'),'train');tests[ds]=convert(readl(raw/folder/'test.jsonl'),'test')
    for ds,files in [('commonsenseqa',('csqa_train','csqa_test')),('mmlu_stem',('mmlu_auxiliary_train','mmlu_test'))]:
        train=pq.read_table(raw/(files[0]+'.parquet')).to_pylist();test=pq.read_table(raw/(files[1]+'.parquet')).to_pylist()
        if ds=='mmlu_stem':test=[x for x in test if x['subject'] in STEM]
        pools[ds]=[mc(x,ds,'train:'+str(i)) for i,x in enumerate(train)]
        tests[ds]=[mc(x,ds,'test:'+str(i)) for i,x in enumerate(test)]
    mdev=[]
    for split in ['dev','validation']:
        for i,x in enumerate(pq.read_table(raw/f'mmlu_{split}.parquet').to_pylist()):
            if x['subject'] in STEM:mdev.append(mc(x,'mmlu_stem',split+':'+str(i)))
    separate_dev['mmlu_stem']=mdev
    expected={'gsm8k':1319,'math500':500,'commonsenseqa':1221,'mmlu_stem':3018}
    assert {d:len(t) for d,t in tests.items()}==expected
    test_hashes={question_hash(x.get('question_stem',x['question'])) for rows in tests.values() for x in rows}
    used=set(test_hashes);stats={};splits={s:[] for s in ['train','dev','calibration','reserve']}
    for ds in DATASETS:
        rows=pools[ds];rng=random.Random(20261009);rng.shuffle(rows)
        clean=[];removed=0
        for x in rows:
            h=question_hash(x.get('question_stem',x['question']))
            if h in used:removed+=1;continue
            x['question_sha256']=h;used.add(h);clean.append(x)
        if ds in separate_dev:
            dr=[]
            for x in separate_dev[ds]:
                h=question_hash(x.get('question_stem',x['question']))
                # Separate dev supersedes auxiliary_train duplicates.
                if h in test_hashes:continue
                x['question_sha256']=h;dr.append(x)
            dh={x['question_sha256'] for x in dr};clean=[x for x in clean if x['question_sha256'] not in dh]
            rng.shuffle(dr);n=len(dr)//3
            groups={'train':clean[:1536],'dev':dr[:n],'calibration':dr[n:2*n],'reserve':dr[2*n:]}
        else:
            groups={'train':clean[:1536],'dev':clean[1536:1792],'calibration':clean[1792:2048],'reserve':clean[2048:2304]}
        for s,g in groups.items():
            for x in g:x['split']=s
            splits[s].extend(g)
        stats[ds]={'raw_train':len(rows),'removed_duplicates':removed,'selected':{s:len(g) for s,g in groups.items()},'test':len(tests[ds])}
        inputs=[{k:v for k,v in x.items() if k!='answer'} for x in tests[ds]]
        writel(out/'sealed_test'/f'{ds}_inputs.jsonl',inputs);writel(out/'sealed_test'/f'{ds}_targets.jsonl',[{'id':x['id'],'answer':x['answer']} for x in tests[ds]])
    # Deduplicate and audit across all selected train/dev/calibration/reserve splits.
    seen=set(test_hashes)
    for s in ['dev','calibration','reserve','train']:
        unique=[]
        for x in splits[s]:
            h=x['question_sha256']
            if h in seen:continue
            seen.add(h);unique.append(x)
        splits[s]=unique;writel(out/f'{s}.jsonl',unique)
    pilot=[]
    for s,n in [('train',96),('dev',64)]:
        for ds in DATASETS:pilot.extend([x for x in splits[s] if x['dataset']==ds][:n])
    writel(out/'pilot.jsonl',pilot)
    dump(out/'split_manifest.json',{'seed':20261009,'stats_before_final_dedup':stats,'counts':{s:{d:sum(x['dataset']==d for x in rows) for d in DATASETS} for s,rows in splits.items()},'pilot_n':len(pilot),'test_n':expected,'test_label_policy':'No final-test labels loaded by development scripts. Historical published test evidence already known; do not claim untouched benchmarks globally.','hashes':{str(p.relative_to(out)):digest(p) for p in out.rglob('*.jsonl')}})
    print(json.dumps({'split_counts':{s:len(x) for s,x in splits.items()},'pilot_n':len(pilot),'tests':expected}),flush=True)
if __name__=='__main__':main()
