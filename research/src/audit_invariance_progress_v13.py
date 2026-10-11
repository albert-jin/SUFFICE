"""Compare already generated same-query/same-r outputs, without labels."""
from common import *
run=ROOT/'runs/inference_ablations_v13';groups={};records=0
for file in sorted((run/'parts').glob('*.json')):
    p=json.loads(file.read_text());key='_'.join(p['key'].split('_')[:2]);groups.setdefault(key,{})[p['condition']]=p;records+=len(p['records'])
checked=0;mismatches=[]
for key,g in groups.items():
    if 'adaptive_relative' not in g:continue
    for r in g['adaptive_relative']['records']:
        condition=f"fixed_{r['ratio']:.1f}"
        if condition not in g:continue
        fixed={x['id']:x for x in g[condition]['records']}[r['id']];checked+=1
        if fixed['token_ids']!=r['token_ids']:mismatches.append({'id':r['id'],'batch':key,'ratio':r['ratio'],'fixed_tokens':fixed['output_tokens'],'adaptive_tokens':r['output_tokens']})
report={'status':'provisional_same_ratio_identity_audit','records_generated':records,'questions_checked':checked,'token_sequence_mismatches':len(mismatches),'mismatches':mismatches,'labels_read':False,'complete':(run/'BANK_COMPLETE.json').exists()}
dump(run/'NUMERICAL_INVARIANCE_PROGRESS.json',report);print(json.dumps(report),flush=True)
