import sys,subprocess,fcntl
from common import *
def main():
    run=ROOT/'runs/inference_ablations_v13';run.mkdir(parents=True,exist_ok=True);lock=(run/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name in ['freeze_inference_ablations_v13.py','generate_inference_ablations_v13.py']:
        dump(run/'pipeline_status.json',{'status':'running','stage':name,'pid':os.getpid(),'training':False})
        with (run/(name+'.log')).open('a') as log:rc=subprocess.call([sys.executable,str(ROOT/'src'/name)],stdout=log,stderr=subprocess.STDOUT)
        if rc:dump(run/'pipeline_status.json',{'status':'failed','stage':name,'returncode':rc});raise SystemExit(rc)
    dump(run/'pipeline_status.json',{'status':'bank_complete_awaiting_analysis','training':False,'next':'Score dev, freeze dev-only baseline selections, then score screen and analyze cached inference ablations. Do not train or change primary policy.'})
if __name__=='__main__':main()
