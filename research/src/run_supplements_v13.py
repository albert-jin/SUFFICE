import sys,time,subprocess,fcntl
from common import *
def main():
    run=ROOT/'runs/inference_ablations_v13';lock=(run/'supplements.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not (run/'ANALYSIS_COMPLETE.json').exists():
        if (run/'status.json').exists() and json.loads((run/'status.json').read_text()).get('status')=='failed':raise RuntimeError('Core bank failed; inspect logs')
        dump(run/'supplements_status.json',{'status':'waiting_for_analysis','pid':os.getpid(),'new_training':False});time.sleep(15)
    for name in ['latency_inference_v13.py','recovery_diagnostic_v13.py']:
        while float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().splitlines()[0])<22000:time.sleep(5)
        dump(run/'supplements_status.json',{'status':'running','stage':name,'pid':os.getpid(),'new_training':False})
        with (run/(name+'.log')).open('a') as log:rc=subprocess.call([sys.executable,str(ROOT/'src'/name)],stdout=log,stderr=subprocess.STDOUT)
        if rc:dump(run/'supplements_status.json',{'status':'failed','stage':name,'returncode':rc});raise SystemExit(rc)
    dump(run/'supplements_status.json',{'status':'complete','new_training':False,'next':'Audit all results, make accurate scientific figures and final user package; disclose unavailable training ablations and image-resolution limitation.'})
if __name__=='__main__':main()
