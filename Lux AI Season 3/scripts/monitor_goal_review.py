"""Read-only bounded-interval resource/progress accounting for independent runs."""
import argparse
import csv
import datetime
import json
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arms',nargs='+',required=True)
    p.add_argument('--output-dir',type=Path,default=ROOT/'experiments/goal_review/autonomous_progress')
    p.add_argument('--interval',type=int,default=30)
    args=p.parse_args()
    if not 1<=args.interval<=60:raise ValueError('bounded polling interval 1..60s')
    args.output_dir.mkdir(parents=True,exist_ok=True);last={}
    with (args.output_dir/'resource_samples.csv').open('a',newline='',buffering=1) as f:
        writer=csv.writer(f)
        if f.tell()==0:writer.writerow(['utc','temperature_c','power_w','gpu_util_percent','gpu_memory_mib','host_available_mib'])
        while True:
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(', ')
            memory=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
            available=int(memory['MemAvailable'].split()[0])/1024
            writer.writerow([datetime.datetime.now(datetime.timezone.utc).isoformat(),*gpu,available])
            states={}
            for arm in args.arms:
                folder=ROOT/'experiments/goal_review'/f'{arm}_seed42'
                summary=folder/'summary.json';log=folder/'train_log.csv'
                if summary.exists():
                    result=json.loads(summary.read_text());states[arm]={'status':result['status'],'epoch':result['stopped_epoch'],'best_epoch':result['best_epoch']}
                elif log.exists():
                    try:
                        rows=list(csv.DictReader(log.read_text().splitlines()));row=rows[-1]
                        states[arm]={'status':'RUNNING','epoch':int(row['epoch']),'macro_f1':float(row['macro_f1']),'joint':float(row['joint_accuracy'])}
                    except (IndexError,KeyError,ValueError):states[arm]={'status':'RUNNING','epoch':0}
                else:states[arm]={'status':'RUNNING','epoch':0}
            if states!=last:
                (args.output_dir/'latest.json').write_text(json.dumps({'arms':states,'gpu_memory_mib':gpu[-1],'host_available_mib':available},indent=2)+'\n')
                print(json.dumps(states),flush=True);last=states
            if all(s['status']=='FINISHED' for s in states.values()):break
            time.sleep(args.interval)


if __name__=='__main__':main()
