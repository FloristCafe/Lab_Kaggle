"""Public-observation-only process boundary for external native agents."""
import json
import os
import select
import subprocess
from pathlib import Path

import numpy as np


class ExternalAgent:
    def __init__(self,name,side,config,seed):
        registry=Path(os.environ["LUX_OPPONENT_REGISTRY"])
        data=json.loads(registry.read_text())
        spec=data["agents"][name]
        if not spec.get("enabled") and os.environ.get("LUX_OPPONENT_ENGINEERING")!="1":
            raise RuntimeError("external agent has not passed admission")
        root=Path(__file__).resolve().parents[2]
        command=[spec["python"],"-B",str(root/"scripts/expert_worker.py"),
                 "--registry",str(registry),"--agent",name,"--side",str(side),"--seed",str(seed)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="",JAX_PLATFORMS="cpu",
                 OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",PYTHONHASHSEED="0",PYTHONDONTWRITEBYTECODE="1")
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                      stderr=None,text=True,bufsize=1,cwd=spec["root"],env=env)
        self.request({"op":"initialize","config":config},60)

    def request(self,payload,timeout):
        self.process.stdin.write(json.dumps(payload)+"\n")
        self.process.stdin.flush()
        if not select.select([self.process.stdout],[],[],timeout)[0]:
            self.close()
            raise TimeoutError("external agent action deadline")
        line=self.process.stdout.readline()
        if not line:raise RuntimeError(f"external opponent exited {self.process.poll()}")
        result=json.loads(line)
        if result.get("error"):raise RuntimeError(result["error"])
        return result

    def act(self,step,obs):
        answer=self.request({"op":"act","step":step,"obs":obs},10)
        action=np.asarray(answer["action"])
        if action.shape!=(16,3) or action.dtype.kind not in "iu":
            raise RuntimeError("invalid external action shape/type")
        if not np.isin(action[:,0],np.arange(6)).all():
            raise RuntimeError("invalid external action code")
        return action.astype(np.int32)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
