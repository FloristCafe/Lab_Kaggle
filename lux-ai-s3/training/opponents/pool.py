"""Frozen admission registry and recoverable curriculum/PFSP sampling."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def registry_path():
    return Path(os.environ["LUX_OPPONENT_REGISTRY"])


def registry():
    data=json.loads(registry_path().read_text())
    for name,spec in data["agents"].items():
        if spec["enabled"]:
            if not spec.get("full_game_acceptance"):
                raise RuntimeError(f"opponent has no complete-game acceptance: {name}")
            query="import sys,torch,numpy,scipy,json;print(json.dumps(dict(python=sys.version.split()[0],torch=torch.__version__,numpy=numpy.__version__,scipy=scipy.__version__)))"
            runtime=json.loads(subprocess.check_output([spec["python"],"-c",query],text=True))
            if runtime!=spec["runtime"]:raise RuntimeError("external runtime version changed")
            if digest(spec["license_path"])!=spec["license_sha256"]:
                raise RuntimeError("external license identity changed")
            for p,h in {**spec["sha256"],**spec.get("wrapper_sha256",{})}.items():
                if digest(p)!=h:
                    raise RuntimeError(f"opponent asset drift: {name} {p}")
    return data


class Pool:
    def __init__(self, seed, entries):
        self.config=json.loads((Path(__file__).resolve().parents[1]/"configs/opponents/pool.json").read_text())
        self.ids=["v2"]+sorted(k for k,v in entries["agents"].items() if v["enabled"])
        if len(self.ids)<2:
            raise RuntimeError("multi-opponent training requires an accepted external agent")
        self.generator=np.random.default_rng(np.random.SeedSequence([seed,7183]))
        self.results={name:[0,0.] for name in self.ids}

    def weights(self):
        # Beta(1,1) prevents zero-visit opponents getting a degenerate estimate.
        alpha,beta=self.config["prior_alpha"],self.config["prior_beta"]
        failure=np.asarray([(1-(w+alpha)/(n+alpha+beta))**self.config["pfsp_power"]
                            for n,w in self.results.values()])
        pfsp=failure/failure.sum()
        epsilon=self.config["pfsp_exploration"]
        distribution=(1-epsilon)*pfsp+epsilon/len(self.ids)
        # An explicit baseline anchor protects early learning against an overly hard pool.
        anchor=self.config["v2_anchor"]
        distribution*=1-anchor
        distribution[self.ids.index("v2")]+=anchor
        return distribution/distribution.sum()

    def sample(self):
        return str(self.generator.choice(self.ids,p=self.weights()))

    def observe(self, opponent, margin):
        self.results[opponent][0]+=1
        self.results[opponent][1]+=1. if margin>0 else .5 if margin==0 else 0.

    def state_dict(self):
        return {"ids":self.ids,"results":self.results,"generator":self.generator.bit_generator.state}

    def load_state_dict(self,state):
        if state["ids"]!=self.ids:raise RuntimeError("opponent roster changed during resume")
        self.results=state["results"]
        self.generator.bit_generator.state=state["generator"]
