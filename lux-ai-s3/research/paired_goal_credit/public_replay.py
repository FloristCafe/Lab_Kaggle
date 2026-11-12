"""Public-prefix verifier and offline tape: changed actions cannot reuse recorded futures."""
import copy
import json

def require(ok, detail):
    if not ok:
        raise RuntimeError(detail)

class Journal:
    def __init__(self, stream, reference):
        self.stream, self.reference, self.matched, self.count = stream, reference, 0, 0

    def put(self, value):
        encoded = json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n"
        if self.reference is not None:
            expected = self.reference.readline()
            require(bool(expected) and expected == encoded,
                    f"public replay mismatch at record {self.matched}: {value['kind']}")
            self.matched += 1
        self.stream.write(encoded)
        self.count += 1

    def fork(self):
        require(self.reference is not None and not self.reference.readline(),
                "committed prefix has unexpected tail")
        self.reference = None


class RecordedPublic:
    """Offline receipt replay only; cannot supply a counterfactual next observation."""
    allowed={"obs","env_cfg","opponent","official_reward","engine_terminal",
             "engine_truncated","task_complete","_pgc_audit"}

    def __init__(self,trace):
        self.rows=iter(copy.deepcopy([row for row in trace if row["kind"]=="public_ipc"]))
        self.consumed=0

    def request(self,**request):
        expected=next(self.rows,None)
        require(expected is not None and expected["request"]==request,"offline public action mismatch")
        response=expected["response"]
        require(set(response)<=self.allowed,"unregistered response fields")
        self.consumed+=1
        return copy.deepcopy({k:v for k,v in response.items() if k!="_pgc_audit"})

    def finish(self):
        require(next(self.rows,None) is None,"unconsumed public replay tail")

    def close(self):
        pass

def official_margin(response,side):
    require(side in (0,1),"invalid side")
    require(response.get("task_complete",False),"terminal record required")
    wins=response["obs"]["team_wins"]
    require(len(wins)==2 and sum(wins)==5,"official five-match counters invalid")
    value=wins[side]-wins[1-side]
    reward=response["official_reward"]
    require(len(reward)==2 and value==reward[0]-reward[1],"terminal reward direction mismatch")
    return value
