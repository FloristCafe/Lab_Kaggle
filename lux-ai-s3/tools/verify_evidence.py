"""Independent release evidence verification, without opening the game engine."""
import argparse
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from training.contracts import read, sha, source_lock
from training.evaluate import verify_case


def verify(directory):
    frozen = read(root/"release_source_lock.json")
    if source_lock() != frozen:
        raise RuntimeError("release source mismatch")
    data = {}
    inputs = {}
    def load(relative):
        p = directory/relative
        inputs[str(relative)] = sha(p)
        return read(p)
    for name in "abc":
        smoke = Path(f"smoke_{name}_final")
        if load(smoke/"identity.json")["sources"] != frozen:
            raise RuntimeError("smoke ran against different sources")
        m = load(smoke/"batch_000000/metrics.json")
        receipt = load(smoke/"batch_000000/commit.json")
        if any(sha(directory/smoke/"batch_000000"/p) != v for p, v in receipt.items()):
            raise RuntimeError("smoke commit mismatch")
        game = load(smoke/"batch_000000/game_000000.json")
        if (m["primitive_steps"] != 112 or m["actor_delta_l2"] <= 0 or
            m["critic_delta_l2"] <= 0 or m["max_replay_error"] > 1e-4 or
            not game["trajectory"][100]["reset"] or not game["summary"]["truncated"]):
            raise RuntimeError("smoke engineering contract not met")
        restart = Path(f"resume_{name}_r02")
        recovery = load(restart/"result.json")
        if load(restart/"continuous/identity.json")["sources"] != frozen or load(restart/"split/identity.json")["sources"] != frozen:
            raise RuntimeError("recovery ran against different sources")
        if recovery["status"] != "PASS" or not all(all(x.values()) for x in recovery["details"].values()):
            raise RuntimeError("deterministic recovery mismatch")
        terminal = Path(f"terminal_{name}_final")
        end = load(terminal/"result.json")
        raw = load(terminal/"case.json")
        margin = verify_case(directory/terminal, raw["identity"], 15001)
        previous = [0,0]
        for x in raw["trajectory"]:
            change = [a-b for a,b in zip(x["team_wins"],previous,strict=True)]
            if sorted(change) != ([0,1] if x["reset"] else [0,0]):
                raise RuntimeError("submatch official win transition mismatch")
            if len(x["actions"]) != 16 or len(x["public_obs_sha256"]) != 64:
                raise RuntimeError("incomplete action or observation audit")
            previous = x["team_wins"]
        for uid in range(16):
            rows = sorted((o for o in raw["options"] if o["uid"] == uid), key=lambda o:o["start"])
            if any(a["end"] > b["start"] for a,b in zip(rows, rows[1:])):
                raise RuntimeError("overlapping UID option lifetimes")
            if any(o["duration"] != o["end"]-o["start"] or o["duration"] <= 0 for o in rows):
                raise RuntimeError("option duration mismatch")
        if (end["status"] != "PASS" or end["details"]["terminal_margin"] != margin or
            end["details"]["actor_delta_l2"] <= 0 or end["details"]["critic_delta_l2"] <= 0):
            raise RuntimeError("terminal replay or PPO check failed")
        data[name] = {"smoke": m, "full_game": end["details"], "resume_equal": True}
    return {"status":"PASS","formal_training_started":False,
            "source_lock_sha256":sha(root/"release_source_lock.json"), "models":data,
            "evidence_sha256":inputs}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("engineering",type=Path)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    result=verify(a.engineering)
    encoded=json.dumps(result,indent=2,allow_nan=False)+"\n"
    if a.output:
        with a.output.open("x") as stream:
            stream.write(encoded)
    else:
        print(encoded,end="")


if __name__=="__main__":
    main()
