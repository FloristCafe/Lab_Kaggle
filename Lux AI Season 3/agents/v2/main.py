"""Lux runner entry point for the v2 agent."""

import json
import os
import sys
from argparse import Namespace

from agent import Agent


agents = {}


def agent_fn(observation, configurations):
    raw = observation.obs
    if isinstance(raw, str):
        raw = json.loads(raw)
    player = observation.player
    if observation.step == 0 or player not in agents:
        agents[player] = Agent(player, configurations["env_cfg"])
    agent = agents[player]
    actions = agent.act(observation.step, raw,
                        observation.remainingOverageTime)
    trace_dir = os.environ.get("LUX_V2_TRACE_DIR")
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
        tag = os.environ.get("LUX_V2_TRACE_TAG", "run")
        path = os.path.join(trace_dir, f"{tag}_{player}.jsonl")
        with open(path, "a", encoding="utf-8") as trace:
            trace.write(json.dumps(agent.last_trace) + "\n")
    return {"action": actions.tolist()}


if __name__ == "__main__":
    env_cfg = None
    for line in sys.stdin:
        item = json.loads(line)
        if env_cfg is None:
            env_cfg = item["info"]["env_cfg"]
        observation = Namespace(**item)
        result = agent_fn(observation, {"env_cfg": env_cfg})
        print(json.dumps(result), flush=True)
