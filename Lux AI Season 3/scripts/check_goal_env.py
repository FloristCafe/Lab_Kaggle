"""Short real-engine contract check; never a strategy-strength evaluation."""

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.strategic.environment import GoalLevelEnv
from agents.strategic.experiment import save_json


def main():
    config = yaml.safe_load((ROOT / "configs/strategic_baseline.yaml").read_text())
    out = ROOT / "experiments/interface_contract"
    out.mkdir(exist_ok=True, parents=True)
    env = GoalLevelEnv(config, out / "env_worker.log")
    records = []
    try:
        for side in (0, 1):
            packet = env.reset(15001, side)
            assert not any(
                k in env.env_cfg
                for k in ("nebula_tile_drift_speed", "unit_sap_dropoff_factor")
            )
            assert packet["history"].shape == (1, 3, 18, 24, 24)
            for _ in range(4):
                f = env.frame
                goals = f["positions"][:, 0].clip(0, 23) * 24 + f["positions"][
                    :, 1
                ].clip(0, 23)
                packet, reward, terminal, info = env.step(np.zeros(16, np.int64), goals)
                assert 1 <= info["tau"] <= config["tactical"]["macro_horizon"]
                assert np.isfinite(reward)
                assert not terminal
            records.append(dict(side=side, summary=env.summary()))
            print(records[-1], flush=True)
        save_json(out / "summary.json", records)
    finally:
        env.close()


if __name__ == "__main__":
    main()
