"""No game/model launch; immutable original implementation identities."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path("/home/issue/Kaggle/Lab_Kaggle/Lux AI Season 3")
HERE = Path(__file__).resolve().parent
PREP = ROOT / "outputs/mappo_v5_2/v2_goal_adapter_preparation_20261026_G01"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    lock = json.loads((PREP / "historical_lock.json").read_text())
    old = json.loads(
        (
            ROOT
            / "outputs/mappo_v5_2/pbrs_preparation_20261025_P01/historical_lock.json"
        ).read_text()
    )
    for table in (old["assets"], lock["files"]):
        for name, expected in table.items():
            if sha(ROOT / name) != expected:
                raise RuntimeError("protected source/artifact drift: " + name)
    for folder, expected in [
        (ROOT, lock["HEAD"]),
        (ROOT / "vendor/lux-design-s3", lock["official_commit"]),
    ]:
        if (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=folder, text=True
            ).strip()
            != expected
        ):
            raise RuntimeError("version drift")
    return {
        "status": "PASS",
        "protected_files": len(lock["files"]),
        "HEAD": lock["HEAD"],
        "official_commit": lock["official_commit"],
    }
