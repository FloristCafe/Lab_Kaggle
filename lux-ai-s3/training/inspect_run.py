"""Inspect a committed training run without loading tensors."""
import argparse
import json
import time
from pathlib import Path

from training.contracts import committed, read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--watch", type=float, default=0)
    args = p.parse_args()
    seen = None
    while True:
        latest = committed(args.run)
        if latest != seen or not args.watch:
            data = read(latest/"metrics.json") if latest else {"games":0,"primitive_steps":0}
            print(json.dumps({"checkpoint": str(latest/"checkpoint.pt") if latest else None,
                              **data}), flush=True)
            seen = latest
        if not args.watch:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
