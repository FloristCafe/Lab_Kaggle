"""Create an explicit path-only local registry; never mutate historical manifests."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def prepare(source,output):
    raw=json.loads(source.read_text())
    protocol=source.parent/"expert_training_protocol.json"
    if protocol.is_file() and json.loads(protocol.read_text())["opponents_registry_sha256"]!=sha(source):
        raise RuntimeError("historical registry hash changed")
    old_root="/home/issue/Kaggle/lux-ai-s3"
    def relocated(value):
        path=Path(value)
        if path==Path(old_root) or path.is_relative_to(old_root):
            return str(ROOT/path.relative_to(old_root))
        return value
    for spec in raw["agents"].values():
        for field in ("root","python","license_path"):
            if field in spec:spec[field]=relocated(spec[field])
        for field in ("sha256","wrapper_sha256"):
            if field in spec:spec[field]={relocated(path):digest for path,digest in spec[field].items()}
        for path,digest in {**spec.get("sha256",{}),**spec.get("wrapper_sha256",{})}.items():
            if sha(path)!=digest:raise RuntimeError("relocated asset bytes differ: "+path)
    raw["path_only_relocation"]={"original_registry_sha256":sha(source),"old_code_root":old_root,
                                  "new_code_root":str(ROOT),"hash_values_changed":False}
    text=json.dumps(raw,indent=2)+"\n"
    if output.exists():
        if output.read_text()!=text:raise RuntimeError("existing derived registry differs; use a new path")
    else:
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open("x") as f:f.write(text)
    os.environ["LUX_OPPONENT_REGISTRY"]=str(output)
    from training.opponents.pool import registry
    accepted=registry()
    print(json.dumps({"status":"PASS","registry":str(output),"sha256":sha(output),
                      "opponents":[k for k,v in accepted["agents"].items() if v["enabled"]],
                      "historical_registry_unchanged":True,"environment_starts":0},indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();prepare(args.source,args.output)
