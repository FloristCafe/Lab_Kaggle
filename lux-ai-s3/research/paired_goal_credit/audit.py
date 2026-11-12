"""Optional read-only audit of archived PGC receipts and negative-control traces.

No worker imports, state cloning, training or environment creation.
This checks receipts and sham traces; formal branch trace bodies are not replayed.
"""
import argparse
import gzip
import hashlib
import json
from itertools import zip_longest
from pathlib import Path

from research.evidence import ROOT, read_evidence
from .streams import verify_fixed_node

def checked_read(file):
    if file.is_symlink() or not file.is_file():
        raise RuntimeError("ordinary archived file required: "+str(file))
    return json.loads(file.read_bytes())

def audit(raw):
    raw=Path(raw)
    protocol_path=ROOT/"reports/evidence/PGC01_protocol.json"
    protocol=hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    order=read_evidence("PGC01_execution_order.json")
    prefixes={};branches={};shams={};header_hashes={};actor_hashes=set()
    for case in order:
        folder=raw/f"case_{case['index']:03d}"
        receipt=checked_read(folder/"committed.json")
        if receipt["case"]!=case or receipt["protocol"]!=protocol:
            raise RuntimeError("receipt identity/protocol mismatch")
        relative=Path(receipt["result"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("unsafe result location")
        file=folder/relative
        result=checked_read(file)
        sha=hashlib.sha256(file.read_bytes()).hexdigest()
        if sha!=receipt["files"][str(relative)] or result["case"]!=case:
            raise RuntimeError("result receipt hash/identity mismatch")
        header_hashes[str(case["index"])]=sha
        actor_hashes.add(result["actor_sha256"])
        if case["kind"]=="prefix":
            prefixes[case["context"]]=result
            continue
        prefix=prefixes[case["context"]]
        verify_fixed_node(prefix["node"],result["node"])
        if result["matched_prefix_records"]!=prefix["records"]:
            raise RuntimeError("reported prefix record coverage mismatch")
        wins=result["official_wins"];side=case["side"]
        if not result["terminal"] or result["steps"]!=505 or sum(wins)!=5 or result["margin"]!=wins[side]-wins[1-side]:
            raise RuntimeError("official reward/termination mismatch")
        if case["kind"]=="branch":
            key=(case["context"],case["repeat"],case["arm"])
            if key in branches:raise RuntimeError("duplicate branch")
            branches[key]=result
        else:
            shams.setdefault(case["context"],[]).append(folder/relative.parent/"trace.jsonl.gz")
    if len(prefixes)!=48 or len(branches)!=192 or len(shams)!=4 or len(actor_hashes)!=1:
        raise RuntimeError("archived case coverage mismatch")
    checked_lines=0
    for files in shams.values():
        if len(files)!=2 or any(f.is_symlink() for f in files):
            raise RuntimeError("invalid negative-control pair")
        with gzip.open(files[0],"rb") as left,gzip.open(files[1],"rb") as right:
            for a,b in zip_longest(left,right):
                if a!=b:raise RuntimeError("negative-control trace divergence")
                checked_lines+=1
    saved=read_evidence("PGC01.json")
    for row in saved["rows"]:
        context=row["context"]
        rebuilt=[branches[(context,r,"A")]["margin"]-branches[(context,r,"B")]["margin"] for r in (0,1)]
        if row["deltas"]!=rebuilt:
            raise RuntimeError("independently reconstructed difference mismatch")
    return {"status":"PASS_WITH_SCOPE","case_receipts":len(order),"prefix_receipts":len(prefixes),
            "branch_results_recomputed":len(branches),"negative_control_pairs_byte_equal":len(shams),
            "negative_control_lines_compared":checked_lines,"actor_hashes":sorted(actor_hashes),
            "result_header_sha256":header_hashes,"protocol_sha256":protocol,
            "scope":"All result headers verified; four sham traces compared decompressed byte-for-byte. Formal branch trace bodies not fully audited or re-simulated.",
            "environment_starts":0,"training_runs":0}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():raise RuntimeError("audit output already exists")
    result=audit(args.raw_dir)
    with output.open("x") as stream:
        json.dump(result,stream,indent=2)
        stream.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k!="result_header_sha256"}))

if __name__=="__main__":
    main()
