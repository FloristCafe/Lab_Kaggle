"""Hash-checked small evidence snapshots; no local dataset or worker dependency."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read_evidence(name):
    relative = "reports/evidence/" + name
    index = json.loads((ROOT / "reports/evidence/source_index.json").read_text())
    matches = [row for row in index if row["file"] == relative]
    if len(matches) != 1:
        raise RuntimeError("evidence identity missing or ambiguous: " + name)
    content = (ROOT / relative).read_bytes()
    if hashlib.sha256(content).hexdigest() != matches[0]["sha256"]:
        raise RuntimeError("evidence hash mismatch: " + name)
    return json.loads(content)

def new_output(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    return directory

def save_svg(figure,file):
    """Strip exporter-only line-end whitespace; leave historical evidence untouched."""
    figure.savefig(file,metadata={"Date":None})
    file.write_text("".join(line.rstrip()+"\n" for line in file.read_text().splitlines()))
