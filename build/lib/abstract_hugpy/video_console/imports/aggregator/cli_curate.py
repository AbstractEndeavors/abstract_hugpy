### Minimal CLI shim (optional)
# cli_curate.py
import json, sys
from pathlib import Path
from curation_utils import aggregate_and_curate

base = Path(sys.argv[1] if len(sys.argv)>1 else ".")
res = aggregate_and_curate(base)
out = base/"aggregated_metadata.json"
out.write_text(json.dumps(res, indent=2), encoding="utf-8")
print("Wrote", out)
print("Best clip:", res["best_clip"]["start"], res["best_clip"]["end"])
