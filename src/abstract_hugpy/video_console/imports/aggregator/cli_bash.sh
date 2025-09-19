#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${1:-/mnt/24T/hugging_face/videos/SOME_ID}"
JSON="$BASE_DIR/aggregated_metadata.json"
PY="python3 - <<'PY'"
$PY
from curation_utils import aggregate_and_curate
import json, sys
res = aggregate_and_curate(sys.argv[1] if len(sys.argv)>1 else ".")
open(f"{sys.argv[1]}/aggregated_metadata.json","w").write(json.dumps(res, indent=2))
print(res["best_clip"]["start"], res["best_clip"]["end"])
PY
