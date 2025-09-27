#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./fix_and_aggregate.sh /path/to/<video_id> [--domain https://example.com] [--title "My Title"] [--keywords "k1,k2,k3"] [--ocr] [--force]
#
# Requirements: bash, jq, python3. (Optional: tesseract-ocr if --ocr)

BASE_DIR="${1:-}"
shift || true

DOMAIN="https://typicallyoutliers.com"
TITLE=""
KEYWORDS=""
DO_OCR=0
FORCE=0

while (( "$#" )); do
  case "$1" in
    --domain)   DOMAIN="$2"; shift 2;;
    --title)    TITLE="$2"; shift 2;;
    --keywords) KEYWORDS="$2"; shift 2;;
    --ocr)      DO_OCR=1; shift;;
    --force)    FORCE=1; shift;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

if [[ -z "$BASE_DIR" || ! -d "$BASE_DIR" ]]; then
  echo "Base dir missing or not a directory: $BASE_DIR" >&2
  exit 1
fi

need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing dependency: $1" >&2; exit 1; }; }
need jq
need python3

json_merge() {
  # json_merge <file> <jq-object>
  local f="$1"; shift
  local obj="$*"
  mkdir -p "$(dirname "$f")"
  if [[ -f "$f" ]]; then
    tmp="$(mktemp)"
    jq -c "(. // {}) * ($obj)" "$f" > "$tmp" && mv "$tmp" "$f"
  else
    echo "$obj" | jq -c . > "$f"
  fi
}

# ---------- Detect files ----------
INFO_JSON="$BASE_DIR/info.json"
META_JSON="$BASE_DIR/metadata.json"
TOTAL_INFO_JSON="$BASE_DIR/total_info.json"
THUMBS_JSON="$BASE_DIR/thumbnails.json"
CAPTIONS_SRT="$BASE_DIR/captions.srt"
THUMBS_DIR="$BASE_DIR/thumbnails"

# video file preference: video.* else first mp4/mkv/webm
VIDEO_PATH=""
if compgen -G "$BASE_DIR/video.*" >/dev/null; then
  VIDEO_PATH="$(ls -1 "$BASE_DIR"/video.* | head -n1)"
else
  VIDEO_PATH="$(find "$BASE_DIR" -maxdepth 1 -type f \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.webm' \) | head -n1 || true)"
fi

# audio file preference: audio.wav else common audio
AUDIO_PATH=""
if [[ -f "$BASE_DIR/audio.wav" ]]; then
  AUDIO_PATH="$BASE_DIR/audio.wav"
else
  AUDIO_PATH="$(find "$BASE_DIR" -maxdepth 1 -type f \( -iname '*.wav' -o -iname '*.m4a' -o -iname '*.aac' -o -iname '*.mp3' \) | head -n1 || true)"
fi

# ---------- Build minimal info fallback (in-memory) ----------
BASENAME="$(basename "${VIDEO_PATH:-$BASE_DIR}")"
VIDEO_ID="$(basename "$BASE_DIR")"

# ---------- Ensure info.json existence (do not clobber) ----------
if [[ ! -f "$INFO_JSON" ]]; then
  json_merge "$INFO_JSON" "{
    \"id\": \"$VIDEO_ID\",
    \"video_id\": \"$VIDEO_ID\",
    \"title\": null,
    \"description\": null,
    \"duration\": null,
    \"webpage_url\": null,
    \"file_path\": $( [[ -n "$VIDEO_PATH" ]] && printf '%s' "\"$VIDEO_PATH\"" || echo null ),
    \"video_path\": $( [[ -n "$VIDEO_PATH" ]] && printf '%s' "\"$VIDEO_PATH\"" || echo null )
  }"
fi

# ---------- Ensure metadata.json existence & hydrate ----------
if [[ ! -f "$META_JSON" ]]; then
  json_merge "$META_JSON" "{}"
fi

# Read current fields to decide fallbacks
readarray -t fields < <(jq -r '[.title, .description] | @tsv' "$META_JSON" 2>/dev/null || echo "")
CURRENT_TITLE="$(jq -r '.title // empty' "$META_JSON" 2>/dev/null || true)"
CURRENT_DESC="$(jq -r '.description // empty' "$META_JSON" 2>/dev/null || true)"

# pull a snippet from whisper/captions for description if missing
DESC_FALLBACK="$CURRENT_DESC"
if [[ -z "$DESC_FALLBACK" ]]; then
  if [[ -f "$BASE_DIR/whisper.json" ]]; then
    DESC_FALLBACK="$(jq -r '(.segments[0].text // .text // "")[:300]' "$BASE_DIR/whisper.json")"
  fi
  if [[ -z "$DESC_FALLBACK" && -f "$CAPTIONS_SRT" ]]; then
    DESC_FALLBACK="$(head -n 40 "$CAPTIONS_SRT" | tr -s ' ' | tr '\n' ' ' | cut -c -300)"
  fi
fi

# title fallback order: --title > metadata.title > info.title > basename
INFO_TITLE="$(jq -r '.title // .fulltitle // empty' "$INFO_JSON" 2>/dev/null || true)"
TITLE_FINAL="${TITLE:-${CURRENT_TITLE:-${INFO_TITLE:-$BASENAME}}}"

# keywords array
KEY_ARR="[]"
if [[ -n "$KEYWORDS" ]]; then
  KEY_ARR="$(printf '%s' "$KEYWORDS" | awk -F',' '
    { printf("["); for(i=1;i<=NF;i++){gsub(/^[ \t]+|[ \t]+$/, "", $i); printf("%s\"%s\"", (i>1?",":""), $i)}; printf("]"); }')"
else
  # try to build from info.json tags/categories
  KEY_ARR="$(jq -c '[.keywords, .tags, .categories] | map(.) | map(select(.!=null)) | add // []' "$INFO_JSON" 2>/dev/null || echo '[]')"
fi

# pick a thumbnail
THUMB_PATH=null
if [[ -f "$THUMBS_JSON" ]]; then
  THUMB_PATH="$(jq -r '(.thumbnail_paths // .paths // .thumbnails // [])[0] // empty' "$THUMBS_JSON" 2>/dev/null || true)"
  [[ -n "$THUMB_PATH" ]] || THUMB_PATH=null
elif [[ -d "$THUMBS_DIR" ]]; then
  CAND="$(find "$THUMBS_DIR" -maxdepth 1 -type f -iname '*.jpg' | sort | head -n1 || true)"
  if [[ -n "$CAND" ]]; then THUMB_PATH="\"$CAND\""; else THUMB_PATH=null; fi
fi

# seodata.seo_data minimal block
json_merge "$META_JSON" "{
  \"title\": \"${TITLE_FINAL}\",
  \"description\": ${DESC_FALLBACK:+\"$DESC_FALLBACK\"}${DESC_FALLBACK:+"":-null},
  \"keywords\": $KEY_ARR,
  \"category\": (.category // .Category // \"General\"),
  \"seodata\": {
    \"seo_data\": {
      \"seo_title\": \"${TITLE_FINAL}\",
      \"seo_description\": ${DESC_FALLBACK:+\"$DESC_FALLBACK\"}${DESC_FALLBACK:+"":-null},
      \"seo_tags\": $KEY_ARR,
      \"thumbnail\": {\"file_path\": $THUMB_PATH},
      \"uploader\": {\"name\": (.uploader // \"typicallyoutliers\"), \"url\": \"$DOMAIN\"},
      \"publication_date\": (.publication_date // now | todate),
      \"canonical_url\": (.canonical_url // \"$DOMAIN\")
    }
  }
}"

# ---------- Patch total_info.json ----------
INFO_EXISTS=$([[ -f "$INFO_JSON" ]] && echo true || echo false)
META_EXISTS=$([[ -f "$META_JSON" ]] && echo true || echo false)
CAP_EXISTS=$([[ -f "$CAPTIONS_SRT" ]] && echo true || echo false)
THB_EXISTS=$([[ -f "$THUMBS_JSON" || -d "$THUMBS_DIR" ]] && echo true || echo false)

json_merge "$TOTAL_INFO_JSON" "{
  \"video_path\": $( [[ -n "$VIDEO_PATH" ]] && printf '%s' "\"$VIDEO_PATH\"" || echo null ),
  \"audio_path\": $( [[ -n "$AUDIO_PATH" ]] && printf '%s' "\"$AUDIO_PATH\"" || echo null ),
  \"info\": $INFO_EXISTS,
  \"metadata\": $META_EXISTS,
  \"captions\": $CAP_EXISTS,
  \"thumbnails\": $THB_EXISTS,
  \"total\": true
}"

# ---------- Optional: OCR thumbnails to populate thumbnail_texts ----------
if [[ $DO_OCR -eq 1 ]]; then
  if ! command -v tesseract >/dev/null 2>&1; then
    echo "[WARN] --ocr requested but tesseract not found; skipping OCR." >&2
  else
    python3 - <<'PY' "$THUMBS_DIR" "$THUMBS_JSON"
import sys, os, json, re
from glob import glob
from pathlib import Path
try:
    import pytesseract
    from PIL import Image
except Exception:
    print("[WARN] OCR deps missing in Python; skipping.")
    sys.exit(0)

thumbs_dir = Path(sys.argv[1])
thumbs_json = Path(sys.argv[2])
paths = []
if thumbs_dir.exists():
    paths = sorted(glob(str(thumbs_dir / "*.jpg")))
data = {}
if thumbs_json.exists():
    try: data = json.load(open(thumbs_json, "r"))
    except Exception: data = {}
texts = []
for p in paths:
    try:
        txt = pytesseract.image_to_string(Image.open(p))
        txt = re.sub(r"\s+", " ", txt).strip()
    except Exception:
        txt = ""
    texts.append({"path": p, "text": txt})
data["thumbnail_paths"] = data.get("thumbnail_paths") or paths
data["thumbnail_texts"] = texts
json.dump(data, open(thumbs_json, "w"), indent=2)
print(f"[OK] OCR thumbnail_texts -> {thumbs_json}")
PY
  fi
fi

# ---------- Run aggregator ----------
echo "[INFO] Running aggregator over $BASE_DIR ..."
python3 - <<'PY' "$BASE_DIR"
import sys, json
base = sys.argv[1]
# try multiple import locations to match your tree
mods = [
    "abstract_hugpy.hugpy_console.imports.aggregator.curation",
    "hugpy_console.imports.aggregator.curation",
    "curation_utils",  # in case it’s on PYTHONPATH as a local module
]
for m in mods:
    try:
        mod = __import__(m, fromlist=['aggregate_from_base_dir'])
        res = mod.aggregate_from_base_dir(base)
        print(json.dumps(res, indent=2))
        break
    except Exception as e:
        last = e
else:
    raise last
PY

echo "[DONE] Patched inputs and aggregated."
