CLIP_START=$(jq -r '.best_clip.start' "$JSON")
CLIP_END=$(jq -r '.best_clip.end' "$JSON")
DUR=$(python3 - <<PY
import sys,json
res=json.load(open(sys.argv[1]))
print(res["best_clip"]["end"]-res["best_clip"]["start"])
PY
"$JSON")
INPUT=$(jq -r '.video_path' "$JSON")
OUT="${INPUT%.*}_best.mp4"

ffmpeg -y -ss "$CLIP_START" -i "$INPUT" -t "$DUR" -c:v libx264 -c:a aac "$OUT"
echo "Wrote $OUT"