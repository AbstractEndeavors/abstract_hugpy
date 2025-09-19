# Aggregator & curator for your uploaded bundle in /mnt/data
# - Merges video_info.json, video_metadata.json, whisper_result.json, captions.srt, thumbnails.json
# - Derives best clip window (deterministic), best thumbnails, hashtags, and verbose SEO metadata
# - Writes /mnt/data/aggregated_metadata.json, /mnt/data/best_clip.txt, /mnt/data/hashtags.txt

import os, json, re, math, string, statistics
from pathlib import Path



def safe_load_json(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def safe_load_text(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def parse_srt(text):
    # Very small SRT parser -> list of {start, end, text}
    out = []
    if not text.strip():
        return out
    blocks = re.split(r"\n\s*\n", text.strip())
    ts = lambda t: (
        int(t.split(":")[0]) * 3600
        + int(t.split(":")[1]) * 60
        + float(t.split(":")[2].replace(",", "."))
    )
    for b in blocks:
        lines = [x.strip() for x in b.splitlines() if x.strip()]
        if len(lines) >= 2:
            # find timestamp line
            m = re.search(r"(\d+:\d+:\d+[,\.]\d+)\s*-->\s*(\d+:\d+:\d+[,\.]\d+)", " ".join(lines))
            if not m:
                continue
            start, end = ts(m.group(1)), ts(m.group(2))
            content = " ".join([ln for ln in lines if "-->" not in ln and not ln.isdigit()])
            out.append({"start": start, "end": end, "text": content})
    return out

# Token helpers
PUNCT = set(string.punctuation)
def tokenize(s):
    s = s.lower()
    s = re.sub(r"[^\w\s'!?]", " ", s)
    return [t.strip("'") for t in s.split() if t.strip("'")]

def tf_idf_scores(segments):
    # Build TF and IDF across segments
    docs = [tokenize(seg["text"]) for seg in segments]
    N = len(docs) or 1
    df = {}
    for doc in docs:
        for w in set(doc):
            df[w] = df.get(w, 0) + 1
    idf = {w: math.log((N + 1) / (df[w] + 1)) + 1 for w in df}
    tfidf_per_seg = []
    for doc in docs:
        tf = {}
        for w in doc:
            tf[w] = tf.get(w, 0) + 1
        length = len(doc) or 1
        tfidf = {w: (tf[w] / length) * idf.get(w, 1.0) for w in tf}
        tfidf_per_seg.append(tfidf)
    return tfidf_per_seg, idf

def score_segments_basic(segments, keywords):
    """Score individual segments using keyword hits, tf-idf salience, and emphasis tokens."""
    tfidf_per_seg, idf = tf_idf_scores(segments)
    kw = [k.lower() for k in keywords]
    emph = {"!", "?", "wow", "omg", "incredible", "amazing", "shocking", "hilarious"}
    scored = []
    for i, seg in enumerate(segments):
        text = seg.get("text", "") or ""
        toks = tokenize(text)
        # keyword overlap
        kw_score = sum(1 for t in toks for k in kw if k and k in t)
        # tf-idf salience: top-k words
        sal = sum(sorted(tfidf_per_seg[i].values(), reverse=True)[:5])  # top-5 tf-idf
        # emphasis
        emph_score = sum(1 for t in toks if t in emph) + text.count("!") * 0.5 + text.count("?") * 0.25
        # duration goodness
        dur = max(0.01, float(seg["end"] - seg["start"]))
        dur_bonus = 0.6 if 6 <= dur <= 30 else (0.2 if 3 <= dur <= 45 else -0.2)
        score = kw_score * 1.2 + sal * 2.0 + emph_score * 1.0 + dur_bonus
        scored.append((score, seg))
    return scored

def sliding_windows_from_segments(segments, min_len=12.0, max_len=45.0, step=1.0):
    """Generate windows by combining consecutive segments to fall within [min_len, max_len]."""
    windows = []
    n = len(segments)
    starts = [s["start"] for s in segments]
    for i in range(n):
        start = segments[i]["start"]
        end = start
        j = i
        text_parts = []
        while j < n and end - start < max_len:
            end = segments[j]["end"]
            text_parts.append(segments[j]["text"])
            if end - start >= min_len:
                windows.append({
                    "start": start,
                    "end": end,
                    "text": " ".join(text_parts)
                })
                break
            j += 1
    # Also sample fixed-step windows using transcript bounds
    if segments:
        total_start = segments[0]["start"]
        total_end = segments[-1]["end"]
        t = total_start
        while t + min_len <= total_end:
            end = min(t + max_len, total_end)
            windows.append({"start": t, "end": end, "text": ""})
            t += step
    # Deduplicate by (start,end)
    seen = set()
    uniq = []
    for w in windows:
        key = (round(w["start"], 2), round(w["end"], 2))
        if key not in seen and w["end"] > w["start"]:
            uniq.append(w)
            seen.add(key)
    return uniq

def score_windows(windows, keywords, idf_hint=None):
    kw = [k.lower() for k in keywords]
    out = []
    for w in windows:
        toks = tokenize(w["text"] or "")
        # keyword coverage
        coverage = len({k for k in kw if any(k in t for t in toks)}) / (len(kw) or 1)
        # length sweet spot centered around 24s
        dur = w["end"] - w["start"]
        sweet = math.exp(-((dur - 24.0) ** 2) / (2 * 7.0 ** 2))  # gaussian around 24s
        # novelty = sum of idf of unique words (if provided)
        novelty = 0.0
        if idf_hint and toks:
            uniq = set(toks)
            novelty = sum(idf_hint.get(t, 1.0) for t in uniq) / len(uniq)
        score = 2.2 * coverage + 1.8 * sweet + 1.0 * novelty
        out.append((score, w))
    out.sort(key=lambda x: x[0], reverse=True)
    return out

def pick_best_clip(segments, keywords):
    if not segments:
        return None, []
    # per-seg scoring for idf hint & quick top spots
    seg_scored = score_segments_basic(segments, keywords)
    seg_scored.sort(key=lambda x: x[0], reverse=True)
    tfidf_per_seg, idf = tf_idf_scores(segments)
    # windows
    windows = sliding_windows_from_segments(segments, 12.0, 45.0, step=4.0)
    win_scored = score_windows(windows, keywords, idf_hint=idf)
    best = win_scored[0][1] if win_scored else {"start": segments[0]["start"], "end": segments[0]["end"], "text": segments[0]["text"]}
    # top 5 candidates for visibility
    top5 = [{"score": float(s), **w} for s, w in win_scored[:5]]
    return best, top5

def calc_sharpness(img_path):
    try:
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            return 0.0
        return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    except Exception:
        return 0.0

def pick_best_thumbnails(thumbnail_paths, top_k=3):
    scored = []
    for p in thumbnail_paths:
        if not os.path.exists(p):
            continue
        s = calc_sharpness(p)
        # simple colorfulness metric
        try:
            from PIL import Image
            import numpy as np
            arr = np.array(Image.open(p).convert("RGB"))
            rg = (arr[:, :, 0] - arr[:, :, 1]).astype(float)
            yb = (0.5 * (arr[:, :, 0] + arr[:, :, 1]) - arr[:, :, 2]).astype(float)
            colorfulness = float(np.std(rg) + np.std(yb))
        except Exception:
            colorfulness = 0.0
        score = s * 0.7 + colorfulness * 0.3
        scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_k]]

def make_hashtags(keywords, clip_text, title=None, limit=12):
    toks = tokenize(clip_text or "") + tokenize(title or "")
    # top frequent words not in stoplist
    stop = {"the","and","to","of","a","in","is","it","that","for","on","with","as","this","at","be","are","was","were","an"}
    freq = {}
    for t in toks:
        if t in stop or len(t) < 3 or t.isdigit():
            continue
        freq[t] = freq.get(t, 0) + 1
    top_toks = [w for w,_ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:8]]
    base = []
    for k in keywords:
        k = re.sub(r"[^A-Za-z0-9]+", "", k)
        if len(k) >= 3:
            base.append(k)
    for t in top_toks:
        base.append(t)
    # dedupe & format
    seen = set()
    tags = []
    for b in base:
        b = b[:40]
        if b.lower() not in seen:
            tags.append("#" + b[0].upper() + b[1:])
            seen.add(b.lower())
        if len(tags) >= limit:
            break
    return tags

def aggregate_and_curate(base_dir: Path):
    info         = safe_load_json(base_dir/"video_info.json")
    metadata     = safe_load_json(base_dir/"video_metadata.json")
    whisper      = safe_load_json(base_dir/"whisper_result.json")
    total_info   = safe_load_json(base_dir/"total_info.json")
    thumbs       = safe_load_json(base_dir/"thumbnails.json")
    srt_text     = safe_load_text(base_dir/"captions.srt")

    # keywords union
    keywords = set()
    for src in (metadata, info):
        for k in ("keywords", "tags", "categories"):
            v = src.get(k)
            if isinstance(v, list):
                keywords.update([str(x).strip().lower() for x in v])
            elif v:
                keywords.add(str(v).strip().lower())

    # transcript segments
    segs = []
    if isinstance(whisper.get("segments"), list):
        for s in whisper["segments"]:
            st, en = float(s.get("start", 0)), float(s.get("end", 0))
            txt = str(s.get("text", "")).strip()
            if en > st:
                segs.append({"start": st, "end": en, "text": txt})
    segs.extend(parse_srt(srt_text))

    segs.sort(key=lambda x: (x["start"], x["end"]))

    # Best clip selection
    best_clip, top5 = pick_best_clip(segs, list(keywords))

    # Thumbnail candidates
    thumb_candidates = []
    if metadata.get("thumbnail_url"):
        thumb_candidates.append(metadata["thumbnail_url"])
    if isinstance(thumbs.get("thumbnail_paths"), list):
        thumb_candidates.extend(thumbs["thumbnail_paths"])
    # include uploaded thumb.jpg if present
    if (base_dir/"thumb.jpg").exists():
        thumb_candidates.append(str(base_dir/"thumb.jpg"))

    best_thumbs = pick_best_thumbnails(thumb_candidates, top_k=3)

    # Title & description
    title = metadata.get("title") or info.get("title")
    description = (
        metadata.get("description")
        or metadata.get("summary")
        or (best_clip.get("text")[:300] + "...") if best_clip and best_clip.get("text") else None
    )

    # Category
    def classify_category(keywords, title="", description=""):
        kws = [k.lower() for k in keywords]
        text = " ".join(kws + [title.lower(), description.lower() if description else ""])
        rules = [
            ("Comedy", ["comedy","funny","skit","humor"]),
            ("Music", ["music","song","album","concert"]),
            ("News & Politics", ["news","politic","report","debate"]),
            ("Education", ["education","tutorial","lesson","howto"]),
            ("Gaming", ["game","gaming","esport","playthrough"]),
            ("Sports", ["sports","match","tournament","league"]),
            ("Science & Technology", ["tech","review","unboxing","product","ai","software","hardware"]),
            ("Travel", ["travel","tour","journey","trip"]),
            ("Food", ["food","cook","recipe","kitchen","restaurant"]),
            ("Film & Animation", ["film","movie","animation","cinema","sketch"]),
        ]
        for label, needles in rules:
            if any(n in text for n in needles):
                return label
        return "Entertainment"

    category = metadata.get("category") or classify_category(list(keywords), title or "", description or "")

    # Hashtags
    hashtags = make_hashtags(sorted(list(keywords)), best_clip.get("text",""), title)

    # Unified object
    aggregated = {
        "id": info.get("id"),
        "title": title,
        "description": description,
        "keywords": sorted(list(keywords)),
        "category": category,
        "canonical_url": metadata.get("seodata",{}).get("seo_data",{}).get("canonical_url") or info.get("webpage_url"),
        "duration": metadata.get("seodata",{}).get("seo_data",{}).get("duration_formatted") or info.get("duration"),
        "uploader": metadata.get("seodata",{}).get("seo_data",{}).get("uploader",{}),
        "publication_date": metadata.get("seodata",{}).get("seo_data",{}).get("publication_date"),
        "video_path": total_info.get("video_path") or info.get("file_path"),
        "audio_path": total_info.get("audio_path"),
        "transcript_excerpt": (best_clip.get("text","")[:500] + ("..." if len(best_clip.get("text",""))>500 else "")) if best_clip else "",
        "thumbnails_ranked": best_thumbs,
        "best_clip": best_clip,
        "candidate_clips": top5,
        "hashtags": hashtags,
        "schema_markup": metadata.get("seodata",{}).get("seo_data",{}).get("schema_markup"),
        "social_metadata": metadata.get("seodata",{}).get("seo_data",{}).get("social_metadata"),
        "source_flags": safe_load_json(base_dir/"total_info.json"),
    }

    # write outputs
    out_json = base_dir/"aggregated_metadata.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)
    # helper files
    with open(base_dir/"best_clip.txt", "w", encoding="utf-8") as f:
        if best_clip:
            f.write(f"{best_clip['start']},{best_clip['end']}\n")
    with open(base_dir/"hashtags.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(hashtags))

    return str(out_json), aggregated
def aggregate_from_base_dir(directory):
    BASE = Path(directory)
    out_path, aggregated = aggregate_and_curate(BASE)
    
