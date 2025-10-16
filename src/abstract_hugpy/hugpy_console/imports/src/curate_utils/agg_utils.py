import os, json
from pathlib import Path
from abstract_utilities import *
def safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def safe_load_text(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def classify_category(keywords, title="", description=""):
    kws = [k.lower() for k in keywords]
    text = " ".join(kws + [title.lower(), description.lower()])

    if any(x in text for x in ["comedy", "funny", "skit", "humor"]):
        return "Comedy"
    if any(x in text for x in ["music", "song", "album", "concert"]):
        return "Music"
    if any(x in text for x in ["news", "politic", "debate", "report"]):
        return "News & Politics"
    if any(x in text for x in ["education", "tutorial", "lesson", "howto"]):
        return "Education"
    if any(x in text for x in ["game", "gaming", "esport", "playthrough"]):
        return "Gaming"
    if any(x in text for x in ["sports", "match", "tournament"]):
        return "Sports"
    if any(x in text for x in ["tech", "review", "unboxing", "product"]):
        return "Science & Technology"
    if any(x in text for x in ["travel", "tour", "journey"]):
        return "Travel"
    if any(x in text for x in ["food", "cook", "recipe", "restaurant"]):
        return "Food"
    if any(x in text for x in ["film", "movie", "animation", "cinema"]):
        return "Film & Animation"

    return "Entertainment"

def aggregate_metadata(base_dir):
    # === Load sources ===
    info         = safe_load_json(os.path.join(base_dir, "video_info.json"))
    metadata     = safe_load_json(os.path.join(base_dir, "video_metadata.json"))
    whisper      = safe_load_json(os.path.join(base_dir, "whisper_result.json"))
    total_info   = safe_load_json(os.path.join(base_dir, "total_info.json"))
    thumbnails   = safe_load_json(os.path.join(base_dir, "thumbnails.json"))
    captions_txt = safe_load_text(os.path.join(base_dir, "captions.srt"))

    # === Collect raw fields ===
    title        = metadata.get("title") or info.get("title")
    description  = metadata.get("description") or metadata.get("summary") or info.get("description")
    keywords     = set()
    for src in (metadata, info):
        for k in ["keywords","tags","categories"]:
            if src.get(k):
                if isinstance(src[k], list):
                    keywords.update(src[k])
                else:
                    keywords.add(str(src[k]))

    # === Transcript / Captions ===
    transcript = []
    if whisper.get("text"):
        transcript.append(whisper["text"])
    if captions_txt:
        transcript.append(captions_txt)
    transcript_text = "\n".join(transcript).strip()

    # === Thumbnails ===
    thumb_candidates = []
    if metadata.get("thumbnail_url"):
        thumb_candidates.append(metadata["thumbnail_url"])
    if metadata.get("seodata",{}).get("seo_data",{}).get("thumbnail",{}).get("file_path"):
        thumb_candidates.append(metadata["seodata"]["seo_data"]["thumbnail"]["file_path"])
    if thumbnails.get("thumbnail_paths"):
        thumb_candidates.extend(thumbnails["thumbnail_paths"])

    # === Duration ===
    duration = (
        metadata.get("duration_formatted")
        or metadata.get("duration_seconds")
        or info.get("duration")
    )

    # === Category ===
    category = (
        metadata.get("category")
        or classify_category(keywords, title=title or "", description=description or "")
    )

    # === Build unified object ===
    aggregated = {
        "id": info.get("id"),
        "title": title,
        "description": description,
        "keywords": sorted(list(keywords)),
        "category": category,
        "url": info.get("webpage_url") or metadata.get("canonical_url"),
        "duration": duration,
        "uploader": (
            metadata.get("seodata",{}).get("seo_data",{}).get("uploader",{}).get("name")
            or info.get("uploader")
        ),
        "publication_date": metadata.get("seodata",{}).get("seo_data",{}).get("publication_date"),
        "video_path": total_info.get("video_path"),
        "audio_path": total_info.get("audio_path"),
        "transcript": transcript_text,
        "thumbnails": thumb_candidates,
        "schema_markup": metadata.get("seodata",{}).get("seo_data",{}).get("schema_markup"),
        "social_metadata": metadata.get("seodata",{}).get("seo_data",{}).get("social_metadata"),
        "video_metadata": metadata.get("seodata",{}).get("seo_data",{}).get("video_metadata"),
        "info_flags": total_info,
    }
    file_path = os.path.join(base_dir,"aggregated_metadata.json")
    data = json.dumps(aggregated, indent=2)
    safe_dump_to_json(data=data,file_path=file_path)
    return file_path,aggregated
def score_segments(segments, keywords):
    scored = []
    for seg in segments:
        text = seg["text"].lower()
        score = 0
        # Keyword hits
        score += sum(1 for k in keywords if k in text)
        # Sentiment/emphasis heuristics
        if "!" in text or "oh my god" in text: score += 2
        if "let's go" in text or "we can" in text: score += 2
        # Length penalty/bonus
        duration = seg["end"] - seg["start"]
        if 10 <= duration <= 45:
            score += 1
        scored.append((score, seg))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored

    



