from .data_utils import *
def aggregate_key_maps(self, video_url=None, video_id=None) -> dict:
    """
    SEO-driven aggregator for video metadata.
    Priority: video_info.json → whisper_result.json → video_metadata.json

    Features:
    - Rolling merge with priority tiers.
    - Keyword dedupe + normalization.
    - Continuity fallbacks (title from keywords, desc from title).
    - Refinement with BigBird + GPT2 generator.
    - Heatmap highlights injection into description.
    - Thumbnail picked from top heatmap peaks.
    - Hashtag generation from keywords.
    - Category classification from keyword clusters.
    - Chapter generation from heatmap peaks + transcript.
    """

    data = self.get_data(video_url=video_url, video_id=video_id)
    files_priority = self.key_maps["file_tires"]
    key_map = self.key_maps["key_maps"]
    merged = {}

    # === Step 1: Rolling Merge ===
    for filename in files_priority:
        path = os.path.join(data["directory"], filename)
        if not os.path.isfile(path):
            continue
        content = safe_read_from_json(path) or {}
        for field, cfg in key_map.items():
            keys = cfg.get("keys", [])
            current_val = merged.get(field)
            if current_val:
                continue
            candidate = get_any_value(content, keys)
            if candidate:
                merged[field] = candidate

    # === Step 2: Continuity / Normalization ===
    if "keywords" in merged:
        kws = make_list(merged.get("keywords"))
        merged["keywords"] = sorted(set([kw.strip().lower() for kw in kws if kw]))

    if not merged.get("title") and merged.get("keywords"):
        merged["title"] = " ".join(merged["keywords"][:5]).title()

    if not merged.get("description") and merged.get("title"):
        merged["description"] = f"Video about {merged['title']}"

    # === Step 3: Refinement with BigBird + GPT2 ===
    transcript_text = merged.get("transcript") or merged.get("description") or ""
    generator = get_generator()

    if merged.get("title"):
        draft = refine_with_gpt(merged["title"], task="title", generator_fn=generator)
        if draft and len(draft.split()) > 3:
            merged["title"] = draft

    if merged.get("description"):
        draft = refine_with_gpt(transcript_text, task="description", generator_fn=generator)
        if draft and len(draft.split()) > 10:
            merged["description"] = draft

    # === Step 4: Heatmap Integration ===
    heatmap = merged.get("heatmap") or []
    highlights = []
    if isinstance(heatmap, list) and heatmap:
        top_segments = sorted(heatmap, key=lambda x: x["value"], reverse=True)[:3]
        for seg in top_segments:
            mins, secs = divmod(int(seg["start_time"]), 60)
            highlights.append(f"{mins}:{secs:02d}")
        if highlights:
            merged["description"] += "\n\nHighlights at: " + ", ".join(highlights)

    # === Step 5: Thumbnail from Heatmap Peaks ===
    try:
        if heatmap:
            top_peak = max(heatmap, key=lambda x: x["value"])
            peak_time = int((top_peak["start_time"] + top_peak["end_time"]) / 2)

            clip = VideoFileClip(data["video_path"])
            frame = clip.get_frame(peak_time)
            clip.close()

            thumb_path = os.path.join(data["directory"], "thumb.jpg")
            cv2.imwrite(thumb_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            thumbnail_url = generate_media_url(
                thumb_path,
                domain=data.get("domain", "https://example.com"),
                repository_dir=self.repository_dir
            )
            merged["thumbnail_url"] = thumbnail_url
    except Exception as e:
        print(f"Thumbnail selection failed: {e}")

    # === Step 6: Hashtag Generation ===
    hashtags = []
    for kw in merged.get("keywords", []):
        clean_kw = kw.replace(" ", "")
        if clean_kw.isalpha() and len(clean_kw) > 2:
            hashtags.append(f"#{clean_kw}")
    hashtags = hashtags[:10]
    if hashtags:
        merged["description"] += "\n\n" + " ".join(hashtags)
        merged["hashtags"] = hashtags

    # === Step 7: Category Classification ===
    def classify_category(keywords):
        kws = [kw.lower() for kw in keywords]
        if any(k in kws for k in ["comedy", "funny", "skit", "humor"]):
            return "Comedy"
        if any(k in kws for k in ["music", "song", "album", "concert"]):
            return "Music"
        if any(k in kws for k in ["news", "politics", "report", "debate"]):
            return "News & Politics"
        if any(k in kws for k in ["education", "tutorial", "lesson", "howto"]):
            return "Education"
        if any(k in kws for k in ["gaming", "playthrough", "walkthrough", "esports"]):
            return "Gaming"
        if any(k in kws for k in ["sports", "game", "match", "tournament"]):
            return "Sports"
        if any(k in kws for k in ["review", "tech", "product", "unboxing"]):
            return "Science & Technology"
        return "Entertainment"

    merged["category"] = classify_category(merged.get("keywords", []))

    # === Step 8: Chapters (YouTube-style) ===
    chapters = []
    if isinstance(heatmap, list) and heatmap:
        top_segments = sorted(heatmap, key=lambda x: x["value"], reverse=True)[:5]
        for seg in top_segments:
            start_time = int(seg["start_time"])
            mins, secs = divmod(start_time, 60)
            timestamp = f"{mins}:{secs:02d}"

            # short label from transcript or keywords
            snippet = transcript_text[:120] if transcript_text else "Segment"
            label = refine_with_gpt(
                snippet,
                task="title",
                generator_fn=generator
            ) or "Chapter"
            chapters.append({"time": timestamp, "title": label})

    if chapters:
        merged["chapters"] = chapters
        # also embed chapter list into description for YouTube
        chapter_lines = [f"{c['time']} {c['title']}" for c in chapters]
        merged["description"] += "\n\nChapters:\n" + "\n".join(chapter_lines)

    # === Step 9: Save Final ===
    total_data_path = os.path.join(data["directory"], "total_data.json")
    safe_dump_to_file(merged, total_data_path)

    return merged
