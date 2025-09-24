from ..imports import *
from .seoData import *
from ..utils import derive_video_metadata

def _merge_sources(self, video_url):
    """Collect whisper, captions, thumbnails into one big text blob."""
    data = self.get_data(video_url)
    whisper = self.get_whisper_result(video_url)
    txt_parts = [whisper.get("text", "")]
    if os.path.isfile(data.get("srt_path", "")):
        txt_parts.append(safe_load_from_file(data["srt_path"]) or "")
    thumbs = self.get_thumbnails(video_url)
    if thumbs.get("thumbnail_texts"):
        txt_parts.extend(thumbs["thumbnail_texts"])
    return " ".join([t for t in txt_parts if t])

def _summarize_text(text, max_len=120, min_len=50):
    try:
        return summarizer(
            text, max_length=max_len, min_length=min_len,
            truncation=True, num_beams=4, no_repeat_ngram_size=3
        )[0]["summary_text"].strip()
    except Exception as e:
        logger.warning(f"Summarization failed: {e}")
        return text[:200] + "..."

def _extract_keywords(text, top_n=10):
    try:
        kws = [kw for kw, _ in extract_keywords(
            text, keyphrase_ngram_range=(1,2), stop_words="english", top_n=top_n
        )]
        return list(dict.fromkeys(kws))  # dedup, preserve order
    except Exception as e:
        logger.warning(f"Keyword extraction failed: {e}")
        return []

def _generate_title(text):
    try:
        out = summarizer(
            "headline: " + text,
            max_length=15, min_length=5,
            truncation=True, num_beams=4
        )[0]["summary_text"]
        return out.strip()
    except Exception:
        kws = _extract_keywords(text, top_n=1)
        return kws[0].title() if kws else "Untitled"

# === exposed methods ===

def get_video_summary(self, video_url):
    merged_text = _merge_sources(self, video_url)
    return _summarize_text(merged_text)

def get_video_keywords(self, video_url):
    merged_text = _merge_sources(self, video_url)
    return _extract_keywords(merged_text)

def get_video_title(self, video_url):
    merged_text = _merge_sources(self, video_url)
    return _generate_title(merged_text)

def update_meta_data(self, metadata, video_url=None, video_id=None, data=None):
    return self.update_spec_data(
        metadata, "metadata", "metadata_path",
        video_url=video_url, video_id=video_id, data=data
    )
def get_meta_data(self,video_url):
    data = self.get_data(video_url)
    old_metadata = self.get_metadata_data(video_url)
    domain='https://typicallyoutliers.com'
    directory = data.get('directory')
    video_path = data.get('video_path')
    whisper_text = self.get_whisper_text(video_url)
    metadata = derive_video_metadata(
        video_path=video_path,
        repo_dir=directory,
        domain=domain,
        transcript=whisper_text
        )
    return metadata
def get_metadata(self, video_url):
    data = self.get_data(video_url)
    metadata = data.get('metadata') or {}
    if [key for key in ["summary","keywords"] if metadata.get(key) == None]:
        metadata = self.get_meta_data(video_url)
    if not metadata.get("title"):
        metadata["title"] = self.get_video_title(video_url)
        data = self.update_meta_data(metadata, video_url)
    if not metadata.get("summary"):
        metadata["summary"] = self.get_video_summary(video_url)
        data = self.update_meta_data(metadata, video_url)

    if not metadata.get("keywords"):
        metadata["keywords"] = self.get_video_keywords(video_url)
        data = self.update_meta_data(metadata, video_url)
    if not metadata.get("seodata"):
        metadata["seodata"] = get_seo_data(
             video_path=data.get('video_path'),
             filename=data.get('video_path'),
             title=metadata.get("title"),
             summary=metadata.get("summary"),
             description=metadata.get("summary"),
             keywords=metadata.get("keywords"),
             thumbnails_dir=data.get('thumbnails_dir'),
             thumbnail_paths=data.get('thumbnail_paths'),
             whisper_result=self.get_whisper_result(video_url),
             audio_path=self.get_audio_path(video_url),
             domain='https://typicallyoutliers.com'
            )
        data = self.update_meta_data(metadata, video_url)
    self.is_complete(key="metadata", video_url=video_url)
    return data["metadata"]
