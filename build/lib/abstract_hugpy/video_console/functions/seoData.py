from abstract_ocr.text_utils import *
from abstract_ocr.audio_utils import *
from abstract_ocr.video_utils import *
from abstract_ocr.functions import (
    logger,
    create_key_value,
    os,
    timestamp_to_milliseconds,
    format_timestamp,
    get_time_now_iso,
    parse_timestamp,
    url_join,
    get_from_list,
    quote,
    datetime,
    update_sitemap,
)
from abstract_utilities import *

def get_whisper_result_data(**kwargs):
    """Load whisper result JSON if path is provided."""
    whisper_result_path = kwargs.get("whisper_result_path")
    if whisper_result_path and os.path.isfile(whisper_result_path):
        return safe_load_from_file(whisper_result_path)
    return {}

def generate_info_json(
    filepath=None,
    prompt=None,
    alt_text=None,
    title=None,
    description=None,
    keywords=None,
    domain=None,
    video_path=None,
    repository_dir=None,
    generator=None,
    LEDTokenizer=None,
    LEDForConditionalGeneration=None,
):
    """
    Build structured info.json for an image/video, including SEO schema & social metadata.
    """
    dirname = os.path.dirname(filepath or "")
    basename = os.path.basename(filepath or "")
    filename, ext = os.path.splitext(basename)

    # AI prompts
    title_prompt = generate_with_bigbird(f"Video of {filename} with text {alt_text}", task="title")
    description_prompt = generate_with_bigbird(f"Video of {filename} with text {alt_text}", task="description")
    caption_prompt = generate_with_bigbird(f"Video of {filename} with text {alt_text}", task="caption")

    # File metadata
    img_meta = get_image_metadata(str(filepath)) if filepath and os.path.isfile(filepath) else {
        "dimensions": {"width": 0, "height": 0}, "file_size": 0.0
    }
    dimensions = img_meta.get("dimensions", {})
    width, height = dimensions.get("width"), dimensions.get("height")
    file_size = img_meta.get("file_size")

    # Defaults
    description = alt_text or description or ""
    title = title or filename
    caption = alt_text or caption_prompt

    # Optional HuggingFace generator
    if generator and prompt:
        try:
            gen = generator(prompt, max_length=100, num_return_sequences=1)[0]
            description = gen.get("generated_text", description)[:150]
        except Exception as e:
            logger.warning(f"Generator failed: {e}")

    info = {
        "alt": alt_text,
        "caption": caption,
        "keywords_str": keywords,
        "filename": filename,
        "ext": ext,
        "title": f"{title} ({width}×{height})",
        "dimensions": dimensions,
        "file_size": file_size,
        "license": "CC BY-SA 4.0",
        "attribution": "Created by thedailydialectics for educational purposes",
        "longdesc": description,
        "schema": {
            "@context": "https://schema.org",
            "@type": "ImageObject",
            "name": filename,
            "description": description,
            "url": generate_media_url(filepath, domain=domain, repository_dir=repository_dir),
            "contentUrl": generate_media_url(video_path, domain=domain, repository_dir=repository_dir),
            "width": width,
            "height": height,
            "license": "https://creativecommons.org/licenses/by-sa/4.0/",
            "creator": {"@type": "Organization", "name": "thedailydialectics"},
            "datePublished": datetime.now().strftime("%Y-%m-%d"),
        },
        "social_meta": {
            "og:image": generate_media_url(filepath, domain=domain, repository_dir=repository_dir),
            "og:image:alt": alt_text,
            "twitter:card": "summary_large_image",
            "twitter:image": generate_media_url(filepath, domain=domain, repository_dir=repository_dir),
        },
    }
    return info

def get_seo_title(title=None, keywords=None, filename=None, title_length=70, description=None):
    """Construct SEO title with keyword priority."""
    primary_keyword = filename or (keywords[0] if keywords else "")
    print(primary_keyword)
    seo_title = f"{primary_keyword} - {title}"
    title_length= title_length or 70
    return get_from_list(seo_title, length=title_length)

def get_seo_description(description=None, keywords=None, keyword_length=3, desc_length=300):
    """Construct SEO description with keyword hints."""
    seo_desc = f"{description or ''} Explore {keywords or ''}"
    return get_from_list(seo_desc, length=desc_length)

def get_title_tags_description(
    title=None,
    keywords=None,
    summary=None,
    filename=None,
    title_length=None,
    summary_length=150,
    keyword_length=3,
    desc_length=300,
    description=None,
):
    """Return SEO title, keyword string, description, and filtered tags."""
    summary_desc = get_from_list(description, length=summary_length)
    keywords_str = ""
    seo_title = get_seo_title(title=title, keywords=keywords, filename=filename, title_length=title_length)

    if isinstance(keywords, list):
        keywords = get_from_list(keywords, length=keyword_length)
        if keywords and len(keywords) > 0 and isinstance(keywords[0], list):
            keywords = keywords[0]
        if keywords:
            kedomainywords_str = ", ".join(keywords)

    seo_description = eatAll(
        get_seo_description(summary_desc, keywords_str, keyword_length=keyword_length, desc_length=desc_length),["'",'"',' ','\n','\t']
    )
    seo_tags = [kw for kw in (keywords or []) if kw.lower() not in ["video", "audio", "file"]]
    return seo_title, keywords_str, seo_description, seo_tags

def get_seo_data(video_path=None,
                 filename=None,
                 title=None,
                 summary=None,
                 description=None,
                 keywords=None,
                 thumbnails_dir=None,
                 thumbnail_paths=None,
                 whisper_result=None,
                 audio_path=None,
                 domain=None):
    """
    Enrich video/image info dict with SEO fields, captions, thumbnails, whisper, schema markup.
    """
    

    # Title/filename normalization

    info = {}
    domain = domain or "https://typicallyoutliers.com"
    if not filename and video_path:
        basename = os.path.basename(video_path)
        filename, ext = os.path.splitext(basename)
    title = title or filename

    # SEO text
    seo_title, keywords_str, seo_description, seo_tags = get_title_tags_description(
        title=title,
        keywords=keywords,
        summary=summary,
        filename=filename,
        description=description
        )
    info["seo_data"] = {"seo_title": seo_title, "seo_description": seo_description, "seo_tags": seo_tags,"keywords_str":keywords_str}

    # Thumbnail defaults

    
    if thumbnail_paths:
        thumb_file = thumbnail_paths[0]
        thumb_base = os.path.basename(thumb_file)
        alt_text = os.path.splitext(thumb_base)[0]
        info["seo_data"]["thumbnail"] = {"file_path": thumb_file, "alt_text": alt_text}
    elif thumbnails_dir and os.path.isdir(thumbnails_dir):
        thumbs = os.listdir(thumbnails_dir)
        thumb_file = thumbs[0]
        thumb_base = os.path.join(thumbnails_dir,thumb_file)
        alt_text = os.path.splitext(thumb_file)[0]
        info["seo_data"]["thumbnail"] = {"file_path": thumb_file, "alt_text": alt_text}
    if whisper_result.get("segments"):
        thumb_score = pick_optimal_thumbnail(whisper_result, keywords, thumbnails_dir, info=info)
        if thumb_score:
            frame, score, matched_text = thumb_score
            info["seo_data"]["thumbnail"].update({
                "file_path": os.path.join(thumbs_dir, frame),
                "alt_text": get_from_list(matched_text, length=100),
            })


    # Audio duration
    dur_s, dur_fmt = get_audio_duration(audio_path)
    info["seo_data"]["duration_seconds"]=dur_s
    info["seo_data"]["duration_formatted"]=dur_fmt

    # Schema + social metadata
    info["seo_data"]["schema_markup"] = {
        "@context": "https://schema.org",
        "@type": "VideoObject",
        "name": info["seo_data"]["seo_title"],
        "description": info["seo_data"]["seo_description"],
        "thumbnailUrl": info["seo_data"]["thumbnail"]["file_path"],
        "duration": f"PT{int(dur_s // 60)}M{int(dur_s % 60)}S",
        "uploadDate": get_time_now_iso(),
        "contentUrl": video_path,
        "keywords": info["seo_data"]["seo_tags"],
    }
    info["seo_data"]["social_metadata"] = {
        "og:title": info["seo_data"]["seo_title"],
        "og:description": info["seo_data"]["seo_description"],
        "og:image": info["seo_data"]["thumbnail"]["file_path"],
        "og:video": video_path,
        "twitter:card": "player",
        "twitter:title": info["seo_data"]["seo_title"],
        "twitter:description": info["seo_data"]["seo_description"],
        "twitter:image": info["seo_data"]["thumbnail"]["file_path"],
    }

    # Misc
    info["seo_data"]["categories"] = info["seo_data"].get("category",{})
    info["seo_data"]["category"] = next(
        (v for k, v in info["seo_data"]["categories"].items() if k in " ".join(info["seo_data"]["seo_tags"] or "").lower()), "General"
    )
    info["seo_data"]["uploader"] = info["seo_data"].get("uploader","typicallyoutliers")
    info["seo_data"]["uploader"] = {"name": info["seo_data"]["uploader"], "url": domain}
    info["seo_data"]["publication_date"] = get_time_now_iso()
    info["seo_data"]["video_metadata"] = get_video_metadata(video_path)
    info["seo_data"]["canonical_url"] = domain

    # Sitemap update
##    update_sitemap(info, f"{os.path.dirname(info['info_dir'])}/../sitemap.xml")
    return info
