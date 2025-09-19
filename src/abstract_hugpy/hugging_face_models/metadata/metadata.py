import os
from ..imports import get_summary,refine_with_gpt,get_generator, generate_media_url,refine_keywords
from moviepy.editor import VideoFileClip
import cv2
from ..imports import *
def derive_video_metadata(video_path: str, repo_dir: str, domain: str,transcript:str) -> dict:
    """
    Derive title, keywords, category, and thumbnail URL from a video.

    Args:
        video_path (str): Path to the video file.
        repo_dir (str): Local repo directory (root for media storage).
        domain (str): Public domain for media URLs, e.g. "https://abstractendeavors.com".

    Returns:
        dict with keys: title, keywords, category, thumbnail_url
    """

    # 1. Transcribe
    # 2. Summarize → Draft Title
    summary = get_summary(transcript, summary_mode="medium")
    generator = get_generator()
    title = refine_with_gpt(summary, task="title", generator_fn=generator)

    # 3. Keywords
    keyword_data = refine_keywords(transcript, top_n=12)
    keywords = keyword_data["combined_keywords"]

    # 4. Category (simple rules)
    def choose_category(kws):
        if any(k in kws for k in ["comedy", "skit", "funny", "humor"]):
            return "Comedy"
        if any(k in kws for k in ["news", "analysis", "report"]):
            return "News & Politics"
        if any(k in kws for k in ["music", "song", "album"]):
            return "Music"
        return "Entertainment"
    category = choose_category(keywords)

    # 5. Thumbnail
    clip = VideoFileClip(video_path)
    best_frame, max_sharp = None, 0
    for t in range(0, int(clip.duration), 2):
        frame = clip.get_frame(t)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        if sharpness > max_sharp:
            max_sharp, best_frame = sharpness, frame
    clip.close()

    thumb_path = os.path.join(repo_dir, "thumb.jpg")
    cv2.imwrite(thumb_path, cv2.cvtColor(best_frame, cv2.COLOR_RGB2BGR))
    thumbnail_url = generate_media_url(thumb_path, domain=domain, repository_dir=repo_dir)

    return {
        "title": title,
        "keywords": keywords,
        "category": category,
        "thumbnail_url": thumbnail_url
    }
