from ..imports import *
from .manager import *
def get_keywords_from_url(url):
    text = get_soup_text(url)
    return extract_keywords(text=text)
def get_text_keywords(
    *args,
    text=None,
    keywords=None,
    url=None,
    video_url=None,
    **kwargs
    ):
    text=get_text(
        text=text,
        url=url,
        video_url=video_url
        )
    if text and not keywords:
        keywords = run_keybert_func(extract_keywords_nlp,
                    *args,
                    text=text,
                    **kwargs
                    )   
        # Normalize: extract just the strings if tuples
        keywords = [kw if isinstance(kw, str) else kw[0] for kw in keywords]
    
    return text,keywords
def get_keyword_density(
    text=None,
    keywords=None,
    url=None,
    video_url=None
):
    text, keywords = get_text_keywords(
        text=text,
        keywords=keywords,
        url=url,
        video_url=video_url
    )
    # Ensure keywords are plain strings
    keywords = [kw if isinstance(kw, str) else kw[0] for kw in (keywords or [])]
    return run_keybert_func(
        calculate_keyword_density,
        text=text,
        keywords=keywords
    )

def get_refined_keywords(
        *args,
        text=None,
        keywords=None,
        url=None,
        video_url=None,
        **kwargs
        ):
    text=get_text(
        text=text,
        url=url,
        video_url=video_url
        )
    return run_keybert_func(
        refine_keywords,
        *args,
        text=text,
        **kwargs
        )    
def get_extracted_keywords(
        *args,
        text=None,
        keywords=None,
        url=None,
        video_url=None,
        **kwargs
        ):
    text=get_text(
        text=text,
        url=url,
        video_url=video_url
        )
    return run_keybert_func(extract_keywords_nlp,
        *args,
        text=text,
        **kwargs
        )   
