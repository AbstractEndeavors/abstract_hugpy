from abstract_utilities import get_logFile,make_list,get_any_value
import urllib,bs4
from urllib.parse import urlparse, parse_qs, unquote
from typing import List, Optional
logger = get_logFile(__name__)
from abstract_webs import *
from for_dl_video import *
def sanitize_youtube_url(url: str) -> str:
    if "youtu.be" in url:
        return url.replace("www.youtu.be", "youtu.be")
    return url
def get_metas(url):
        type_vars={}
##    try:
        for soup in get_soup(url):
            for sou in soup:
                for meta in sou.find_all('meta'):
                    meta_prop = meta.get('property')
                    if meta_prop:
                        for typ in ['twitter','og']:
                            if meta_prop.startswith(typ):
                                if typ not in type_vars:
                                    type_vars[typ] = {}
                                prop_typ = meta_prop.split(':')[-1]
                                if meta:
                                    type_vars[typ][prop_typ] = meta.get('content')
                                if prop_typ not in type_vars:
                                    type_vars[prop_typ] = type_vars[typ][prop_typ]
##    except Exception as e:
##        logger.info(f"{e}")
        return type_vars
def get_dl_vid(url,download_directory=None,output_filename=None,get_info=None,download_video=None):
##    try:
        video_mgr = dl_video(url,download_directory=download_directory,output_filename=output_filename,get_info=get_info,download_video=download_video)
        video_info = get_video_info_from_mgr(video_mgr)
        if video_info:
            return video_info
##    except:
##        pass

def for_dl_soup_vid(url,download_directory=None,output_filename=None,get_info=None,download_video=None):
    videos = soupManager(url).soup.find_all('video')
    for video in videos:
        video_info=None
##        try:
        if video and isinstance(video,dict):
            video_mgr = dl_video(video.get("src"),download_directory=download_directory,output_filename=output_filename,get_info=get_info,download_video=download_video)
            video_info = get_video_info_from_mgr(video_mgr)
##        except:
##            video_info=None
        if video_info:
            return video_info
        


def is_valid_url(url: str) -> bool:
    """Check if a string is a valid URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])  # Must have scheme (e.g., http) and netloc (e.g., example.com)
    except ValueError:
        return False

def process_urls(urls: List[str], extract: str = "base") -> List[str]:
    """
    Process a list of URLs to extract either the base URL or a specific query parameter.
    Args:
        urls: List of URLs or strings containing URLs.
        extract: What to extract ("base" for base URL, or a query parameter like "v").
    Returns:
        List of unique, processed URL components.
    """
    result = []
    for url in make_list(urls):
        # Handle strings that may contain multiple URLs or fragments
        url = unquote(url.strip())  # Decode URL-encoded characters
        if not is_valid_url(url):
            # Try to extract URLs from fragments containing 'http'
            if 'http' in url:
                for part in url.split('http')[1:]:
                    candidate = f"http{part}"
                    if is_valid_url(candidate):
                        parsed = urlparse(candidate)
                        if extract == "base":
                            base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                            if base_url not in result:
                                result.append(base_url)
                        else:
                            query_params = parse_qs(parsed.query)
                            values = query_params.get(extract, [])
                            result.extend([v for v in values if v and v not in result])
            continue
        # Valid URL
        parsed = urlparse(url)
        if extract == "base":
            base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if base_url not in result:
                result.append(base_url)
        else:
            query_params = parse_qs(parsed.query)
            values = query_params.get(extract, [])
            result.extend([v for v in values if v and v not in result])
    return result
def get_urls(url,urls = []):
    for url in make_list(url):
        if isinstance(url,list):
            for url in make_list(url):
               urls = get_urls(url,urls = urls)
        elif isinstance(url,dict):
            urls.append(url.get('value'))
        else:
            urls.append(url)
    return urls
def get_url_list(urls = []):
    url_list = []
    for url in make_list(urls):
        string = ''
        if url.startswith('http'):
            string = 'http'
        for piece in url.split('http'):
            string += piece
            string = sanitize_youtube_url(string)
            url_list.append(string.split('?')[0])
            string = 'http'
            
    return url_list
def get_desired_links(url):
    urls = make_list(url)
##    try:
    urlMgr = linkManager(url)
    urls = urlMgr.all_desired_links
##    except Exception as e:
##        logger.info(f"{e}")
    return urls
def deriveUrlList(url):
    urls = get_desired_links(url)
    url_functions = [get_urls,get_url_list,process_urls]
    for url_function in url_functions:
        try:
            urls = url_function(urls)
        except Exception as e:
            logger.info(f"{e}")
    return urls
def validate_video_urls(urls,
                        get_info_url=False,
                        get_for_video=False,
                        download_directory=None,
                        output_filename=None,
                        get_info=True,
                        download_video=False):
    output_urls = []
    for url in make_list(urls):
        video_info=None
        if url:
            video_info = dl_video(
                            url,
                            download_directory=download_directory,
                            output_filename=output_filename,
                            get_info=get_info,
                            download_video=download_video
                            )
        if video_info:
            output_urls.append(url)
            if get_info_url or get_for_video and video_info and isinstance(video_info,dict) and video_info != {}:
                output_urls[-1] = video_info
                if get_for_video:   
                    dl_info  = for_dl_video(
                                    url,
                                    download_directory=download_directory,
                                    output_filename=output_filename,
                                    get_info=True,
                                    download_video=True
                                    )
                    if dl_info:
                        output_urls[-1]=dl_info
                if get_info_url:
                    if isinstance(output_urls[-1],dict):
                        output_urls[-1]['initial_url'] = url
                return output_urls[-1]
                    
    return output_urls


def sanitize_youtube_url(url: str) -> str:
    """Fix known bad shortlinks like www.youtu.be → youtu.be, 
    and normalize to youtube.com/watch if desired."""
    if not url:
        return url
    parsed = urlparse(url)
    if parsed.netloc in {"threads.com", "www.threads.com"}:
        parsed = parsed._replace(netloc="www.threads.net")  # fallback for older extractors
        return urlunparse(parsed)
    if parsed.netloc == "www.youtu.be":
        parsed = parsed._replace(netloc="youtu.be")
    # normalize short link → full form
    if parsed.netloc == "youtu.be":
        video_id = parsed.path.strip("/")
        return f"https://www.youtube.com/watch?v={video_id}"

    return urlunparse(parsed)

def get_bolshevid_videos(url,
                        get_info_url=True,
                        get_for_video=True,
                        download_directory=None,
                        output_filename=None,
                        get_info=True,
                        download_video=True):

    urls = deriveUrlList(url)
    video_urls = validate_video_urls(urls,
                        get_info_url=get_info_url,
                        get_for_video=get_for_video,
                        download_directory=download_directory,
                        output_filename=output_filename,
                        get_info=get_info,
                        download_video=download_video)
    return video_urls

