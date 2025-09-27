
from abstract_utilities import *

import torch
import os,json,unicodedata,hashlib,re,math,pytesseract,cv2,PyPDF2,argparse,whisper,shutil,os,sys,json,logging,glob,hashlib
from datetime import datetime
from  datetime import timedelta 
from PIL import Image
import numpy as np
from pathlib import Path
import moviepy.editor as mp
from pdf2image import convert_from_path
import speech_recognition as sr
from pydub.silence import detect_nonsilent
from pydub.silence import split_on_silence
from pydub import AudioSegment
from abstract_math import (divide_it,
                           multiply_it)
from typing import *
from urllib.parse import quote
from abstract_utilities import (timestamp_to_milliseconds,
                                format_timestamp,
                                get_time_now_iso,
                                parse_timestamp,
                                get_logFile,
                                url_join,
                                make_dirs,
                                safe_dump_to_file,
                                safe_read_from_json,
                                read_from_file,
                                write_to_file,
                                path_join,
                                confirm_type,
                                get_media_types,
                                get_all_file_types,
                                eatInner,
                                eatOuter,
                                eatAll,
                                get_all_file_types,
                                is_media_type,
                                safe_load_from_json)
##def clip_frames(items,initial=None,end=None):
##    initial = initial or 35
##    end = end or 35
##    items_length = len(items)
##    if items_length> initial:
##        items = items[initial:]
##    items_length = len(items)
##    if items_length> end:
##        items = items[:-end]
##    return items
def get_audio_duration(file_path):
    audio = AudioSegment.from_wav(file_path)
    duration_seconds = len(audio) / 1000
    duration_formatted = format_timestamp(len(audio))
    return duration_seconds,duration_formatted

def pick_optimal_thumbnail(whisper_result, keywords,thumbnail_paths=None,directory=None, *args,initial=None,end=None,**kwargs):
    scores = []
    keywords = keywords or []
    dirbase = os.path.basename(os.path.dirname(directory))
    image_files = thumbnail_paths
    if not thumbnail_paths:
        
        image_files = sort_frames(directory=directory)
##    image_files = clip_frames(image_files)
    first_image_file = image_files[0]
    filename,thumb_ext = os.path.splitext(first_image_file)
    
##    thumb_name,thumb_ext = os.path.splitext(os.listdir(directory)[0])
    # Process each Whisper segment
    for segment in whisper_result["segments"]:
        text = segment["text"].lower().strip()
        start_time = segment["start"]        # Find the closest thumbnail based on start time
        frame_number = math.floor(start_time)
        thumbnail_name = f"{dirbase}_frame_{frame_number}{thumb_ext}"
        
        # Check if thumbnail exists
        if thumbnail_name not in image_files:
            continue
        
        # Score the caption
        keyword_score = sum(1 for kw in keywords if kw.lower() in text)
        clarity_score = 1 if len(text) > 20 else 0
        end_phrase_penalty = -1 if "thanks for watching" in text else 0
        total_score = keyword_score + clarity_score + end_phrase_penalty
        
        # Store thumbnail path, score, and caption
        thumbnail_path = os.path.join(directory, thumbnail_name)
        scores.append((thumbnail_path, total_score, text))
       
    # Sort by score (highest first)
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores[0] if scores else None

def extract_video_frames(video_path,directory,video_id=None,frame_interval=None):
    frame_interval = frame_interval or 1
    video = VideoFileClip(video_path)
    duration = video.duration
    video_id = video_id or generate_file_id(video_path)
    for t in range(0, int(duration), frame_interval):
        frame_path = os.path.join(directory,f"{video_id}_frame_{t}.jpg")
        if not os.path.isfile(frame_path):
            frame = video.get_frame(t)
            cv2.imwrite(frame_path, cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))

def get_video_metadata(file_path):
    video = mp.VideoFileClip(file_path)
    
    metadata = {
        'resolution': f"{video.w}x{video.h}",
        'format': 'MP4',
        'file_size_mb': os.path.getsize(file_path) / (1024 * 1024)
    }
    
    video.close()
    return metadata


def preprocess_for_ocr(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    # Denoise
    denoised = cv2.bilateralFilter(contrast, d=9, sigmaColor=75, sigmaSpace=75)

    # Thresholding
    thresh = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=2
    )
    # Sharpen
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.dilate(thresh, kernel, iterations=1)    
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(thresh, -1, sharpen_kernel)

    return sharpened

def extract_text_from_image(image_path: str) -> str:
    try:
        processed_img = preprocess_for_ocr(image_path)
        pil_img = Image.fromarray(cv2.bitwise_not(processed_img))  # invert for OCR
        text = pytesseract.image_to_string(pil_img, lang='eng')
        return text
    except Exception as e:
        print(f"[OCR Error] {e}")
        return ""

def is_frame_analyzed(frame_file,video_text_data):
    for values in video_text_data:
        frame = values
        if isinstance(values,dict):
            frame = values.get("frame")
        if frame_file == frame:
            return True

def extract_image_text(image_path,remove_phrases=None):
    remove_phrases = remove_phrases or []
    if is_media_type(image_path,media_types=['image']):
        raw_text = extract_text_from_image(image_path)
        cleaned_text = clean_text(raw_text)
        text = determine_remove_text(cleaned_text,remove_phrases=remove_phrases)
        return text

def extract_text_from_frame(image_path,image_texts,remove_phrases=None):
    basename = os.path.basename(image_path)
    if not is_frame_analyzed(basename,image_texts):
        text = extract_image_text(image_path,remove_phrases)
        if text:
            image_texts.append( {"frame": basename, "text": text})
    return image_texts

def extract_image_texts_from_directory(directory,image_texts=None,remove_phrases=None):
    image_texts = image_texts or []
    image_files = get_all_file_types(types=['image'],directory=directory)
    for i,image_path in enumerate(image_files):
        image_texts = extract_text_from_frame(image_path,image_texts)
        
    image_texts = sort_frames(image_texts)
    return image_texts


def determine_remove_text(text,remove_phrases=None):
    remove_phrases=remove_phrases or []
    found = False
    for remove_phrase in remove_phrases:
        if remove_phrase in text:
            found = True
            break
    if found == False:
        return text




def getPercent(i):
    return divide_it(i, 100)


def determine_remove_text(text,remove_phrases=None):
    remove_phrases=remove_phrases or []
    found = False
    for remove_phrase in remove_phrases:
        if remove_phrase in text:
            found = True
            break
    if found == False:
        return text


def generate_file_id(path: str, max_length: int = 50) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    base = unicodedata.normalize('NFKD', base).encode('ascii', 'ignore').decode('ascii')
    base = base.lower()
    base = re.sub(r'[^a-z0-9]+', '-', base).strip('-')
    base = re.sub(r'-{2,}', '-', base)
    if len(base) > max_length:
        h = hashlib.sha1(base.encode()).hexdigest()[:8]
        base = base[: max_length - len(h) - 1].rstrip('-') + '-' + h
    return base

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s:.,-]', '', text)
    text = text.strip()
    return text

def get_frame_number(file_path):
    if isinstance(file_path,dict):
        file_path = file_path.get('frame')
        
    file_path = '.'.join(file_path.split('.')[:-1])
    return int(file_path.split('_')[-1])

def sort_frames(frames=None,directory=None):
    if frames in [None,[]] and directory and os.path.isdir(directory):
        frames = get_all_file_types(types=['image'],directory=directory)
    frames = frames or []
    
    frames = sorted(
        frames,
        key=lambda x: get_frame_number(x) 
    )
    return frames
    

def get_from_list(list_obj=None,length=1):
    list_obj = list_obj or []
    if len(list_obj) >= length:
        list_obj = list_obj[:length]
    return list_obj

def get_image_metadata(file_path):
    """Extract image metadata (dimensions, file size)."""
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            file_size = get_file_size(file_path)
        return {
            "dimensions": {"width": width, "height": height},
            "file_size": round(file_size, 3)
        }
    except Exception as e:
        return {"dimensions": {"width": 0, "height": 0}, "file_size": 0}

def update_json_data(json_data,update_data,keys=None):
    if keys == True:
        values_string = ''
        for key,value in update_data.items():
            values_string+= f"{key} == {value}"

        logger.info(f"new_datas:{values_string}")
        keys = valid_keys
    
    for key,value in update_data.items():
        if keys:
            if key in keys:
                json_data[key] = json_data.get(key) or value 
        else:
            json_data[key] = json_data.get(key) or value 
    return json_data


def update_sitemap(video_data,
                   sitemap_path):
    with open(sitemap_path, 'a') as f:
        f.write(f"""
<url>
    <loc>{video_data.get('canonical_url')}</loc>
    <video:video>
        <video:title>{video_data.get('seo_title')}</video:title>
        <video:description>{video_data.get('seo_description')}</video:description>
        <video:thumbnail_loc>{video_data.get('thumbnail',{}).get('file_path',{})}</video:thumbnail_loc>
        <video:content_loc>{video_data.get('video_path')}</video:content_loc>
    </video:video>
</url>
""")

def prune_inputs(func, *args, **kwargs):
    """
    Adapt the provided args/kwargs to fit the signature of func.
    Returns (args, kwargs) suitable for calling func.
    """
    sig = inspect.signature(func)
    params = sig.parameters

    # Handle positional arguments
    new_args = []
    args_iter = iter(args)
    for name, param in params.items():
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY,
                          inspect.Parameter.POSITIONAL_OR_KEYWORD):
            try:
                new_args.append(next(args_iter))
            except StopIteration:
                break
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            # collect all remaining args
            new_args.extend(args_iter)
            break
        else:
            break

    # Handle keyword arguments
    new_kwargs = {}
    for name, param in params.items():
        if name in kwargs:
            new_kwargs[name] = kwargs[name]
        elif param.default is inspect.Parameter.empty and param.kind == inspect.Parameter.KEYWORD_ONLY:
            # Required keyword not provided
            raise TypeError(f"Missing required keyword argument: {name}")

    # Only include keywords func accepts
    accepted_names = {
        name for name, p in params.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.KEYWORD_ONLY)
    }
    new_kwargs = {k: v for k, v in new_kwargs.items() if k in accepted_names}

    return tuple(new_args), new_kwargs




def generate_file_id(path: str, max_length: int = 50) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    base = unicodedata.normalize('NFKD', base).encode('ascii', 'ignore').decode('ascii')
    base = base.lower()
    base = re.sub(r'[^a-z0-9]+', '-', base).strip('-')
    base = re.sub(r'-{2,}', '-', base)
    if len(base) > max_length:
        h = hashlib.sha1(base.encode()).hexdigest()[:8]
        base = base[: max_length - len(h) - 1].rstrip('-') + '-' + h
    return base

def get_video_id(**kwargs):
    info_data = kwargs.get('info_data',kwargs) or kwargs or {}
    info_dir = info_data.get('info_dir') or info_data.get('info_directory')
    video_id = info_data.get('video_id')
    video_path = info_data.get('video_path')
    if info_dir:
        video_id = os.path.basename(info_dir)
    if video_path:
        video_id = generate_file_id(video_path)
    if video_id:
        return video_id

def get_videos_path(directory = None, info_data = None):
    info_data = info_data or {}
    if info_data and directory == None:
        directory = info_data['output_dir']
    directory = directory or TEXT_DIR
    return directory

def get_video_basenames(directory = None, info_data = None):
    directory = get_videos_path(directory = None, info_data = None)
    directory_items = os.listdir(directory)
    return directory_items


def get_videos_paths(directory = None, info_data = None):
    directory = get_videos_path(directory = directory, info_data = info_data)
    video_basenames = get_video_basenames(directory = directory, info_data = directory)
    directory_items = [os.path.join(directory,basename) for basename in video_basenames]
    return directory_items


def get_thumbnails_dir(info_dir=None,**kwargs):
    video_info_dir = info_dir or get_video_info_dir(**kwargs)
    thumbnails_directory=os.path.join(video_info_dir,'thumbnails')
    os.makedirs(thumbnails_directory,exist_ok=True)
    return thumbnails_directory


def get_video_info_dir(**kwargs):
    video_id = get_video_id(**kwargs)
    info_dir = make_dirs(TEXT_DIR,video_id)
    os.makedirs(info_dir,exist_ok=True)
    get_thumbnails_dir(info_dir)
    return info_dir


def get_video_info_path(**kwargs):
    info_dir = get_video_info_dir(**kwargs)
    info_path = os.path.join(info_dir,'info.json')
    return info_path


def get_video_info_data(**kwargs):
    info_data=kwargs.get('info_data',kwargs) or kwargs  or {}
    info_file_path = None
    if info_data and isinstance(info_data,str) and os.path.isdir(info_data):
        info_dir = info_data
        info_file_path = os.path.join(info_dir,'info.json')
    elif info_data and isinstance(info_data,str) and os.path.isfile(info_data):
        info_file_path = info_data
    else:
        info_file_path = get_video_info_path(**info_data)
    if os.path.isfile(info_file_path):
        info_data = safe_load_from_json(info_file_path)
        return info_data


def get_audio_path(**kwargs):
    info_dir = get_video_info_dir(**kwargs)
    audio_path = os.path.join(info_dir,'audio.wav')
    return audio_path


def get_video_basename(**kwargs):
    video_path = kwargs.get('video_path')
    if not video_path:
        info_data = get_video_info_data(**kwargs)
        video_path = info_data.get('video_path')
    if video_path:
        basename= os.path.basename(video_path)
        return basename

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
        get_seo_description(summary_desc, keywords_str, keyword_length=keyword_length, desc_length=desc_length),["'",'"',' ','','	']
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
        thumb_score = pick_optimal_thumbnail(whisper_result, keywords,thumbnail_paths, thumbnails_dir, info=info)
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

