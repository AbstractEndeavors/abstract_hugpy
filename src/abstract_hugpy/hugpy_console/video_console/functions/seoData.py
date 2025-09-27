
from abstract_utilities import *
def create_key_value(json_obj, key, value):
    json_obj[key] = json_obj.get(key, value) or value
    return json_obj

def getPercent(i):
    return divide_it(i, 100)

def getPercentage(num, i):
    percent = getPercent(i)
    percentage = multiply_it(num, percent)
    return percentage

def if_none_get_def(value, default):
    if value is None:
        value = default
    return value

def if_not_dir_return_None(directory):
    str_directory = str(directory)
    if os.path.isdir(str_directory):
        return str_directory
    return None

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
            values_string+= f"{key} == {value}\n"
        logger.info(f"new_datas:\n{values_string}")
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
def _format_srt_timestamp(seconds: float) -> str:
    """
    Convert seconds (e.g. 3.2) into SRT timestamp "HH:MM:SS,mmm"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - math.floor(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
def execute_if_bool(bool_key,function,keys,req=None,info_data=None):
    new_data,info_data = get_key_vars(keys,req,info_data)
    bool_response = bool_key
    if not isinstance(bool_response,bool):
        bool_response = info_data.get(bool_key) in [None,'',[],"",{}]
    logger.info(f"{bool_key} == {bool_response}")
    if bool_response:
        args, kwargs = prune_inputs(function, **new_data, flag=True)
        info = function(*args, **kwargs)

        info_data = update_json_data(info_data,info,keys=True)
    safe_dump_to_file(data=info_data,file_path=get_video_info_path(**info_data))
    return info_data
import inspect

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

def get_videos_infos(directory = None, info_data = None):
    directory_items = get_videos_paths(directory = directory, info_data = info_data)
    directory_infos = [get_video_info_data(item_path) for item_path in directory_items]
    return directory_infos

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

def get_audio_bool(**kwargs):
    audio_path = get_audio_path(**kwargs)
    if audio_path:  
        return os.path.isfile(audio_path)
    return False
def get_video_basename(**kwargs):
    video_path = kwargs.get('video_path')
    if not video_path:
        info_data = get_video_info_data(**kwargs)
        video_path = info_data.get('video_path')
    if video_path:
        basename= os.path.basename(video_path)
        return basename
def get_video_filename(**kwargs):
    basename = get_video_basename(**kwargs)
    filename,ext = os.path.splitext(basename)
    return filename
def get_video_ext(**kwargs):
    basename = get_video_basename(**kwargs)
    filename,ext = os.path.splitext(basename)
    return ext
def get_canonical_url(**kwargs):
    video_id = get_video_id(**kwargs)
    videos_url = kwargs.get('videos_url') or kwargs.get('video_url') or VIDEO_URL
    canonical_url = f"{videos_url}/{video_id}"
    return canonical_url
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
