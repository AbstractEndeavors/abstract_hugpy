from abstract_utilities import *
def get_hugging_face_flask_dir():
    abs_file_path = os.path.abspath(__file__)
    abs_dir = os.path.dirname(abs_file_path)
    abs_parent_dir = os.path.dirname(abs_dir)
    hugging_face_flask_dir = os.path.join(abs_parent_dir,'hugging_face_flasks')
    return hugging_face_flask_dir
def capitalize(string):
    if not string:
        return string
    if len(string)>1:
        return f"{string[0].upper()}{string[1:].lower()}"
    return string.upper()
def capitalize_underlines(strings):
    strings = strings.split('_')
    for i,string in enumerate(strings):
        string = string.lower()
        if i >0:
            string = capitalize(string)
        strings[i] = string
    return ''.join(strings)

def get_function_attribs(text):
    function = text.split(' ')[1].split(':')[0]
    function_call = function.split('(')[0]
    function_name = capitalize_underlines(function_call)
    return function_call,function_name
def create_init(filenames):
    string = ''
    for filename in filenames:
        string+=f'from .{filename} import *\n'
    hugging_face_flask_dir = get_hugging_face_flask_dir()
    __init___path = os.path.join(hugging_face_flask_dir,'__init__.py')
    write_to_file(contents=string,file_path=__init___path)    
def get_video_texts():
    texts = """def download_video(url): return video_mgr.download_video(url)
def extract_video_audio(url): return video_mgr.extract_audio(url)
def get_video_whisper_result(url): return video_mgr.get_whisper(url)
def get_video_whisper_text(url): return get_video_whisper_result(url).get('text')
def get_video_whisper_segments(url): return get_video_whisper_result(url).get('segments')
def get_video_metadata(url): return video_mgr.get_metadata(url)
def get_video_captions(url): return video_mgr.get_captions(url)
def get_video_info(url): return video_mgr.get_data(url).get('info')
def get_video_directory(url): return video_mgr.get_data(url).get('directory')
def get_video_path(url): return video_mgr.get_data(url).get('video_path')
def get_video_audio_path(url): return video_mgr.get_data(url).get('audio_path')
def get_video_srt_path(url): return video_mgr.get_data(url).get('srt_path')
def get_video_metadata_path(url): return video_mgr.get_data(url).get('metadata_path')
"""
    return texts.split('\n')
def get_proxy_url_flask_string(function_call,function_name):
    return f'''
@proxy_video_url_bp.route("/api/{function_call}", methods=["POST","GET"])
def {function_name}():
    initialize_call_log()
    try:
        result = get_from_local_host('{function_call}',request)
        logger.info(result)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{{e}}"
        return get_json_response(value=message,status_code=500)'''
def get_video_url_flask_string(function_call,function_name):
    return f'''
@video_url_bp.route("/{function_call}", methods=["POST","GET"])
def {function_name}():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        url = data.get('url')
        if not url:
            return get_json_response(value=f"url in {{data}}",status_code=400)
        result = {function_call}(url)
        if not result:
            return get_json_response(value=f"no result for {{data}}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{{e}}"
        return get_json_response(value=message,status_code=500)'''

def get_deep_coder_flask_string():
    return f'''
@deep_coder_bp.route("/deepcoder_generate", methods=["POST","GET"])
def deepcoderGenerate():
    data = get_request_data(request)
    initialize_call_log(data=data)
    try:        
        if not data:
            return get_json_response(value=f"not prompt in {{data}}",status_code=400)
        result = deepcoder.generate(**data)
        if not result:
            return get_json_response(value=f"no result for {{data}}",status_code=400)
        return get_json_response(value=result,status_code=200)
    except Exception as e:
        message = f"{{e}}"
        return get_json_response(value=message,status_code=500)'''



def save_proxy_video_url_flask(proxy_video_url_flask_strings):
    file_name = 'proxy_video_url_flask'
    basename = f"{file_name}.py"
    hugging_face_flask_dir = get_hugging_face_flask_dir()
    proxy_video_url_flask_path = os.path.join(hugging_face_flask_dir,basename)
    write_to_file(contents='\n'.join(proxy_video_url_flask_strings),file_path=proxy_video_url_flask_path)
    return file_name

def save_video_url_flask(video_url_flask_strings):
    file_name = 'video_url_flask'
    basename = f"{file_name}.py"
    hugging_face_flask_dir = get_hugging_face_flask_dir()
    video_url_flask_path = os.path.join(hugging_face_flask_dir,basename)
    write_to_file(contents='\n'.join(video_url_flask_strings),file_path=video_url_flask_path)
    return file_name
def save_deep_coder_flask(deep_coder_flask_strings):
    file_name = 'deep_coder_flask'
    basename = f"{file_name}.py"
    hugging_face_flask_dir = get_hugging_face_flask_dir()
    deep_coder_flask_path = os.path.join(hugging_face_flask_dir,basename)
    write_to_file(contents='\n'.join(deep_coder_flask_strings),file_path=deep_coder_flask_path)
    return file_name
video_url_flask_string ="""from abstract_flask import *
from abstract_utilities import *
from ..video_utils import *
video_url_bp,logger = get_bp('video_url_bp')"""
video_url_flask_strings = [video_url_flask_string]

proxy_video_url_flask_string ="""from abstract_flask import *
from abstract_utilities import *
proxy_video_url_bp,logger = get_bp('proxy_video_url_bp')"""
proxy_video_url_flask_strings = [proxy_video_url_flask_string]

deep_coder_flask_string ="""from abstract_flask import *
from abstract_utilities import *
deep_coder_bp,logger = get_bp('deep_coder_bp')
from .. import get_deep_coder
deepcoder = get_deep_coder()"""
deep_coder_flask_strings = [deep_coder_flask_string,get_deep_coder_flask_string()]
for text in get_video_texts():
    if not text:
        continue
    function_call,function_name = get_function_attribs(text)
    
    proxy_video_url_flask_string = get_proxy_url_flask_string(function_call,function_name)
    proxy_video_url_flask_strings.append(proxy_video_url_flask_string)
    
    video_url_flask_string = get_video_url_flask_string(function_call,function_name)
    video_url_flask_strings.append(video_url_flask_string)


deep_coder_flask_name = save_deep_coder_flask(deep_coder_flask_strings)    
proxy_video_url_flask_name = save_proxy_video_url_flask(proxy_video_url_flask_strings)    
video_url_flask_name = save_video_url_flask(video_url_flask_strings)
create_init([deep_coder_flask_name,proxy_video_url_flask_name,video_url_flask_name])


