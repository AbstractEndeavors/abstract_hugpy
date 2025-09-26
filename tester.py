from src.abstract_hugpy import *
from abstract_webtools import *
##from abstract_apis import *
video_url = 'https://youtu.be/0XFudmaObLI?list=RDMM8Q0cp4b9pvg'


#all_data = postRequest(url,data={"url":'https://www.youtube.com/shorts/rLlWcvLBluI'})
##info = registry.get_video_info(url)
video_mgr = get_video_mgr()

data = video_mgr.get_data(video_url)
input(data)
video_id = get_video_id(video_url)
input(video_id)
info = video_mgr.download_video(video_url)
input(info)
audio = video_mgr.extract_audio(video_url)
input(audio)
whisper_result = video_mgr.get_whisper_result(video_url)
input(whisper_result)
thumbnails = video_mgr.get_thumbnails(video_url)
input(thumbnails)
captions= video_mgr.get_captions(video_url)
input(captions)
metadata = video_mgr.get_metadata(video_url)
input(metadata)
aggregated_data =video_mgr.get_aggregated_data(video_url)
input(aggregated_data)
