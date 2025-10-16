from src.abstract_hugpy import *
from abstract_webtools import *
##from abstract_apis import *
video_url = 'https://youtu.be/0XFudmaObLI?list=RDMM8Q0cp4b9pvg'

video_mgr = VideoPipeline(video_url)
#all_data = postRequest(url,data={"url":'https://www.youtube.com/shorts/rLlWcvLBluI'})
##info = registry.get_video_info(url)
everything = {}
result = deepcoder.generate("hello")
input(result)
everything["video_id"]=get_video_id(video_url)
print(everything["video_id"])
everything["info"]=video_mgr.download_video()
print(everything["info"])
everything["audio"]=video_mgr.ensure_audio()
print(everything["audio"])
everything["whisper_result"]=video_mgr.get_whisper()
print(everything["whisper_result"])
everything["thumbnails"]=video_mgr.get_thumbnails()
print(everything["thumbnails"])
everything["captions"]=video_mgr.get_captions()
print(everything["captions"])
everything["metadata"]=video_mgr.get_metadata()
print(everything["metadata"])
everything["aggregated_data"]=video_mgr.get_aggregated_data()
print(everything["aggregated_data"])
