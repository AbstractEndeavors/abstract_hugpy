from src.abstract_hugpy import get_video_mgr
##from abstract_apis import *
url = 'https://www.youtube.com/watch?v=t-knFuqQdGc'
#all_data = postRequest(url,data={"url":'https://www.youtube.com/shorts/rLlWcvLBluI'})
##info = registry.get_video_info(url)
video_mgr = get_video_mgr()
input(video_mgr.download_video(url))
