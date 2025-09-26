from src.abstract_hugpy.video_console import get_all_data
from abstract_apis import *
url = 'https://typicallyoutliers.com/api/video_url/get_all_data'
all_data = postRequest(url,data={"url":'https://www.youtube.com/shorts/rLlWcvLBluI'})
input(all_data)

