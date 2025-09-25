from src.abstract_hugpy.video_console import get_all_data
from abstract_apis import *
url = 'https://typicallyoutliers.com/api/video_url/get_video_metadata_path'
all_data = requests.post(url,data={"url":'https://www.youtube.com/shorts/rLlWcvLBluI'})
input(all_data)

