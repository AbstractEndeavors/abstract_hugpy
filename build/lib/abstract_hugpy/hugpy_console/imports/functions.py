from .imports import *
from .image_utils import *
def get_abs_videos_directory(directory=None):
    if not directory:
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'videos')
    os.makedirs(directory, exist_ok=True)
    return directory

def get_video_id(video_url):
    return video_url.split('/')[-1].split('=')[-1]

def export_srt(segments, path):
    with open(path, 'w') as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n{str(seg['start']).replace('.', ',')} --> {str(seg['end']).replace('.', ',')}\n{seg['text']}\n\n")

def get_from_local_host(endpoint, **kwargs):
    return postRequest(f"https://abstractendeavors.com{endpoint}", data=kwargs)

def download_audio(youtube_url, audio_path, output_format="wav"):
    if output_format.startswith("."):
        output_format = output_format[1:]

    if audio_path.endswith(f".{output_format}"):
        audio_path = audio_path[:-(len(output_format)+1)]

    if output_format == "webm":
        # raw download, no conversion
        ydl_opts = {
            "format": "251",  # opus/webm
            "outtmpl": f"{audio_path}.webm",  # force extension
            "overwrites": True,
        }
        final_path = f"{audio_path}.webm"
    else:
        # conversion via ffmpeg
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": audio_path,
            "overwrites": True,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": output_format,
                "preferredquality": "0",
            }],
        }
        final_path = f"{audio_path}.{output_format}"

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    return final_path





