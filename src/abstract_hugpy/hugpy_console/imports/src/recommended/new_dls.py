from abstract_webtools import get_soup
def rid_all(string,key,rep):
    while True:
        if key in string:
            string = string.replace(key,rep)
        else:
            return string
def for_dl_soup_vid(url,download_directory=None,output_filename=None,get_info=None,download_video=None):
    videos = get_soup(url)
    for video in videos:
        video_info=None
        if video and isinstance(video,dict):
            video_mgr = dl_video(video.get("src"),download_directory=download_directory,output_filename=output_filename,get_info=get_info,download_video=download_video)
            video_info = get_video_info_from_mgr(video_mgr)
        if video_info:
            return video_info

url = 'https://www.youtube.com/shorts/rLlWcvLBluI'
#response = for_dl_soup_vid(url)
string = str(get_soup(url))
string= rid_all(string,'\n',' ')
string= rid_all(string,'  ',' ')
string= rid_all(string,'> <','><')
string= string.replace('><','>*^**^*<')
strings = [string for string in string.split('*^*') if string]
for piece in strings:
    if piece.startswith('<script'):
        start = piece.split('>')[0]+'>'
        end = '</script>'
        code = piece.split(end)[0]
        lines = code.split(';')
        for line in lines:
            input(line)
