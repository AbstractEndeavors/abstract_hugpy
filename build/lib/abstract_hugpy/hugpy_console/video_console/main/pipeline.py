# src/pipeline.py
from ..imports import *
from .utilities import *



whisper_model = whisper.load_model("small")
def get_thumbnails(video_id: str, video_path: str,thumbnails_path: str):
    """
    Extract thumbnails + OCR → persist into DB JSON.
    """

    paths = extract_video_frames_unique(video_path, thumbnails_path, video_id=video_id)
    texts = [ocr_image(p) for p in paths]
    thumbnails = {"paths": paths, "texts": texts}
    upsert_video(video_id, thumbnails=thumbnails)
    return thumbnails
class VideoPipeline:
    def __init__(self, video_url=None,video_path=None,force_refresh=False):
        self.video_url = video_url
        self.video_path = video_path
        if self.video_path and not self.video_url:
            self.video_id = generate_video_id(video_path)
        else:
            self.video_url = get_corrected_url(self.video_url)
            self.video_id = get_video_id(self.video_url)
        
        self.registry = infoRegistry()
        # Ensure DB entry at least has info
        
        self.info = self.registry.get_video_info(
            url=video_url,
            video_id=self.video_id,
            video_path=self.video_path,
            force_refresh=force_refresh
            ) or {}
        self.video_info = self.info.get('info')
    def download_video(self):
        return VideoDownloader(
            url=self.video_url,
            download_video=True
            )
    def get_file_path(self,string):
        path = self.video_info.get(string)
        exists = os.path.isfile(path)
        isexist = "exists" if path and exists else 'does not exist'
        logger.info(f"{string} is  {path} and {isexist}")
        return path,exists
    # === AUDIO ===
    def ensure_audio(self):
        record = get_video_record(self.video_id)
        audio_path,exists = self.get_file_path('audio_path')
        audio_format = audio_path.split('.')[-1]
        if record and audio_path and exists:
            return audio_path, audio_format
        self.download_video()
        video_path,exists = self.get_file_path('video_path')
        os.system(f"ffmpeg -y -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}")
        
        upsert_video(self.video_id, audio_path=audio_path, audio_format=audio_format)
        return audio_path, audio_format

    # === WHISPER ===
    def get_whisper(self):
        record = get_video_record(self.video_id)
        whisper_result = record.get('whisper')
        audio_path, fmt = self.ensure_audio()
        if record and whisper_result:
            return whisper_result

        
        result = whisper_model.transcribe(audio_path)

        upsert_video(self.video_id, whisper=result)
        return result

    # === CAPTIONS ===
    def get_captions(self):
        record = get_video_record(self.video_id)
        if record and record.get("captions"):
            return record["captions"]

        whisper = self.get_whisper()
        captions = [{"start": seg["start"], "end": seg["end"], "text": seg["text"]}
                    for seg in whisper.get("segments", [])]

        upsert_video(self.video_id, captions=captions)
        return captions

    # === METADATA ===
    def get_metadata(self):
        record = get_video_record(self.video_id)
        if record and record.get("metadata"):
            return record["metadata"]

        whisper = self.get_whisper()
        captions = self.get_captions()

        text_blob = " ".join([whisper.get("text", "")] + [c["text"] for c in captions])
        title = self.info.get('title') or self.video_info.get('title') or text_blob[:50] or "Untitled"
        summary = get_summarizer_summary(text=text_blob)
        keywords = self.info.get('keywords') or self.video_info.get('keywords') or []
        tags = self.info.get('tags') or self.video_info.get('tags') or []
        extracted_keywords = get_extracted_keywords(text=text_blob) or []
        keywords = keywords+tags+extracted_keywords
        category = choose_category(keywords)
        if len(keywords) >=10:
            keywords = keywords[:10]
        metadata = {
            "title": title,
            "summary": summary,
            "category":category,
            "keywords":keywords 
        }

        upsert_video(self.video_id, metadata=metadata)
        return metadata

    # === ALL ===
    def get_thumbnails(self):
##        from abstract_utilities import get_keys
        record = get_video_record(self.video_id)
        if record and record.get("thumbnails"):
            return record["thumbnails"]
        self.download_video()
        video_id = self.video_id
        video_path = self.video_info.get("video_path")
        thumbnails_dir = self.video_info.get("thumbnails_directory")
        thumbnails = get_thumbnails(video_id,video_path,thumbnails_dir )
        upsert_video(self.video_id, thumbnails=thumbnails)
        return thumbnails

    def get_seodata(self):
##        from abstract_utilities import get_keys
        record = get_video_record(self.video_id)
        if record and record.get("seodata"):
            return record["seodata"]
        dl = VideoDownloader(url=self.video_url, download_video=True)
        whisper_result = self.get_whisper()
        captions = self.get_captions()
        
        thumbnails = self.get_thumbnails()
        thumbnails_dir = self.video_info.get("thumbnails_directory")
        thumbnails_paths = thumbnails.get('paths')
        
        
        metadata = self.get_metadata()
        title = metadata.get('title')
        summary = metadata.get('summary')
        keywords = metadata.get('keywords')
        description = metadata.get('description',summary)
        
        video_id = self.video_id
        video_path = self.video_info.get("video_path")
        if not os.path.isfile(video_path):
            file_map = get_file_map(video_path,types='video')
            video_paths = file_map.get('video')
            if video_paths and len(video_paths)>0:
                video_path = video_paths[0]
        audio_path = self.video_info.get("audio_path")
        if not os.path.isfile(audio_path):
            file_map = get_file_map(video_path,types='audio')
            audio_paths = file_map.get('audio')
            if audio_paths and len(audio_paths)>0:
                audio_path = audio_paths[0]
        
        seodata = get_seo_data(
            video_path=video_path,
            filename=video_id,
            title=title,
            summary=summary,
            description=description,
            keywords=keywords,
            thumbnails_dir=thumbnails_dir,
            thumbnail_paths=thumbnails_paths,
            whisper_result=whisper_result,
            audio_path=audio_path,
            domain='thedailydialectics.com'
            )
        upsert_video(self.video_id, seodata=seodata)
        return seodata

    def get_aggregated_data(self):
        record = get_video_record(self.video_id)
        if record and record.get("aggregated"):
            return record["aggregated"]
        video_info = record.get('info')
        directory = video_info.get('directory')
        input(directory)

        aggregated = aggregate_from_base_dir(directory=directory)
        upsert_video(self.video_id, aggregated=aggregated)
        return aggregated
    # === ALL ===
    def get_info(self):
        return get_video_record(self.video_id, hide_audio=True) or {}
    def get_all(self):
        
        return {
            "info": self.get_info(),
            "whisper": self.get_whisper(),
            "captions": self.get_captions(),
            "metadata": self.get_metadata(),
            "thumbnails": self.get_thumbnails(),
            "seodata": self.get_seodata(),
            "aggregated": self.get_aggregated_data()
        }
