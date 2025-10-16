from ..imports import *
from .manager import get_summarizer_summary
from ...whisperManager.src.manager_utils import get_whisper_text
def get_summary(
        *args,
        text=None,
        keywords=None,
        url=None,
        audio_path=None,
        video_url=None,
        video_path=None,
        **kwargs
        ):
    if video_url or audio_path or video_path:
        text = get_whisper_text(
            *args,
            audio_path=audio_path,
            video_url=video_url,
            video_path=video_path,
            **kwargs
            )
 
        return run_pruned_func(
            get_summarizer_summary,
            *args,
            text=text,
            **kwargs
            )
def get_summary_result(
        *args,
        text=None,
        url=None,
        file_path=None,
        audio_path=None,
        video_url=None,
        video_path=None,
        **kwargs
        ):
    
    if url or file_path:
        text=get_text(
            text=text,
            url=url,
            file_path=file_path,
            )
        summary = get_summary(
            *args,
            text=text,
            url=url,
            file_path=file_path,
            **kwargs
            )
        return summary
    metadata_path = get_metadata_path(
        video_url=video_url,
        video_path=video_path
        )
    if not os.path.isfile(metadata_path):
        safe_dump_to_file(data={},file_path=metadata_path)
    metadata = safe_read_from_json(metadata_path)
    summary = metadata.get('summary')
    if not summary:
        summary = get_summary(
            *args,
            text=text,
            audio_path=audio_path,
            video_url=video_url,
            video_path=video_path,
            **kwargs
            )
        metadata['summary']=summary

        
        safe_dump_to_file(
            data=metadata,
            file_path=metadata_path
            )
    return summary
    


