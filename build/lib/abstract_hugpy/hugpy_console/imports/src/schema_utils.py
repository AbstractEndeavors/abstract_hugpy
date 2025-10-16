from .constants import *
from .info_utils import *
def expand_schema(
    video_id: str,
    root_dir: str=None,
    folder: str=None,
    video_path: str =None,
    schema=None,
    video_url=None,
    flat_layout: bool = False
    ) -> dict:
    """
    Expand VIDEO_SCHEMA into concrete paths (recursively).
    - Replaces {video_id} placeholder with the actual ID.
    - If flat_layout=True, do NOT create a <video_id> subdir.
    """
    schema = schema or VIDEO_SCHEMA
    root_dir = root_dir or VIDEOS_ROOT_DEFAULT
    folder = folder or os.path.join(root_dir, video_id)
    os.makedirs(folder, exist_ok=True)
    result = {}
    for key, rel in schema.items():
        if isinstance(rel, dict):
            dirname = key.replace("_dir", "").replace("_directory", "")
            # If flat, stay in current folder
            subfolder = folder if flat_layout else os.path.join(folder, dirname)
            check_create_logs(f"making this directory == {subfolder} == expand_schema line 323")
            os.makedirs(subfolder, exist_ok=True)
            result[f"{dirname}_directory"] = subfolder
            result[key] = expand_schema(video_id=video_id, folder=subfolder, schema=rel, flat_layout=flat_layout)
        elif isinstance(rel, str):
            rel = rel.format(video_id=video_id, i="{i}")
            path = os.path.join(folder, rel)
            if key.endswith("_dir"):
                os.makedirs(path, exist_ok=True)
            result[key] = path
    return result

def ensure_standard_paths(
    info: dict,
    video_id: str,
    root_dir: str=None,
    folder: str=None,
    video_path: str =None,
    schema=None,
    video_url=None,
    flat_layout: bool = False
    ) -> dict:
    """
    Ensure standard paths exist inside <video_root>/ or <video_root>/<video_id>/.
    Controlled by flat_layout.
    """
    video_id = video_id or info.get("video_id") or info.get("id")
    if not video_id:
        if video_url:
             video_id = get_video_id(video_url)
        if not video_id:
            return info
    flat_layout = flat_layout or False
    root_dir = root_dir or VIDEOS_ROOT_DEFAULT
    folder = os.path.join(root_dir, video_id) if video_id else root_dir
    if video_path:
        folder = os.path.dirname(video_path)
        dirbase = os.path.basename(dirname)
        if flat_layout:
            root_dir = folder
        else:
            root_dir = os.path.dirname(folder)
        video_id = video_id or dirbase
    schema = schema or VIDEO_SCHEMA
    os.makedirs(folder, exist_ok=True)
    info["directory"] = folder
    schema_paths = expand_schema(
        video_id=video_id,
        root_dir=root_dir,
        folder=folder,
        video_path=video_path,
        schema=schema,
        video_url=video_url,
        flat_layout=flat_layout,
    )

    # flatten for convenience
    def flatten(d, parent_key="", sep="_"):
        flat = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                flat.update(flatten(v, new_key, sep))
            else:
                flat[new_key] = v
        return flat

    flat_paths = flatten(schema_paths)
    for k, v in flat_paths.items():
        if not info.get(k):
            info[k] = v

    info["schema_paths"] = schema_paths
    return info
