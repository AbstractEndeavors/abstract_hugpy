from .constants import *
from .info_utils import *
from .schema_utils import *
def get_video_folder(video_id, envPath=None, flat_layout: bool = False):
    """Return the canonical video folder (with optional flat layout)."""
    root = get_video_directory(envPath=envPath)
    dir_path = root if flat_layout else os.path.join(root, video_id)
    check_create_logs(f"making this directory == {dir_path} == ensure_standard_paths line 377")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def get_video_paths(video_id, envPath=None, flat_layout: bool = False):
    """Return dict of canonical paths for this video_id."""
    folder = get_video_folder(video_id, envPath=envPath, flat_layout=flat_layout)
    return expand_schema(video_id, folder, VIDEO_SCHEMA, flat_layout=flat_layout)

def get_video_env(key=None, envPath=None):
    """Pull video directory from env file or environment variables."""
    key = key or VIDEO_ENV_KEY
    return get_env_value(key=key, path=envPath)

def get_video_root(video_root=None):
    """Fallback root directory if no env override is found."""
    home = os.path.expanduser("~")
    candidates = [
        video_root,
        os.path.join(home, "videos"),
        os.path.join(home, "Videos"),
        os.path.join(home, "Downloads"),
        os.path.join(home, "downloads"),
        home,
    ]
    for directory in candidates:
        if directory and os.path.isdir(directory):
            return directory
    return home  # last resort

def get_video_directory(key=None, envPath=None):
    """Assure that a valid video directory exists and return its path."""
    video_directory = get_video_env(key=key, envPath=envPath)
    if not video_directory:
        video_directory = get_video_root()

    os.makedirs(video_directory, exist_ok=True)
    return video_directory

def get_video_folder(video_id, envPath=None):
    """Return the canonical per-video folder and ensure subdirs exist."""
    root = get_video_directory(envPath=envPath)
    dir_path = os.path.join(root, video_id)
    os.makedirs(dir_path, exist_ok=True)

    # Ensure schema directories exist
    for key, rel in VIDEO_SCHEMA.items():
        if rel.endswith("/") or "dir" in key:
            os.makedirs(os.path.join(dir_path, rel), exist_ok=True)

    return dir_path

def get_video_paths(video_id, envPath=None):
    """Return dict of canonical paths for this video_id."""
    folder = get_video_folder(video_id, envPath=envPath)
    return {key: os.path.join(folder, rel) for key, rel in VIDEO_SCHEMA.items()}

