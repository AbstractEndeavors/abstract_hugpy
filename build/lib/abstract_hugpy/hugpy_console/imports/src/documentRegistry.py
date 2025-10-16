from .imports import *
from .constants import *
from .schema_utils import *
from .info_utils import *
from .directory_utils import *
from .download_utils import *



def generate_file_id(path: str, max_length: int = 50) -> str:
    """
    Generate a normalized, filesystem-safe identifier from a file path.

    - Strips extension, normalizes Unicode to ASCII, lowers case.
    - Replaces non-alphanumeric characters with hyphens.
    - Collapses duplicate hyphens.
    - Truncates and appends a hash suffix if longer than `max_length`.

    Args:
        path (str): Input file path or string to normalize.
        max_length (int): Maximum allowed length of the identifier.

    Returns:
        str: A normalized file identifier.
    """
    base = os.path.splitext(os.path.basename(path))[0]
    base = unicodedata.normalize("NFKD", base).encode("ascii", "ignore").decode("ascii")
    base = base.lower()
    base = re.sub(r"[^a-z0-9]+", "-", base).strip("-")
    base = re.sub(r"-{2,}", "-", base)

    if len(base) > max_length:
        h = hashlib.sha1(base.encode()).hexdigest()[:8]
        base = base[: max_length - len(h) - 1].rstrip("-") + "-" + h

    return base


def clean_text(text: str) -> str:
    """
    Normalize text content for storage or analysis.

    - Collapses whitespace into single spaces.
    - Removes unwanted characters except alphanumerics, spaces,
      and basic punctuation (: , . -).
    - Strips leading/trailing whitespace.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s:.,-]", "", text)
    return text.strip()
def ensure_standard_paths(info: dict, base_dir: str, schema: dict=None,video_path=None) -> dict:
    """
    Ensure all paths in `schema` are realized under `base_dir`.

    - info: dict with partial metadata (video_id, url, etc.)
    - base_dir: canonical directory to store files
    - schema: dict defining standard keys + default filenames

    Returns updated `info` with concrete absolute paths.
    """

    os.makedirs(base_dir, exist_ok=True)
    schema = schema or VIDEO_SCHEMA
    for key, value in schema.items():
        # Nested schema (directories)
        if isinstance(value, dict):
            sub_dir_name = key
            sub_dir_path = os.path.join(base_dir, sub_dir_name)
            os.makedirs(sub_dir_path, exist_ok=True)

            # Recursively resolve
            sub_schema = value
            sub_info = info.get(key, {})
            info[key] = ensure_standard_paths(sub_info, sub_dir_path, sub_schema)
            continue

        # File pattern placeholder
        if isinstance(value, str) and "{video_id}" in value:
            vid = info.get("video_id") or "unknown"
            info[key] = os.path.join(base_dir, value.format(video_id=vid, i="{i}"))
            continue

        # If it's a relative filename, join it
        if isinstance(value, str) and not os.path.isabs(value):
            path = os.path.join(base_dir, value)
            info.setdefault(key, path)
        else:
            info.setdefault(key, value)

    # Ensure info.json exists if schema defines it
    if "info_path" in schema:
        info_path = info.get("info_path")
        if info_path and not os.path.isfile(info_path):
            safe_dump_to_file(data=info, file_path=info_path)

    return info

class DataRegistry:
    def __init__(self, root="/mnt/24T/data_registry"):
        self.root = root
        self.by_id = {}

    def make_id(self, url=None, file_path=None):
        base = url or file_path or get_time_now_iso()
        return generate_file_id(base)

    def get_info(self, url=None, file_path=None, data_id=None):
        data_id = data_id or self.make_id(url, file_path)
        directory = make_dirs(self.root, data_id)
        info_path = os.path.join(directory, "info.json")

        if os.path.isfile(info_path):
            info = safe_read_from_json(info_path)
            
        else:
            info = {"data_id": data_id, "url": url, "file_path": file_path}
            info = ensure_standard_paths(info, directory, schema=DATA_SCHEMA)
            safe_dump_to_file(data=info, file_path=info_path)

        self.by_id[data_id] = info
        return info


dataRegistry = DataRegistry()
