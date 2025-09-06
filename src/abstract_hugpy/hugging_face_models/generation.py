class generatorManager:
    def __init__(self):
        self.generator = pipeline('text-generation', model='distilgpt2', device= -1)
def get_generator():
    generator_mgr = generatorManager()
    return generator_mgr.generator

# ------------------------------------------------------------------------------
# 5. UTILITY: MEDIA URL BUILDER
# ------------------------------------------------------------------------------
EXT_TO_PREFIX = {
    ".png": "images",
    ".jpg": "images",
    ".jpeg": "images",
    ".gif": "images",
    ".mp4": "videos",
    ".mp3": "audio",
    ".wav": "audio",
    ".pdf": "documents",
    # add more as needed
}


def generate_media_url(
    fs_path: str,
    domain: str = None,
    repository_dir: str = None
) -> str | None:
    """
    Convert a local filesystem path (fs_path) inside repository_dir into a public URL.
    E.g., if domain="https://example.com", repository_dir="/home/user/repo",
    and fs_path="/home/user/repo/assets/img.png",
    returns "https://example.com/images/assets/img.png".

    Args:
        fs_path (str): Absolute or relative file path.
        domain (str): Base domain (including protocol), e.g. "https://mydomain.com".
        repository_dir (str): The root of the repo, so that fs_path starts with repository_dir.

    Returns:
        str | None: The constructed URL, or None if fs_path not under repository_dir.
    """
    if not repository_dir or not domain:
        return None

    fs_path_abs = os.path.abspath(fs_path)
    repo_abs = os.path.abspath(repository_dir)
    if not fs_path_abs.startswith(repo_abs):
        return None

    # Compute relative path under repository_dir
    rel_path = fs_path_abs[len(repo_abs) :].lstrip(os.sep)
    rel_path_unix = quote(rel_path.replace(os.sep, "/"))
    ext = os.path.splitext(fs_path_abs)[1].lower()
    prefix = EXT_TO_PREFIX.get(ext, "repository")

    return f"{domain.rstrip('/')}/{prefix}/{rel_path_unix}"

