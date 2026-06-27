from .imports import *       # ← the missing name

_SIZE_CACHE: dict[str, tuple[float, int | None]] = {}
_SIZE_TTL = 600
_size_lock = threading.Lock()

def model_size(hub_id: str) -> int | None:
    now = time.time()
    with _size_lock:
        hit = _SIZE_CACHE.get(hub_id)
        if hit and now - hit[0] < _SIZE_TTL:
            return hit[1]
    try:
        info = hfApi.model_info(hub_id, files_metadata=True)
        total = sum(s.size for s in info.siblings if s.size) or None
    except Exception as exc:
        logger.warning("model_size(%s) failed: %s", hub_id, exc)   # don't hide it
        total = None
    with _size_lock:
        _SIZE_CACHE[hub_id] = (now, total)
    return total

def free_bytes() -> int | None:
    """Headroom on the filesystem where this repo will actually be written.

    Uses MODELS_DIR (search package's storage root) — the same root
    destination_for_model builds paths from — so fits_disk can't disagree
    with where the file goes. Deliberately NOT list_peers(): that reports a
    pydantic settings.storage_root that may point at a different mount.
    """
    try:
        probe = MODELS_DIR if os.path.exists(MODELS_DIR) else "/"
        return shutil.disk_usage(probe).free
    except OSError:
        return None
