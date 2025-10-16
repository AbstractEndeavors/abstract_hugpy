
class infoRegistrys(metaclass=SingletonMeta):
    """
    Videos go under <videos_root>/<video_id>/...
    Registry (registry.json) goes under <documents_root>/
    """

    def __init__(self, video_root=None, documents_root=None, flat_layout: bool = False, **kwargs):
        if not hasattr(self, 'initialized'):
            self.initialized   = True
            self.videos_root   = get_videos_root(videos_root)
            self.documents_root= get_documents_root(documents_root)
            self.flat_layout   = flat_layout

            os.makedirs(self.videos_root, exist_ok=True)
            os.makedirs(self.documents_root, exist_ok=True)

            # Registry lives with documents
            self.registry_path = os.path.join(self.documents_root, "registry.json")

            self.default = {
                "by_url": {}, "by_id": {}, "by_path": {},
                "by_doc": {}, "by_data_url": {}
            }
            self.registry = dict(self.default)
            self._rwlock = fasteners.ReaderWriterLock()
            self._load_registry()
    # ---------------- load / save ----------------
    def _save_registry_unlocked(self):
        tmp = f"{self.registry_path}.tmp"
        safe_dump_to_file(self.registry, tmp)
        with open(self.registry_path, "wb") as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            with open(tmp, "rb") as src:
                f.write(src.read())
            f.flush()
            os.fsync(f.fileno())
            portalocker.unlock(f)
        os.remove(tmp)

    def _save_registry(self):
        with self._rwlock.write_lock():
            self._save_registry_unlocked()


    def _load_registry(self):
        with self._rwlock.write_lock():
            try:
                self.registry = load_json_keys(
                    filepath=self.registry_path,
                    main_data=self.registry,
                    default=self.default
                )
            except Exception:
                pass


    # ---------------- pruning ----------------
    def prune_registry(self, dry_run: bool = False):
        removed = []
        with self._rwlock.write_lock():
            to_delete = []
            for vid, meta in list(self.registry["by_id"].items()):
                vpath = meta.get("video_path")
                url = meta.get("url", "")
                if not vpath or vpath.endswith(".NA") or not os.path.exists(vpath):
                    to_delete.append(vid)
                if url.endswith("/watch") and "v=" not in url:
                    to_delete.append(vid)

            for vid in set(to_delete):
                removed.append(vid)
                self.registry["by_id"].pop(vid, None)
                self.registry["by_url"] = {u: v for u, v in self.registry["by_url"].items() if v != vid}
                self.registry["by_path"] = {p: v for p, v in self.registry["by_path"].items() if v != vid}
                self.registry["by_doc"] = {p: v for p, v in self.registry["by_doc"].items() if v != vid}
                self.registry["by_data_url"] = {p: v for p, v in self.registry["by_data_url"].items() if v != vid}

            if not dry_run:
                self._save_registry_unlocked()

        if removed:
            logger.info(f"[infoRegistry] Pruned {len(removed)} invalid entries: {removed}")
        return removed

    # ---------------- internal save helpers ----------------
    def _save_registry_unlocked(self):
        # Caller MUST hold write_lock already
        get_atomic_write(self.registry_path, self.registry)

    def _save_registry(self):
        # Public save: acquires write lock and persists
        with self._rwlock.write_lock():
            self._save_registry_unlocked()



    # ---------------- cache helpers ----------------
    def _read_cached_info(self, video_id: str) -> dict | None:
        # Fallback location when we only know the id
        cache_dir = self.videos_root if self.flat_layout else os.path.join(self.videos_root, video_id)
        cache = os.path.join(cache_dir, "info.json")
        if os.path.isfile(cache):
            try:
                with open(cache, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _write_cached_info(self, video_id: str, info: dict) -> str:
        # Prefer the actual folder that already holds the video
        cache_dir = info.get("directory") or (
            self.videos_root if self.flat_layout else os.path.join(self.videos_root, video_id)
        )
        os.makedirs(cache_dir, exist_ok=True)
        cache = os.path.join(cache_dir, "info.json")
        get_atomic_write(cache, info)
        return cache

   # ---------------- registry maps ----------------
    # ---------------- registry maps ----------------
    def _resolve_video_id(self, url: str | None, video_path: str | None,
                          hint_id: str | None, video_url: str | None = None,
                          document_path: str | None = None,
                          data_url: str | None = None) -> str | None:
        url = get_video_url(url or video_url)
        with self._rwlock.read_lock():
            if hint_id:
                return hint_id
            if video_path and video_path in self.registry["by_path"]:
                return self.registry["by_path"][video_path]
            if document_path and document_path in self.registry["by_doc"]:
                return self.registry["by_doc"][document_path]
            if data_url and data_url in self.registry["by_data_url"]:
                return self.registry["by_data_url"][data_url]
            if url and url in self.registry["by_url"]:
                return self.registry["by_url"][url]
        return None

    def _link_unlocked(self, video_id: str, url: str | None, video_path: str | None,
                       video_url: str | None = None, document_path: str | None = None,
                       data_url: str | None = None):
        url = get_video_url(url or video_url)
        if url:
            self.registry["by_url"][url] = video_id
        if video_path:
            self.registry["by_path"][video_path] = video_id
        if document_path:
            self.registry["by_doc"][document_path] = video_id
        if data_url:
            self.registry["by_data_url"][data_url] = video_id

        rec = self.registry["by_id"].get(video_id, {})
        if url:
            rec["url"] = url
        if video_path:
            rec["video_path"] = video_path
        if document_path:
            rec["document_path"] = document_path
        if data_url:
            rec["data_url"] = data_url
        rec["timestamp"] = time.time()

        self.registry["by_id"][video_id] = rec
        self._save_registry_unlocked()

    def _link(self, video_id: str, url: str | None, video_path: str | None,
              video_url: str | None = None, document_path: str | None = None,
              data_url: str | None = None):
        with self._rwlock.write_lock():
            self._link_unlocked(video_id, url, video_path, video_url, document_path, data_url)
    def add_file(self,
                 file_path: str,
                 url: str | None = None,
                 video_id: str | None = None,
                 save_file: str | None = None,
                 data_url: str | None = None) -> dict:
        """
        Register a new file (document, dataset, etc.) in the registry.

        Args:
            file_path: Path to the file to register.
            url: Optional source URL associated with the file.
            video_id: Optional fixed ID; if not provided, will be generated.
            data_url: Optional inline data identifier (e.g. data: URI).

        Returns:
            dict containing the file's registry metadata.
        """
        
        if not os.path.isfile(save_file):
            raise FileNotFoundError(f"No such file: {file_path}")
        
        # Generate stable ID from file name if not given
        vid = video_id or generate_file_id(file_path)

        # Build metadata
        info = {
            "video_id": vid,
            "file_path": os.path.abspath(file_path),
            "document_path": file_path,
            "url": url,
            "data_url": data_url,
            "timestamp": time.time(),
            "size": os.path.getsize(file_path),
            "mtime": os.path.getmtime(file_path),
        }

        # Write info.json next to file
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)
        info_path = os.path.join(file_dir, "info.json")
        safe_dump_to_file(info, info_path)
        info["info_path"] = info_path

        # Update registry maps
        self._link(vid, url, None,
                   document_path=file_path,
                   data_url=data_url)

        return info
    # ---------------- public API ----------------
    def edit_info(self, data: dict, url: str | None = None,
                  video_id: str | None = None, video_path: str | None = None, video_url: str | None = None):
        url = get_video_url(url or video_url)
        with self._rwlock.write_lock():
            cur = self.get_video_info(url=url, video_id=video_id, video_path=video_path, force_refresh=False)
            if not cur:
                raise RuntimeError("No existing info to edit")

            cur.update(data or {})

            canonical = (
                cur.get("video_id") or cur.get("id") or video_id or (get_sha12(url) if url else None)
            ) or generate_video_id(video_path or "video")
            cur["video_id"] = canonical

            # Prefer the real video directory
            video_dir = (
                cur.get("directory")
                or (os.path.dirname(os.path.abspath(cur.get("file_path"))) if cur.get("file_path") else None)
                or (os.path.dirname(os.path.abspath(video_path)) if video_path else None)
            )

            if video_dir:
                cur = ensure_standard_paths(cur, video_dir, flat_layout=True)
                cache_path = os.path.join(video_dir, "info.json")
            else:
                base = self.videos_root if self.flat_layout else os.path.join(self.videos_root, canonical)
                os.makedirs(base, exist_ok=True)
                cur = ensure_standard_paths(cur, base, flat_layout=self.flat_layout)
                cache_path = os.path.join(base, "info.json")

            get_atomic_write(cache_path, cur)
            self._link_unlocked(canonical, url, cur.get("file_path") or video_path)
            cur["info_path"] = cache_path
            return cur
    def get_video_info(self, url: str | None = None, video_id: str | None = None,
                       force_refresh: bool = False, video_path: str | None = None,
                       download=False,video_url: str | None=None) -> dict | None:
        # This method does a mix of reads and potential writes.
        # Keep lock scopes minimal and use write locks only when mutating.
        url = get_video_url(url or video_url)
        # prune mutates → write lock inside prune_registry
        self.prune_registry(dry_run=False)

        url = get_video_url(url)
        if url and "youtube.com/watch" in url and "v=" not in url:
            logger.debug(f"[infoRegistry] Ignoring bare watch URL: {url}")
            return None

        # If a real file is provided, construct info immediately (no registry lookup needed yet)
        if video_path and os.path.isfile(video_path):
            vid = video_id or generate_video_id(video_path)
            info = make_video_info(video_path)
            cache = self._write_cached_info(vid, info)
            # registry mutation
            self._link(vid, url, os.path.abspath(video_path))
            info["info_path"] = cache
            info["video_id"] = vid
            return ensure_standard_paths(info, self.videos_root)

        # Check registry for an existing id
        vid = self._resolve_video_id(url, video_path, video_id)

        if vid and not force_refresh:
            cached = self._read_cached_info(vid)  # disk read; no mutation
            if cached:
                # registry mutation to refresh mappings/timestamp
                self._link(vid, url, cached.get("file_path"))
                cached["info_path"] = os.path.join(self.videos_root, vid, "info.json")
                cached["video_id"] = vid
                return ensure_standard_paths(cached, self.videos_root)

        if url:
            info = get_yt_dlp_info(url)
            if info:
                vid = info.get("id") or get_sha12(url)
                cache = self._write_cached_info(vid, info)
                # registry mutation
                self._link(vid, url, None)
                info["info_path"] = cache
                info["video_id"] = vid
                return ensure_standard_paths(info, self.videos_root)

        return None

    # ---------------- public API ----------------
    def list_cached_items(self):
        with self._rwlock.read_lock():
            return [
                {
                    "video_id": vid,
                    "url": meta.get("url"),
                    "video_path": meta.get("video_path"),
                    "document_path": meta.get("document_path"),
                    "data_url": meta.get("data_url"),
                    "timestamp": meta.get("timestamp"),
                }
                for vid, meta in self.registry["by_id"].items()
            ]
