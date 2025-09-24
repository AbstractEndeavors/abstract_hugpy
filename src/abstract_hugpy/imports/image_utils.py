from .imports import *
_logger = get_logFile('image_utils')
_logger.info('started')
# Placeholder for abstract_ocr (using pytesseract)
def ocr_image(image_path):
    return pytesseract.image_to_string(Image.open(image_path))

# Clean text function
def clean_text(text):
    for phrase in REMOVE_PHRASES:
        text = text.replace(phrase, '')
    return text.strip()
# ---------- similarity helpers ----------
def _dhash(img_bgr: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """Return a boolean 8x8 difference hash."""
    # convert to gray, resize to (hash_size+1, hash_size)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]  # shape (hash_size, hash_size) bool
    return diff

def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    """Hamming distance between two boolean hashes."""
    return int(np.count_nonzero(a ^ b))

try:
    # optional SSIM (if installed)
    from skimage.metrics import structural_similarity as ssim
    _HAS_SSIM = True
except Exception:
    _HAS_SSIM = False

def frames_similar(
    prev_frame_bgr: np.ndarray,
    curr_frame_bgr: np.ndarray,
    method: str = "dhash",
    dhash_thresh: int = 4,
    ssim_thresh: float = 0.992,
) -> bool:
    """
    True if frames are 'similar enough' to skip saving the current one.
    - dhash_thresh: max Hamming distance (0..64) to consider identical-ish
    - ssim_thresh: min SSIM (0..1) to consider too similar
    """
    if method == "ssim" and _HAS_SSIM:
        # SSIM expects gray, equal sizes
        g1 = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(curr_frame_bgr, cv2.COLOR_BGR2GRAY)
        if g1.shape != g2.shape:
            h = min(g1.shape[0], g2.shape[0])
            w = min(g1.shape[1], g2.shape[1])
            g1 = g1[:h, :w]
            g2 = g2[:h, :w]
        score = ssim(g1, g2)
        return score >= ssim_thresh

    # default: dHash
    h1 = _dhash(prev_frame_bgr)
    h2 = _dhash(curr_frame_bgr)
    return _hamming(h1, h2) <= dhash_thresh
# Extract frames from video (for OCR, if needed)
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames
# ---------- your extractor with duplicate skipping ----------
def extract_video_frames_unique(
    video_path: str,
    directory: str,
    video_id: str = None,
    every: int = 1,                # seconds between samples
    method: str = "dhash",         # "dhash" (default) or "ssim"
    dhash_thresh: int = 4,         # good starting point (0..64)
    ssim_thresh: float = 0.992,    # if method="ssim" and skimage present
    jpeg_quality: int = 95,
    min_force_gap: int = 10,       # always save at least one frame every N seconds
) -> list[str]:
    """
    Sample one frame every `every` seconds and skip near-duplicates.
    Returns list of saved file paths.
    """
    os.makedirs(directory, exist_ok=True)
    clip = VideoFileClip(video_path)
    duration = int(clip.duration or 0)
    if duration <= 0:
        return []

    if video_id is None:
        video_id = generate_file_id(video_path)

    saved = []
    last_saved_frame = None
    last_forced_t = -10**9  # large negative

    for t in range(0, duration, max(1, int(every))):
        # grab RGB frame from moviepy, convert to BGR for OpenCV
        frame_rgb = clip.get_frame(t)
        frame_bgr = cv2.cvtColor(np.asarray(frame_rgb), cv2.COLOR_RGB2BGR)

        # force-save at interval even if similar (safety net)
        force_save = (t - last_forced_t) >= int(min_force_gap)

        similar = False
        if last_saved_frame is not None and not force_save:
            try:
                similar = frames_similar(last_saved_frame, frame_bgr, method, dhash_thresh, ssim_thresh)
            except Exception as e:
                # if anything goes sideways, don't skip (fail open)
                similar = False

        if not similar or force_save:
            out_path = os.path.join(directory, f"{video_id}_frame_{t}.jpg")
            if not os.path.exists(out_path):
                cv2.imwrite(out_path, frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            saved.append(out_path)
            last_saved_frame = frame_bgr
            last_forced_t = t

    clip.close()
    return saved
def extract_unique_via_capture(
    video_path: str,
    directory: str,
    video_id: str = None,
    sample_every_sec: float = 1.0,
    method: str = "dhash",
    dhash_thresh: int = 4,
    ssim_thresh: float = 0.992,
    jpeg_quality: int = 95,
    min_force_gap: float = 10.0,
) -> list[str]:
    os.makedirs(directory, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    if video_id is None:
        video_id = generate_file_id(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0

    next_sample_ts = 0.0
    last_saved_frame = None
    last_forced_ts = -10**9
    saved = []
    frame_idx = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        if t + 1e-6 < next_sample_ts:
            frame_idx += 1
            continue

        force_save = (t - last_forced_ts) >= float(min_force_gap)
        similar = False
        if last_saved_frame is not None and not force_save:
            similar = frames_similar(last_saved_frame, frame_bgr, method, dhash_thresh, ssim_thresh)

        if not similar or force_save:
            sec = int(t + 0.5)
            out_path = os.path.join(directory, f"{video_id}_frame_{sec}.jpg")
            if not os.path.exists(out_path):
                cv2.imwrite(out_path, frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            saved.append(out_path)
            last_saved_frame = frame_bgr
            last_forced_ts = t

        next_sample_ts += float(sample_every_sec)
        frame_idx += 1

    cap.release()
    return saved
def extract_video_frames(video_path,directory,video_id=None,frame_interval=None):
    frame_interval = frame_interval or 1
    video = VideoFileClip(video_path)
    duration = video.duration
    video_id = video_id or generate_file_id(video_path)
    for t in range(0, int(duration), frame_interval):
        frame_path = os.path.join(directory,f"{video_id}_frame_{t}.jpg")
        if not os.path.isfile(frame_path):
            frame = video.get_frame(t)
            cv2.imwrite(frame_path, cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
