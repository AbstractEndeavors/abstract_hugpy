# src/thumbnail_utils.py
from .imports import (
    os,
    re,
    cv2,
    Image,
    pytesseract,
    VideoFileClip
    )

def ocr_image(image_path: str) -> str:
    """Extract text from an image using Tesseract OCR."""
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
    except Exception as e:
        return ""
    return clean_text(text)

def clean_text(text: str) -> str:
    """Normalize OCR text."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text

def extract_video_frames_unique(video_path, directory, num_frames=10, min_distance=30, prefix="frame"):
    """Extract unique frames from a video."""
    os.makedirs(directory, exist_ok=True)
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // num_frames)

    last_frame = None
    extracted = 0
    frame_idx = 0

    while cap.isOpened() and extracted < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            if last_frame is None or cv2.norm(frame, last_frame, cv2.NORM_L2) > min_distance:
                frame_name = f"{prefix}_{frame_idx}.jpg"
                frame_path = os.path.join(directory, frame_name)
                cv2.imwrite(frame_path, frame)
                frames.append(frame_path)
                extracted += 1
                last_frame = frame

        frame_idx += 1

    cap.release()
    return frames


