"""
abstract_ocr.extract_pdf_text
-----------------------------
Smart PDF text extractor with OCR fallback, per-page .txt files,
and unified manifest JSON.
"""
from abstract_utilities import *
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image
import pytesseract, cv2, os, tempfile, json
from pathlib import Path
from typing import Dict, List
import hashlib

DEFAULT_LICENSE = "CC BY-SA 4.0"
DEFAULT_ATTRIBUTION = "Created by thedailydialectics for educational purposes"


def preprocess_for_ocr(image_path: str) -> Image.Image:
    """Enhance an image for OCR."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return Image.fromarray(thresh)


def get_keywords_from_text(text: str, max_len: int = 12) -> str:
    """Simple keyword collector."""
    words = [w.strip(",.()[]") for w in text.split() if len(w) > 2]
    uniq = []
    for w in words:
        if w not in uniq:
            uniq.append(w)
        if len(uniq) >= max_len:
            break
    return ",".join(uniq)


def get_file_size_mb(file_path: str) -> float:
    return round(os.path.getsize(file_path) / (1024 * 1024), 3)


def extract_pdf_pages(
    pdf_path: str,
    output_dir: str,
    domain: str = "https://thedailydialectics.com",
    subpath: str = "pdfs"
) -> List[Dict]:
    """
    Extracts text and image per page, saves .txt and .png, and
    compiles a single manifest JSON for all pages.
    """
    dir_path = os.path.dirname(pdf_path)
    rel_path=dir_path.split('/mnt/24T/media/thedailydialectics/pdfs')[1]
    pdf_path = Path(pdf_path)
    pdf_name = pdf_path.stem
    subpath = f"pdfs/{rel_path}"
    pdf_url = f"{domain}/{subpath}/{pdf_name}/{pdf_name}.pdf"

    output_dir = Path(output_dir)
    text_dir = output_dir / "text"
    img_dir = output_dir / "thumbnails"
    text_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    manifest = []

    for i, page in enumerate(doc, start=1):
        page_num_str = str(i).zfill(3)
        filename = f"{pdf_name}_page_{page_num_str}"
        img_path = img_dir / f"{filename}.png"
        txt_path = text_dir / f"{filename}.txt"

        # --- Extract text
        text = page.get_text("text").strip()
        if not text:
            # OCR fallback
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_img = os.path.join(tmpdir, f"page_{page_num_str}.png")
                images = convert_from_path(pdf_path, first_page=i, last_page=i, dpi=300)
                if images:
                    images[0].save(tmp_img, "PNG")
                    processed = preprocess_for_ocr(tmp_img)
                    text = pytesseract.image_to_string(processed, lang="eng").strip()
                    processed.save(img_path)
        else:
            # Render page as image directly
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            pix.save(img_path)

        # Save text file per page
        with open(txt_path, "w", encoding="utf-8") as tf:
            tf.write(text)

        # Metadata
        with Image.open(img_path) as im:
            width, height = im.size
        file_size = get_file_size_mb(img_path)
        keywords_str = get_keywords_from_text(text)

        meta = {
            "alt": f"{filename} | page {i} | {pdf_url}",
            "caption": f"{pdf_name}.pdf page {i}",
            "keywords_str": keywords_str,
            "filename": filename,
            "ext": ".png",
            "title": f"{pdf_name}.pdf_page_{i}",
            "dimensions": {"width": width, "height": height},
            "file_size": file_size,
            "license": DEFAULT_LICENSE,
            "attribution": DEFAULT_ATTRIBUTION,
            "longdesc": text[:8000],
            "schema": {"name": filename, "url": pdf_url},
            "social_meta": {
                "og:image": f"{domain}/{subpath}/{pdf_name}/thumbnails/{filename}.png",
                "og:image:alt": f"{domain}/{subpath}/{pdf_name}/thumbnails/{filename}.png",
                "twitter:image": f"{domain}/{subpath}/{pdf_name}/thumbnails/{filename}.png"
            },
            "text_path": str(txt_path),
            "image_path": str(img_path)
        }

        manifest.append(meta)

    doc.close()
    return manifest


def save_pdf_text_metadata(pdf_path: str, output_dir: str = None):
    """Run extraction and save manifest JSON."""
    if os.path.isdir(pdf_path):
        pdf_dir = pdf_path
        pdf_paths = [os.path.join(pdf_path,item) for item in os.listdir(pdf_path) if item.endswith('.pdf')]
        if pdf_paths:
            pdf_path = pdf_paths[0]
    
    path_parts = get_file_parts(pdf_path)
    dirbase = path_parts.get('dirbase')
    dirname = path_parts.get('dirname')
    filename = path_parts.get('filename')
    basename = path_parts.get('basename')
    if output_dir == None and dirbase != filename:
       output_dir = os.path.join(dirname,filename)
       os.makedirs(output_dir, exist_ok=True)
       shutil.move(pdf_path,output_dir)
       pdf_path = os.path.join(output_dir,basename)
    else:
        output_dir = dirname
    print(f"pdf_path = {pdf_path}")
    manifest = extract_pdf_pages(pdf_path, output_dir)
    manifest_path = os.path.join(output_dir, f"{Path(pdf_path).stem}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Extracted {len(manifest)} pages to: {output_dir}")
    print(f"📘 Manifest: {manifest_path}")
    return manifest_path
def scan_matadata_from_pdf_dirs(pdf_dirs,output_dir=None):
    if pdf_dirs and isinstance(pdf_dirs,str) and os.path.isdir(pdf_dirs):
        pdf_dirs = [os.path.join(pdf_dirs,item) for item in os.listdir(pdf_dirs) if os.path.isdir(os.path.join(pdf_dirs,item)) or item.endswith('.pdf')]
    pdf_dirs
    for pdf_dir in make_list(pdf_dirs):
        try:
            save_pdf_text_metadata(pdf_path=pdf_dir, output_dir=output_dir)
        except Exception as e:
            print(f"{e}")
