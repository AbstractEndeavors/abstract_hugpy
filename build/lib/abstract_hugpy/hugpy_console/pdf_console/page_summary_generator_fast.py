#!/usr/bin/env python3
"""
Fast Page Summary Generator (Manifest-Integrated)
-------------------------------------------------
Summarizes each PDF page text (~150 words, SEO-optimized)
and appends the results into the patent's manifest.json.

Structure created:
    US_20180013491/
        page_001.txt
        page_002.txt
        summaries/
            page_001.json
            page_002.json
        manifest.json
"""

import os, re, json, multiprocessing as mp
from pathlib import Path
from transformers import pipeline
from tqdm import tqdm

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------
BASE_DIR = Path("/mnt/24T/media/thedailydialectics/pdfs2/wipow")
MODEL_NAME = "t5-small"       # or "t5-base", "bart-large-cnn" for better quality
SUMMARY_WORDS = 150
MIN_W, MAX_W = 90, 160
CHARS_LIMIT = 2000
SUMMARY_DIR = "summaries"
MANIFEST_NAME = "manifest.json"
N_PROCESSES = max(1, os.cpu_count() // 2)

# -----------------------------------------------------
# Utility Functions
# -----------------------------------------------------
def safe_load_json(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def safe_write_json(path: Path, data: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.,;:!?'\-\(\)]", "", text)
    return text.strip()

def truncate_text(text: str, max_chars: int = CHARS_LIMIT):
    return text[:max_chars]

# -----------------------------------------------------
# Summarizer (lazy global)
# -----------------------------------------------------
_summarizer = None
def get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline("summarization", model=MODEL_NAME)
    return _summarizer

# -----------------------------------------------------
# Core summarization logic
# -----------------------------------------------------
def summarize_text(text: str) -> str:
    summarizer = get_summarizer()
    prompt = "summarize: " + text
    result = summarizer(prompt, min_length=MIN_W, max_length=MAX_W)
    summary = result[0]["summary_text"].strip()
    words = summary.split()
    if len(words) > SUMMARY_WORDS * 1.2:
        summary = " ".join(words[:SUMMARY_WORDS]) + "..."
    elif len(words) < SUMMARY_WORDS * 0.5:
        summary += " (short)"
    return summary

def build_seo_json(page_id: str, summary: str) -> dict:
    desc = " ".join(summary.split()[:150])
    return {
        "page_id": page_id,
        "title": f"{page_id} | Patent Page Summary",
        "description": desc,
        "alt": f"Patent page {page_id} abstract",
        "summary": summary,
        "length_words": len(summary.split()),
    }

# -----------------------------------------------------
# Worker for each .txt page
# -----------------------------------------------------
def process_page(txt_path: Path):
    try:
        summary_dir = txt_path.parent / SUMMARY_DIR
        summary_dir.mkdir(exist_ok=True)
        out_json = summary_dir / f"{txt_path.stem}.json"
        out_txt = summary_dir / f"{txt_path.stem}.txt"

        if out_json.exists():
            return None

        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        text = clean_text(truncate_text(text))
        if len(text) < 40:
            return None

        summary = summarize_text(text)
        seo_json = build_seo_json(txt_path.stem, summary)
        safe_write_json(out_json, seo_json)
        out_txt.write_text(summary, encoding="utf-8")

        return {"id": txt_path.stem, "data": seo_json}
    except Exception as e:
        return {"id": txt_path.name, "error": str(e)}

# -----------------------------------------------------
# Manifest management
# -----------------------------------------------------
def update_manifest(patent_dir: Path, new_entries: list):
    manifest_path = patent_dir / MANIFEST_NAME
    manifest = safe_load_json(manifest_path)

    if "pages" not in manifest:
        manifest["pages"] = {}

    for entry in new_entries:
        if not entry or "error" in entry:
            continue
        pid = entry["id"]
        manifest["pages"][pid] = entry["data"]

    safe_write_json(manifest_path, manifest)

# -----------------------------------------------------
# Patent directory processing
# -----------------------------------------------------
def process_patent_dir(patent_dir: Path):
    txt_files = list(patent_dir.glob("*.txt"))
    if not txt_files:
        return

    print(f"\n📄 Processing {patent_dir.name} ({len(txt_files)} pages)...")
    results = []
    with mp.Pool(N_PROCESSES) as pool:
        for res in tqdm(pool.imap_unordered(process_page, txt_files),
                        total=len(txt_files), desc=patent_dir.name):
            if res:
                results.append(res)
    update_manifest(patent_dir, results)
    print(f"✅ Updated manifest for {patent_dir.name}")

# -----------------------------------------------------
# Main entry point
# -----------------------------------------------------
def run_page_summary_generator_fast():
    patents = [p for p in BASE_DIR.glob("US_*") if p.is_dir()]
    for p in patents:
        process_patent_dir(p)

