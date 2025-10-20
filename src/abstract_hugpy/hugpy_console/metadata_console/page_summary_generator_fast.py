#!/usr/bin/env python3
"""
Fast Page Summary Generator (Env-Aware + Auto-Detect)
-----------------------------------------------------
Summarizes each PDF page text (~150 words, SEO-optimized),
compares new summaries with existing ones via DeepZero,
and updates manifest.json.

It auto-detects a working base directory based on:
    - Environment variables
    - Host-specific defaults
    - Fallback to CWD

Usage
-----
    from abstract_hugpy.hugpy_console.metadata_console.page_summary_generator_fast import run_page_summary_generator_fast
    run_page_summary_generator_fast()              # auto-detects base_dir
    run_page_summary_generator_fast("/custom/path")
"""


from ..imports import *
from .metadata_utils import scan_matadata_from_pdf_dirs
from .summary_judge import SummaryJudge

# ------------------------------------------------------------------
# Environment keys (override in ~/.bashrc, .env, or systemd)
# ------------------------------------------------------------------
BASE_DIR_KEY_PROD   = "HUGPY_BASE_DIR_SUMMARY_GENERATOR_FAST_PROD"
BASE_DIR_KEY_SERVER = "HUGPY_BASE_DIR_SUMMARY_GENERATOR_FAST_SERVER"
BASE_DIR_KEY_LOCAL  = "HUGPY_BASE_DIR_SUMMARY_GENERATOR_FAST_LOCAL"

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
SUMMARY_WORDS  = 150
CHARS_LIMIT    = 2000
SUMMARY_DIR    = "summaries"
MANIFEST_NAME  = "manifest.json"
N_PROCESSES    = max(1, os.cpu_count() // 2)
judge          = SummaryJudge()

# ------------------------------------------------------------------
# Environment detection helpers
# ------------------------------------------------------------------
def get_env_value_or_none(key: str, path: str | None = None) -> str | None:
    """Pull a variable from env file or system environment."""
    try:
        val = get_env_value(key=key, path=path)
        return val if val and os.path.exists(val) else None
    except Exception:
        return None

def get_env_basedirs(env_path: str | None = None) -> list[Path]:
    """Return list of candidate base paths in order of preference."""
    prod   = get_env_value_or_none(BASE_DIR_KEY_PROD, env_path)
    server = get_env_value_or_none(BASE_DIR_KEY_SERVER, env_path)
    local  = get_env_value_or_none(BASE_DIR_KEY_LOCAL, env_path)
    return [p for p in [prod, server, local] if p]

def detect_base_dir(env_path: str | None = None) -> Path:
    """Choose a valid base directory using environment + defaults."""
    for path_str in get_env_basedirs(env_path):
        p = Path(path_str)
        if p.exists():
            return p

    # Fallback defaults
    candidates = [
        Path("/mnt/24T/media/thedailydialectics/pdfs"),
        Path("/var/www/media/thedailydialectics/pdfs"),
        Path.home() / "Documents/pythonTools/data/pdfs",
        Path.cwd(),
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path.cwd()

# ------------------------------------------------------------------
# Core utilities
# ------------------------------------------------------------------
def truncate_text(text: str, max_chars: int = CHARS_LIMIT) -> str:
    return text[:max_chars]

def summarize_text(text: str) -> str:
    try:
        summary = get_summarizer_summary(
            text=text,
            summary_mode="medium",
            max_chunk_tokens=200,
            summary_words=SUMMARY_WORDS,
        ).strip()
        if len(summary.split()) < 30:
            summary += " (short)"
        return summary
    except Exception as e:
        return f"[Summarizer error: {e}]"

def build_seo_json(page_id: str, summary: str) -> dict:
    desc = " ".join(summary.split()[:SUMMARY_WORDS])
    return {
        "page_id": page_id,
        "title": f"{page_id} | Patent Page Summary",
        "description": desc,
        "alt": f"Patent page {page_id} abstract",
        "summary": summary,
        "length_words": len(summary.split()),
    }

# ------------------------------------------------------------------
# Page processing
# ------------------------------------------------------------------
def process_page(txt_path: Path):
    try:
        summary_dir = txt_path.parent / SUMMARY_DIR
        summary_dir.mkdir(exist_ok=True)
        out_json = summary_dir / f"{txt_path.stem}.json"
        out_txt  = summary_dir / f"{txt_path.stem}.txt"

        text = clean_text(truncate_text(txt_path.read_text(encoding="utf-8", errors="ignore")))
        if len(text) < 40:
            return None

        new_summary = summarize_text(text)
        seo_json    = build_seo_json(txt_path.stem, new_summary)

        # Compare or create
        if out_json.exists():
            existing = safe_load_json(out_json)
            old_summary = existing.get("summary") or existing.get("text") or ""
            best_summary, best_score, other_score = judge.compare(text, new_summary, old_summary)
            if best_summary == new_summary:
                safe_write_json(out_json, seo_json)
                out_txt.write_text(new_summary, encoding="utf-8")
                action = "replaced"
            else:
                action = "kept_old"
        else:
            safe_write_json(out_json, seo_json)
            out_txt.write_text(new_summary, encoding="utf-8")
            best_score, other_score, action = 1.0, 0.0, "new"

        return {"id": txt_path.stem, "action": action,
                "best_score": best_score, "other_score": other_score}

    except Exception as e:
        return {"id": txt_path.name, "error": str(e)}

# ------------------------------------------------------------------
# Manifest update
# ------------------------------------------------------------------
def update_manifest(patent_dir: Path, new_entries: list):
    manifest_path = patent_dir / MANIFEST_NAME
    manifest = safe_load_json(manifest_path)
    manifest.setdefault("pages", {})

    for entry in new_entries:
        if not entry or "error" in entry:
            continue
        manifest["pages"][entry["id"]] = entry.get("data", {})

    safe_write_json(manifest_path, manifest)

# ------------------------------------------------------------------
# Directory processing
# ------------------------------------------------------------------
def process_patent_dir(patent_dir: Path):
    txt_files = list(patent_dir.glob("*.txt"))
    if not txt_files:
        scan_matadata_from_pdf_dirs([patent_dir],output_dir=patent_dir)
        txt_files = list(patent_dir.glob("*.txt"))
        if not txt_files:
            return 
    print(f"\n📄 Processing {patent_dir.name} ({len(txt_files)} pages)...")
    results = []
    with mproc.Pool(N_PROCESSES) as pool:
        for res in tqdm(pool.imap_unordered(process_page, txt_files),
                        total=len(txt_files), desc=patent_dir.name):
            if res:
                results.append(res)
    update_manifest(patent_dir, results)
    print(f"✅ Updated manifest for {patent_dir.name}")

# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------
def run_page_summary_generator_fast(base_dir: str | Path = None, env_path: str | None = None):
    base_dir = Path(base_dir) if base_dir else detect_base_dir(env_path)
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    pdf_dirs = [p for p in get_files_and_dirs(str(BASE_DIR),allowed_exts=['.pdf'])[-1] if '_page_' not in p]
    if not pdf_dirs:
        print(f"⚠️ No subdirectories found in {base_dir}")
        return

    print(f"🏗 Using base directory: {base_dir}")
    for p in patents:
        directory = p
        if os.path.isfile(p):
            directory = os.path.dirname(p)
        process_patent_dir(Path(directory))
    print("🏁 All summaries complete.")


