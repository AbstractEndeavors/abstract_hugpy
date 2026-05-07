import os.path as osp
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, List, Dict, Any
from abstract_ocr.layout_ocr.ocr_utils.text_utils import convert_image_to_text
from abstract_webtools import *
from ..seo.pdf_utils import _analyze, PDFSeoReport


# ---- schemas ---------------------------------------------------------------

@dataclass(frozen=True)
class GenParams:
    """Decode-time params for deep_coder_generate. One place, sane defaults."""
    max_new_tokens: int = 100
    temperature: float = 0.6
    top_p: float = 0.95
    use_chat_template: bool = False
    messages: Optional[List[Dict[str, str]]] = None
    do_sample: bool = False

    def to_kwargs(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AnalyzePresets:
    """How _analyze is parameterized for a given scope."""
    scope: str = "full"
    summary_preset: str = "article"
    keyword_preset: str = "seo"


# ---- extractor registry ----------------------------------------------------

# An extractor turns a source (path or url) into plain text.
Extractor = Callable[[str], str]
_EXTRACTORS: dict[str, Extractor] = {}


def register_extractor(kind: str, fn: Extractor) -> None:
    if kind in _EXTRACTORS:
        raise ValueError(f"extractor {kind!r} already registered")
    _EXTRACTORS[kind] = fn


def get_extractor(kind: str) -> Extractor:
    if kind not in _EXTRACTORS:
        raise KeyError(f"unknown extractor {kind!r}; have {sorted(_EXTRACTORS)}")
    return _EXTRACTORS[kind]


# Concrete extractors. Each one is the only thing that knows how its source
# becomes text. If a new format shows up, add one extractor here, done.

register_extractor("image", convert_image_to_text)
register_extractor("audio", transcribe_file)
register_extractor("video", transcribe_file)


def _website_body_text(url: str) -> str:
    soup = BeautifulSoup(get_body_from_url(url), "html.parser")
    lines = [line for line in soup.text.split("\n") if line]
    return "\n".join(lines)

register_extractor("website", _website_body_text)


def _pdf_full_text(path: str) -> str:
    """Whole-PDF text. The page-level SEO report is a separate operation."""
    parts = []
    for page_num in range(get_num_pdf_pages(path)):
        parts.append(extract_single_pdf_page_text(pdf_path=path, page_index=page_num))
    return "\n\n".join(parts)

register_extractor("pdf", _pdf_full_text)


# ---- core operations -------------------------------------------------------

def summarize(source: str, kind: str, presets: AnalyzePresets = AnalyzePresets()) -> dict:
    """Run the SEO analyzer over text extracted from `source`."""
    text = get_extractor(kind)(source)
    report = _analyze(
        text,
        scope=presets.scope,
        summary_preset=presets.summary_preset,
        keyword_preset=presets.keyword_preset,
    )
    return report.to_dict()


def analyze(
    source: str,
    kind: str,
    prompt: str = "Please analyze the following content",
    params: GenParams = GenParams(),
) -> Any:
    """Extract text, prepend prompt, hand to deep_coder_generate."""
    text = get_extractor(kind)(source)
    full_prompt = f"{prompt}\n\n{text}"
    return deep_coder_generate(prompt=full_prompt, **params.to_kwargs())


# ---- PDF: the one operation that's actually different ----------------------

def summarize_pdf_by_page(path: str) -> dict:
    """PDF gets its own per-page summary because PDFSeoReport is page-structured."""
    report = PDFSeoReport()
    for page_num in range(get_num_pdf_pages(path)):
        text = extract_single_pdf_page_text(pdf_path=path, page_index=page_num)
        report.pages.append(
            _analyze(
                text,
                scope=f"page:{page_num}",
                summary_preset="brief",
                keyword_preset="page_seo",
            )
        )
    return report.to_dict()


def analyze_pdf_by_page(
    path: str,
    prompt: str = "Please analyze this PDF page",
    params: GenParams = GenParams(),
) -> List[Any]:
    """Generate one analysis per page. Returns a list, indexed by page number."""
    results: List[Any] = []
    for page_num in range(get_num_pdf_pages(path)):
        text = extract_single_pdf_page_text(pdf_path=path, page_index=page_num)
        page_prompt = f"{prompt} (page {page_num})\n\n{text}"
        results.append(deep_coder_generate(prompt=page_prompt, **params.to_kwargs()))
    return results


# ---- image analysis: doesn't go through text, separate path ----------------

def image_analysis(path: str, prompt: str = "Please describe this image", max_new_tokens: int = 100):
    return deepcoder_image_analysis(
        image_path=path,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )


# ---- back-compat shims -----------------------------------------------------
# Keep the old names working so nothing downstream breaks while you migrate.
# Mark them deprecated; delete in a follow-up pass.

def get_pdf_text(path):
    return [
        {"page_num": i, "text": extract_single_pdf_page_text(pdf_path=path, page_index=i)}
        for i in range(get_num_pdf_pages(path))
    ]

def summarize_pdf(path):       return summarize_pdf_by_page(path)
def analyze_pdf(path=None, prompt="Please analyze the the pdf component", **kw):
    return analyze_pdf_by_page(path, prompt=prompt, params=GenParams(**_filter_gen_kw(kw)))

def image_to_text(path):       return get_extractor("image")(path)
def summarize_image(path):      return summarize(path, "image")  # legacy: was always image-backed
def analyze_image(path=None, prompt="Please analyze the the text", **kw):
    return analyze(path, "image", prompt=prompt, params=GenParams(**_filter_gen_kw(kw)))

def video_to_text(path):       return get_extractor("video")(path)
def summarize_video(path):     return summarize(path, "video")
def analyze_video(path=None, prompt="Please analyze the the video transcription", **kw):
    return analyze(path, "video", prompt=prompt, params=GenParams(**_filter_gen_kw(kw)))

def audio_to_text(path):       return get_extractor("audio")(path)
def summarize_audio(path):     return summarize(path, "audio")
def analyze_audio(path=None, prompt="Please analyze the the audio transcription", **kw):
    return analyze(path, "audio", prompt=prompt, params=GenParams(**_filter_gen_kw(kw)))

def website_to_text(url):           return get_soup_text(url)
def website_body_to_text(url):      return get_extractor("website")(url)
def summarize_website(url):         return summarize(url, "website")
def analyze_website(url=None, prompt="Please analyze the the website", **kw):
    return analyze(url, "website", prompt=prompt, params=GenParams(**_filter_gen_kw(kw)))


_GEN_KEYS = {"max_new_tokens", "temperature", "top_p", "use_chat_template", "messages", "do_sample"}

def _filter_gen_kw(kw: dict) -> dict:
    """Drop unknown kwargs so legacy callers passing extras don't blow up GenParams."""
    return {k: v for k, v in kw.items() if k in _GEN_KEYS}

