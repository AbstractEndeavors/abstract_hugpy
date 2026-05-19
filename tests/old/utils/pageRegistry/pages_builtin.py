# pages_builtin.py
from .pages_schema import FieldSpec, PageSpec
from .pages_registry import register_page

HUGPY_PREFIX = "/hugpy_bp"   # explicit; matches how you mount the blueprint

register_page(PageSpec(
    key="summarizer/summarize",
    title="Summarize Text",
    category="summarizer",
    endpoint=f"{HUGPY_PREFIX}/summarizer/summarize",
    description="Run text through a summarizer backend with an optional preset.",
    fields=(
        FieldSpec("text", "Text", kind="textarea", required=True),
        FieldSpec("backend", "Backend", kind="select",
                  default="t5", choices=("t5", "flan", "falconsai")),
        FieldSpec("preset", "Preset", kind="select",
                  default="default",
                  choices=("default", "article", "brief", "headline")),
        FieldSpec("max_length", "Max length", kind="number", default=512),
    ),
))

register_page(PageSpec(
    key="keybert/refine_keywords",
    title="Refine Keywords",
    category="keybert",
    endpoint=f"{HUGPY_PREFIX}/keybert/refine_keywords",
    fields=(
        FieldSpec("text", "Text", kind="textarea", required=True),
        FieldSpec("preset", "Preset", kind="select",
                  default="seo",
                  choices=("default", "seo", "metadata", "social", "long_tail")),
    ),
))

register_page(PageSpec(
    key="deepcoder/image",
    title="DeepCoder Image Analysis",
    category="deepcoder",
    endpoint=f"{HUGPY_PREFIX}/deepcoder/image/upload",
    is_upload=True,
    fields=(
        FieldSpec("files", "Image(s)", kind="files", required=True),
        FieldSpec("prompt", "Prompt", kind="textarea",
                  default="Please describe this image."),
        FieldSpec("max_new_tokens", "Max new tokens", kind="number", default=1000),
    ),
))

register_page(PageSpec(
    key="videos/get_all",
    title="Process Video",
    category="videos",
    endpoint=f"{HUGPY_PREFIX}/videos/get_all",
    is_upload=True,
    fields=(
        FieldSpec("files", "Video file", kind="files"),
        FieldSpec("url", "Or URL", kind="text"),
        FieldSpec("force_refresh", "Force refresh", kind="checkbox", default=False),
    ),
))
