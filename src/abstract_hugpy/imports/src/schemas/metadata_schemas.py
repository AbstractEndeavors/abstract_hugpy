"""Model metadata enrichment. Disk first, Hub as gap-filler.

Each resolver returns a partial dict. The merger runs them in order and the
first non-None value for a field wins. `_sources` records which resolver
filled each field so you can audit later without rerunning.
"""
from .imports import Optional,List,dataclass,Any,asdict
# ---------------------------------------------------------------------------
# MetaData — One row of everything we know. All Optional — partial fills are valid.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelMetadata:
    """One row of everything we know. All Optional — partial fills are valid."""
    hub_id:                     Optional[str]       = None

    # task / framework
    pipeline_tag:               Optional[str]       = None
    library_name:               Optional[str]       = None
    auto_model_class:           Optional[str]       = None
    # peft / adapter — set only for LoRA-style adapters
    peft_type:                  Optional[str]       = None   # "LORA", etc.
    base_model:                 Optional[str]       = None   # base_model_name_or_path

    # architecture
    architectures:              Optional[List[str]] = None
    model_type:                 Optional[str]       = None
    torch_dtype:                Optional[str]       = None
    vocab_size:                 Optional[int]       = None

    # context window — keep these separate; they answer different questions
    max_position_embeddings:    Optional[int] = None   # architectural ceiling
    tokenizer_model_max_length: Optional[int] = None   # tokenizer's declared cap
    sliding_window:             Optional[int] = None
    rope_scaling:               Optional[dict] = None

    # size
    parameter_count:            Optional[int] = None

    # governance
    license:                    Optional[str] = None
    gated:                      Optional[bool] = None
    languages:                  Optional[List[str]] = None
    tags:                       Optional[List[str]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
