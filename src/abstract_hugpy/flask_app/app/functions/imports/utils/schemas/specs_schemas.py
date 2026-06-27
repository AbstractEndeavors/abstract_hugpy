from .imports import *

class Runtime(str, Enum):
    transformers = "transformers"
    llama_cpp = "llama_cpp"
    dataset = "dataset"
    unknown = "unknown"


