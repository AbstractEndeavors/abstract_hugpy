"""
Text generation manager (distilgpt2).

Singleton-loaded so repeated calls don't re-init the model.
"""

from .imports import SingletonMeta, get_transformers, get_torch


class GeneratorManager(metaclass=SingletonMeta):
    def __init__(self):
        if not hasattr(self, "_ready"):
            device = 0 if get_torch().cuda.is_available() else -1
            self.pipeline = get_transformers("pipeline")(
                "text-generation",
                model="distilgpt2",
                device=device,
            )
            self._ready = True


def get_generator():
    return GeneratorManager().pipeline
