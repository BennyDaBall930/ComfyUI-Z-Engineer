from .zengineer.api_node import ZEngineer
from .zengineer.lfm_local import (
    ZEngineerEnhanceLFM,
    ZEngineerLFMLoader,
)
from .zengineer.local_nodes import (
    ZEngineerCLIPLoader,
    ZEngineerCLIPLoaderGGUF,
    ZEngineerEnhance,
)

NODE_CLASS_MAPPINGS = {
    "ZEngineerCLIPLoader": ZEngineerCLIPLoader,
    "ZEngineerCLIPLoaderGGUF": ZEngineerCLIPLoaderGGUF,
    "ZEngineerEnhance": ZEngineerEnhance,
    "ZEngineerLFMLoader": ZEngineerLFMLoader,
    "ZEngineerEnhanceLFM": ZEngineerEnhanceLFM,
    "ZEngineer": ZEngineer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZEngineerCLIPLoader": "Z-Engineer CLIP Loader (Safetensors / Shards)",
    "ZEngineerCLIPLoaderGGUF": "Z-Engineer CLIP Loader (GGUF)",
    "ZEngineerEnhance": "Z-Engineer Prompt Enhancer (Local)",
    "ZEngineerLFMLoader": "Z-Engineer LFM2.5 Enhancer Loader (GGUF / Safetensors)",
    "ZEngineerEnhanceLFM": "Z-Engineer Prompt Enhancer (LFM2.5 Local)",
    "ZEngineer": "Z-Engineer Prompt Enhancer (API)",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
