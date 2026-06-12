from .zengineer.api_node import ZEngineer
from .zengineer.local_nodes import (
    ZEngineerCLIPLoader,
    ZEngineerCLIPLoaderGGUF,
    ZEngineerEnhance,
)

NODE_CLASS_MAPPINGS = {
    "ZEngineerCLIPLoader": ZEngineerCLIPLoader,
    "ZEngineerCLIPLoaderGGUF": ZEngineerCLIPLoaderGGUF,
    "ZEngineerEnhance": ZEngineerEnhance,
    "ZEngineer": ZEngineer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZEngineerCLIPLoader": "Z-Engineer CLIP Loader (Safetensors / Shards)",
    "ZEngineerCLIPLoaderGGUF": "Z-Engineer CLIP Loader (GGUF)",
    "ZEngineerEnhance": "Z-Engineer Prompt Enhancer (Local)",
    "ZEngineer": "Z-Engineer Prompt Enhancer (API)",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
