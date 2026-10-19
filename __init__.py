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

try:
    # ComfyUI loads custom nodes alphabetically, so ComfyUI-GGUF is normally
    # imported already; registering its FP4 (NVFP4/MXFP4) torch dequant here
    # also speeds up the stock CLIPLoaderGGUF. The Z-Engineer GGUF loader
    # registers again on use in case the load order differs.
    from .zengineer.local_nodes import _resolve_comfyui_gguf

    if _resolve_comfyui_gguf() is None:
        # Load order is not alphabetical on every OS: retry on each queued
        # prompt until ComfyUI-GGUF is found, then stop.
        from server import PromptServer

        def _register_fp4_on_prompt(json_data):
            if not getattr(_register_fp4_on_prompt, "done", False):
                _register_fp4_on_prompt.done = _resolve_comfyui_gguf() is not None
            return json_data

        PromptServer.instance.add_on_prompt_handler(_register_fp4_on_prompt)
except Exception:
    pass

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
