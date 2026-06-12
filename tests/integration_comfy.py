"""Headless integration test. Run with the ComfyUI venv from the ComfyUI root:

    cd <ComfyUI root>
    .venv\\Scripts\\python.exe <repo>/tests/integration_comfy.py [--stage gguf|shards|fallback|all]

Loads the Z-Image-Engineer model as CLIP (GGUF via ComfyUI-GGUF, sharded
safetensors, and the no-ComfyUI-GGUF fallback), runs a short enhancement
generation, and encodes a prompt to conditioning. Prints VRAM usage.
"""

import argparse
import gc
import importlib
import importlib.util
import os
import sys

COMFY_ROOT = os.environ.get("COMFY_ROOT", r"D:\AI\ComfyUI-NVIDIA-ObjectLab")
NODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, COMFY_ROOT)
os.chdir(COMFY_ROOT)

import torch  # noqa: E402

import comfy.model_management  # noqa: E402
import folder_paths  # noqa: E402  (must come after sys.path insert)


def load_package(name, path):
    init_py = os.path.join(path, "__init__.py")
    spec = importlib.util.spec_from_file_location(name, init_py, submodule_search_locations=[path])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def vram(tag):
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"[vram] {tag}: {(total - free) / 2**30:.2f} GiB used / {total / 2**30:.2f} GiB total (device-wide)")


def unload():
    comfy.model_management.unload_all_models()
    gc.collect()
    comfy.model_management.soft_empty_cache(True)


SEED_PROMPT = "a red umbrella on a wet sidewalk at night, neon reflections"


def run_clip_checks(clip, tag):
    from zengineer_pkg import prompt_utils

    # 1) text-encoder path (what CLIPTextEncode does)
    tokens = clip.tokenize("a red umbrella on a wet sidewalk")
    cond = clip.encode_from_tokens_scheduled(tokens)
    tensor = cond[0][0]
    print(f"[{tag}] conditioning shape: {tuple(tensor.shape)} dtype={tensor.dtype}")
    assert tensor.shape[-1] == 2560, "unexpected hidden size for Qwen3-4B"

    # 2) enhancer path
    local_nodes = sys.modules["zengineer_pkg.local_nodes"]
    node = local_nodes.ZEngineerEnhance()
    out = node.enhance(
        clip=clip,
        input_prompt=SEED_PROMPT,
        system_prompt=prompt_utils.V6_SYSTEM_PROMPT,
        seed=6606,
        temperature=0.2,
        top_p=0.9,
        top_k=40,
        min_p=0.03,
        repetition_penalty=1.05,
        max_tokens=96,
        enforce_seed_terms=False,
        strip_reasoning=True,
        sanitize_output=True,
        batch_mode=False,
        batch_separator="\\n---\\n",
        keep_terms="m4rty style, XJ-9_TriGGer",
    )
    text = out["result"][0]
    print(f"[{tag}] enhanced ({len(text.split())} words): {text[:300]}...")
    assert len(text.split()) > 10, "generation produced too little text"
    assert "<think>" not in text and "<|im_" not in text
    assert "m4rty style" in text and "XJ-9_TriGGer" in text, "keep_terms missing from output"
    vram(f"{tag} after generate")


def stage_gguf(local_nodes):
    print("\n=== Stage: GGUF via ComfyUI-GGUF ===")
    # make sure ComfyUI-GGUF is importable like ComfyUI would have it
    gguf_dir = os.path.join(COMFY_ROOT, "custom_nodes", "ComfyUI-GGUF")
    if "ComfyUI-GGUF" not in sys.modules and os.path.isdir(gguf_dir):
        load_package("ComfyUI-GGUF", gguf_dir)
    loader = local_nodes.ZEngineerCLIPLoaderGGUF()
    entries = local_nodes.list_gguf_entries()
    print("gguf entries:", sorted(entries.keys()))
    name = next(k for k in sorted(entries) if "Z-Image-Engineer-V6-Q4_K_M" in k)
    (clip,) = loader.load_clip(name)
    vram("gguf after load")
    run_clip_checks(clip, "gguf")
    del clip
    unload()
    vram("gguf after unload")


def stage_shards(local_nodes):
    print("\n=== Stage: sharded safetensors ===")
    loader = local_nodes.ZEngineerCLIPLoader()
    entries = local_nodes.list_safetensors_entries()
    print("safetensors entries:", sorted(entries.keys())[:10])
    name = next(k for k in sorted(entries) if k.rstrip("/").endswith("Z-Image-Engineer-V6"))
    (clip,) = loader.load_clip(name)
    vram("shards after load")
    run_clip_checks(clip, "shards")
    del clip
    unload()
    vram("shards after unload")


def stage_fallback(local_nodes):
    print("\n=== Stage: GGUF fallback (no ComfyUI-GGUF) ===")
    fallback = sys.modules["zengineer_pkg.gguf_fallback"]
    entries = local_nodes.list_gguf_entries()
    name = next(k for k in sorted(entries) if "Z-Image-Engineer-V6-Q4_K_M" in k)
    sd = fallback.load_gguf_state_dict_dequant(entries[name])
    print(f"fallback state dict: {len(sd)} tensors; embed {tuple(sd['model.embed_tokens.weight'].shape)}")
    assert "model.layers.0.self_attn.q_proj.weight" in sd
    assert sd["model.layers.0.post_attention_layernorm.weight"].shape[0] == 2560
    clip = local_nodes._load_clip_from_state_dict(sd, {})
    del sd
    vram("fallback after load")
    run_clip_checks(clip, "fallback")
    del clip
    unload()
    vram("fallback after unload")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["gguf", "shards", "fallback", "all"])
    args = parser.parse_args()

    load_package("zengineer_pkg", os.path.join(NODE_ROOT, "zengineer"))
    importlib.import_module("zengineer_pkg.local_nodes")
    importlib.import_module("zengineer_pkg.gguf_fallback")
    local_nodes = sys.modules["zengineer_pkg.local_nodes"]

    vram("baseline")
    stages = {
        "gguf": stage_gguf,
        "shards": stage_shards,
        "fallback": stage_fallback,
    }
    selected = list(stages) if args.stage == "all" else [args.stage]
    for key in selected:
        stages[key](local_nodes)

    print("\nIntegration test finished OK")


if __name__ == "__main__":
    main()
