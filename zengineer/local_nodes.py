"""Local (no-LM-Studio) Z-Engineer nodes.

These nodes load the Z-Image-Engineer Qwen3-4B model directly inside
ComfyUI, either from sharded/single-file HuggingFace safetensors or from
GGUF quants, and expose it both as a Z-Image text encoder (CLIP) and as
an in-process prompt enhancer.
"""

import importlib
import json
import logging
import os
import re
import sys

import torch

import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths

from .prompt_utils import (
    V6_SYSTEM_PROMPT,
    build_chat_prompt,
    build_user_prompt,
    decode_separator,
    enforce_keep_terms,
    parse_keep_terms,
    preserve_seed_constraints,
    sanitize_prompt,
    split_batch,
)


SHARD_FILE_PATTERN = re.compile(r"^model-\d{5}-of-\d{5}\.safetensors$", re.IGNORECASE)
SHARD_INDEX_FILE = "model.safetensors.index.json"
MAX_SEED = 0xFFFFFFFFFFFFFFFF


def _model_roots(categories):
    roots = []
    seen = set()
    for category in categories:
        try:
            paths = folder_paths.get_folder_paths(category)
        except Exception:
            continue
        for root in paths:
            norm = os.path.normpath(root)
            key = norm.lower()
            if key not in seen and os.path.isdir(norm):
                seen.add(key)
                roots.append(norm)
    return roots


def _is_shard_dir(dirpath):
    try:
        names = os.listdir(dirpath)
    except OSError:
        return False
    if SHARD_INDEX_FILE in names:
        return True
    return any(SHARD_FILE_PATTERN.match(name) for name in names)


def list_safetensors_entries():
    """Map display name -> absolute path for single .safetensors files and
    sharded HuggingFace model directories under text_encoders/clip."""
    entries = {}
    for root in _model_roots(("text_encoders", "clip")):
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            rel_dir = os.path.relpath(dirpath, root)
            if dirpath != root and _is_shard_dir(dirpath):
                label = rel_dir.replace(os.sep, "/") + "/"
                entries.setdefault(label, dirpath)
                dirnames[:] = []
                continue
            for fname in filenames:
                if not fname.lower().endswith(".safetensors"):
                    continue
                rel = fname if rel_dir == "." else os.path.join(rel_dir, fname)
                entries.setdefault(rel.replace(os.sep, "/"), os.path.join(dirpath, fname))
    return entries


def list_gguf_entries():
    entries = {}
    for root in _model_roots(("text_encoders", "clip", "clip_gguf")):
        for dirpath, _, filenames in os.walk(root, followlinks=True):
            rel_dir = os.path.relpath(dirpath, root)
            for fname in filenames:
                if not fname.lower().endswith(".gguf"):
                    continue
                rel = fname if rel_dir == "." else os.path.join(rel_dir, fname)
                entries.setdefault(rel.replace(os.sep, "/"), os.path.join(dirpath, fname))
    return entries


def load_shard_state_dict(dir_path):
    """Merge a sharded HuggingFace safetensors checkpoint (e.g. the 3-piece
    Z-Image-Engineer-V6 release) into a single state dict."""
    index_path = os.path.join(dir_path, SHARD_INDEX_FILE)
    shard_files = []
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as handle:
            weight_map = json.load(handle).get("weight_map", {})
        shard_files = sorted(set(weight_map.values()))
    if not shard_files:
        shard_files = sorted(name for name in os.listdir(dir_path) if SHARD_FILE_PATTERN.match(name))
    if not shard_files:
        shard_files = sorted(name for name in os.listdir(dir_path) if name.lower().endswith(".safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No safetensors shards found in '{dir_path}'")

    state_dict = {}
    for name in shard_files:
        shard_path = os.path.join(dir_path, name)
        if not os.path.isfile(shard_path):
            raise FileNotFoundError(f"Shard '{name}' referenced by the index is missing in '{dir_path}'")
        logging.info("Z-Engineer: loading shard %s", name)
        part = comfy.utils.load_torch_file(shard_path, safe_load=True)
        state_dict.update(part)
    logging.info("Z-Engineer: merged %s shard(s), %s tensors", len(shard_files), len(state_dict))
    return state_dict


def _base_model_options(device):
    # Keep the initial weight placement on the offload device so loading never
    # spikes VRAM while another job shares the GPU; comfy's model management
    # moves the model to the compute device on first use.
    model_options = {"initial_device": comfy.model_management.text_encoder_offload_device()}
    if device == "cpu":
        model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
    return model_options


def _load_clip_from_state_dict(state_dict, model_options):
    return comfy.sd.load_text_encoder_state_dicts(
        state_dicts=[state_dict],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION,
        model_options=model_options,
    )


def _resolve_comfyui_gguf():
    """Find an already-imported ComfyUI-GGUF custom node package and return
    (gguf_clip_loader, GGMLOps, GGUFModelPatcher), or None.

    Older ComfyUI registered a custom node directory in ``sys.modules`` under
    its bare folder name ("ComfyUI-GGUF"); current ComfyUI uses the full
    on-disk path with dots mangled to ``_x_`` (nodes.py load_custom_node), so
    the old exact-leaf match never fires (issue #13). Match the normalized
    last path/dot segment of the module name, plus the package's on-disk
    folder name. Only real packages (with ``__path__``) can resolve the
    relative ``.loader`` / ``.ops`` / ``.nodes`` submodules; a wrong guess is
    filtered by the import below. Some modules expose exotic ``__path__``
    objects (e.g. ``torch.classes``), so treat every candidate defensively.
    """

    def _norm(text):
        return str(text or "").lower().replace("_", "-")

    def _leaf(name):
        norm = _norm(name).replace("\\", "/")
        return norm.rsplit("/", 1)[-1].rsplit(".", 1)[-1]

    def _folder_names(mod):
        try:
            paths = [p for p in list(getattr(mod, "__path__", None) or []) if isinstance(p, str)]
        except TypeError:
            paths = []
        init = getattr(mod, "__file__", None)
        if isinstance(init, str) and os.path.basename(init) == "__init__.py":
            paths.append(os.path.dirname(init))
        return [os.path.basename(os.path.normpath(p)) for p in paths if p]

    anchors = []
    for name, mod in list(sys.modules.items()):
        try:
            if mod is None or not getattr(mod, "__path__", None):
                continue
            if "comfyui-gguf" in _leaf(name) or any("comfyui-gguf" in _norm(f) for f in _folder_names(mod)):
                anchors.append(name)
        except Exception:
            continue

    for name in sorted(set(anchors), key=len):
        try:
            loader_mod = importlib.import_module(".loader", name)
            ops_mod = importlib.import_module(".ops", name)
            nodes_mod = importlib.import_module(".nodes", name)
            return (
                loader_mod.gguf_clip_loader,
                ops_mod.GGMLOps,
                nodes_mod.GGUFModelPatcher,
            )
        except Exception as exc:
            logging.warning("Z-Engineer: found '%s' but could not use it (%s)", name, exc)
    return None


class ZEngineerCLIPLoader:
    """Load Z-Image-Engineer (Qwen3-4B) as a Z-Image text encoder from a
    single .safetensors file or a sharded HuggingFace folder."""

    @classmethod
    def INPUT_TYPES(cls):
        entries = sorted(list_safetensors_entries().keys())
        return {
            "required": {
                "model_name": (entries, {"tooltip": "Single .safetensors file or a sharded HF folder (model-0000X-of-0000N) under models/text_encoders."}),
            },
            "optional": {
                "device": (["default", "cpu"], {"advanced": True}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load_clip"
    CATEGORY = "Z-Engineer"
    DESCRIPTION = "Loads Z-Image-Engineer / Qwen3-4B safetensors (single file or 3-piece HF shards) as a Z-Image text encoder usable for CLIP Text Encode and the Z-Engineer prompt enhancer."

    def load_clip(self, model_name, device="default"):
        entries = list_safetensors_entries()
        path = entries.get(model_name)
        if path is None:
            raise FileNotFoundError(f"Model '{model_name}' not found under text_encoders/clip folders")

        if os.path.isdir(path):
            state_dict = load_shard_state_dict(path)
        else:
            state_dict = comfy.utils.load_torch_file(path, safe_load=True)

        clip = _load_clip_from_state_dict(state_dict, _base_model_options(device))
        return (clip,)


class ZEngineerCLIPLoaderGGUF:
    """Load Z-Image-Engineer GGUF quants as a Z-Image text encoder."""

    @classmethod
    def INPUT_TYPES(cls):
        entries = sorted(list_gguf_entries().keys())
        return {
            "required": {
                "gguf_name": (entries, {"tooltip": "A llama.cpp-style Qwen3 GGUF (e.g. Z-Image-Engineer-V6-Q4_K_M.gguf) under models/text_encoders."}),
            },
            "optional": {
                "device": (["default", "cpu"], {"advanced": True}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load_clip"
    CATEGORY = "Z-Engineer"
    DESCRIPTION = "Loads Z-Image-Engineer GGUF quants as a Z-Image text encoder. Uses ComfyUI-GGUF for on-the-fly dequantization when installed; otherwise falls back to dequantizing to FP16 at load time."

    def load_clip(self, gguf_name, device="default"):
        entries = list_gguf_entries()
        path = entries.get(gguf_name)
        if path is None:
            raise FileNotFoundError(f"GGUF '{gguf_name}' not found under text_encoders/clip folders")

        model_options = _base_model_options(device)
        resolved = _resolve_comfyui_gguf()
        if resolved is not None:
            gguf_clip_loader, GGMLOps, GGUFModelPatcher = resolved
            logging.info("Z-Engineer: loading GGUF via ComfyUI-GGUF (on-the-fly dequant)")
            state_dict = gguf_clip_loader(path)
            model_options["custom_operations"] = GGMLOps
            model_options.setdefault("initial_device", comfy.model_management.text_encoder_offload_device())
            clip = _load_clip_from_state_dict(state_dict, model_options)
            clip.patcher = GGUFModelPatcher.clone(clip.patcher)
        else:
            from .gguf_fallback import load_gguf_state_dict_dequant

            logging.info(
                "Z-Engineer: ComfyUI-GGUF not installed; dequantizing GGUF to FP16 at load "
                "(install ComfyUI-GGUF for lower VRAM use)"
            )
            state_dict = load_gguf_state_dict_dequant(path)
            clip = _load_clip_from_state_dict(state_dict, model_options)
        return (clip,)


class ZEngineerEnhance:
    """Run the Z-Image-Engineer model loaded as CLIP as an in-process prompt
    enhancer. No external server required."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "The Z-Image-Engineer model loaded with one of the Z-Engineer CLIP loaders (or any Z-Image Qwen3-4B CLIP)."}),
                "input_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Raw prompt seed (or newline-separated batch)...",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": V6_SYSTEM_PROMPT,
                        "placeholder": "Z-Engineer system prompt...",
                    },
                ),
                "seed": ("INT", {"default": 6606, "min": 0, "max": MAX_SEED}),
                "temperature": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 1000}),
                "min_p": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 5.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 320, "min": 32, "max": 4096}),
                "enforce_seed_terms": ("BOOLEAN", {"default": True, "tooltip": "Deterministically re-append seed phrases (counts, colors, quoted text) the model dropped."}),
                "strip_reasoning": ("BOOLEAN", {"default": True}),
                "sanitize_output": ("BOOLEAN", {"default": True}),
                "batch_mode": ("BOOLEAN", {"default": False}),
                "batch_separator": ("STRING", {"multiline": False, "default": "\\n---\\n"}),
            },
            "optional": {
                "keep_terms": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "Comma-separated trigger words/phrases (e.g. LoRA triggers) kept verbatim in the output. Any the model drops are re-appended.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    OUTPUT_NODE = True
    FUNCTION = "enhance"
    CATEGORY = "Z-Engineer"
    DESCRIPTION = "Enhances a raw seed prompt into a polished Z-Image Turbo prompt using the locally loaded Z-Image-Engineer model (the same CLIP also used for text encoding). The enhanced prompt is previewed on the node."

    def enhance(
        self,
        clip,
        input_prompt,
        system_prompt,
        seed,
        temperature,
        top_p,
        top_k,
        min_p,
        repetition_penalty,
        max_tokens,
        enforce_seed_terms,
        strip_reasoning,
        sanitize_output,
        batch_mode,
        batch_separator,
        keep_terms="",
    ):
        input_prompt = str(input_prompt or "").strip()
        if not input_prompt:
            return {"ui": {"text": [""]}, "result": ("",)}

        if clip is None:
            raise RuntimeError("Z-Engineer Enhance: no CLIP provided. Load the model with a Z-Engineer CLIP loader first.")
        cond_model = getattr(clip, "cond_stage_model", None)
        if cond_model is None or not hasattr(cond_model, "generate"):
            raise RuntimeError(
                "Z-Engineer Enhance: this CLIP does not support text generation. "
                "Load Z-Image-Engineer (Qwen3-4B) with the Z-Engineer CLIP Loader or the GGUF loader."
            )

        system_prompt = str(system_prompt or "").strip() or V6_SYSTEM_PROMPT
        kept_terms = parse_keep_terms(keep_terms)
        prompts = split_batch(input_prompt, bool(batch_mode), str(batch_separator or ""))
        outputs = []
        for idx, seed_prompt in enumerate(prompts):
            chat_text = build_chat_prompt(system_prompt, build_user_prompt(seed_prompt, kept_terms))
            tokens = clip.tokenize(chat_text, llama_template="{}")
            do_sample = float(temperature) > 0.0
            token_ids = clip.generate(
                tokens,
                do_sample=do_sample,
                max_length=int(max_tokens),
                temperature=max(float(temperature), 1e-3),
                top_k=int(top_k),
                top_p=float(top_p),
                min_p=float(min_p),
                repetition_penalty=float(repetition_penalty),
                seed=(int(seed) + idx) & MAX_SEED,
            )
            raw_text = clip.decode(token_ids)
            cleaned = sanitize_prompt(raw_text, bool(strip_reasoning), bool(sanitize_output))
            if enforce_seed_terms:
                cleaned = preserve_seed_constraints(seed_prompt, cleaned)
            if kept_terms:
                cleaned = enforce_keep_terms(cleaned, kept_terms)
            outputs.append(cleaned)

        if batch_mode:
            result = decode_separator(batch_separator).join(outputs)
        else:
            result = outputs[0]
        return {"ui": {"text": outputs}, "result": (result,)}
