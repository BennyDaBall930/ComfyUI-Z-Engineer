"""LFM2.5 Z-Image-Engineer support (prompt enhancer ONLY).

Loads the LFM2.5-1.2B-Z-Image-Engineer models (GGUF quants or the HF
safetensors folder) as an in-ComfyUI prompt enhancer running on plain
torch + transformers, so it works on any backend torch supports (CUDA,
ROCm, MPS, CPU) with no llama.cpp build required.

IMPORTANT: LFM2.5 is NOT a Z-Image text encoder. Z-Image Turbo's text
encoder is Qwen3-4B; ComfyUI has no `lfm2` text-encoder implementation,
so this model can never produce Z-Image conditioning. It writes prompts;
a Qwen3-4B model (e.g. Z-Image-Engineer, loaded with the Z-Engineer CLIP
loaders) still does the encoding.
"""

import contextlib
import json
import logging
import os
import re

import numpy as np
import torch

import comfy.model_management
import comfy.model_patcher

from .gguf_fallback import read_gguf_architecture
from .local_nodes import _model_roots, list_gguf_entries
from .prompt_utils import (
    LFM_V4_SYSTEM_PROMPT,
    build_chat_prompt,
    build_user_prompt,
    decode_separator,
    enforce_keep_terms,
    parse_keep_terms,
    preserve_seed_constraints,
    sanitize_prompt,
    split_batch,
)

MAX_SEED = 0xFFFFFFFFFFFFFFFF

NOT_A_TEXT_ENCODER = (
    "LFM2.5-Z-Image-Engineer is a prompt writer, not a text encoder: ComfyUI has no "
    "'lfm2' text-encoder implementation, so it cannot produce Z-Image conditioning. "
    "Use it with 'Z-Engineer Prompt Enhancer (LFM2.5 Local)' and keep a Qwen3-4B model "
    "(e.g. Z-Image-Engineer, via the Z-Engineer CLIP loaders) as the CLIP."
)

# llama.cpp -> HuggingFace tensor names (lfm2 architecture). Verified
# numerically against the HF export: F32 tensors match bit-exactly, so no
# permutation is involved.
LFM2_BLOCK_MAP = {
    "attn_norm": "operator_norm",
    "ffn_norm": "ffn_norm",
    "attn_q_norm": "self_attn.q_layernorm",
    "attn_k_norm": "self_attn.k_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.out_proj",
    "shortconv.conv": "conv.conv",
    "shortconv.in_proj": "conv.in_proj",
    "shortconv.out_proj": "conv.out_proj",
    "ffn_gate": "feed_forward.w1",
    "ffn_down": "feed_forward.w2",
    "ffn_up": "feed_forward.w3",
}
LFM2_TOP_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "token_embd_norm.weight": "model.embedding_norm.weight",
    "output_norm.weight": "model.embedding_norm.weight",
    "output.weight": "lm_head.weight",
}
_BLK_RE = re.compile(r"^blk\.(\d+)\.(.+?)\.(weight|bias)$")

# Byte-level BPE pre-tokenizer regex used by the LFM2/LFM2.5 tokenizer
# (tokenizer.ggml.pre == "lfm2"), copied verbatim from the HF tokenizer.json.
LFM2_PRETOK_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def _require_transformers():
    try:
        from transformers import Lfm2Config, Lfm2ForCausalLM  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "The LFM2.5 Z-Engineer nodes need a transformers version with LFM2 support. "
            "Upgrade with: pip install -U \"transformers>=4.54\" (use your ComfyUI venv's pip)."
        ) from exc


def _remap_lfm2_key(name):
    if name in LFM2_TOP_MAP:
        return LFM2_TOP_MAP[name]
    match = _BLK_RE.match(name)
    if match:
        idx, frag, kind = match.groups()
        hf_frag = LFM2_BLOCK_MAP.get(frag)
        if hf_frag is not None:
            return f"model.layers.{idx}.{hf_frag}.{kind}"
    return None


def _field(reader, key, default=None):
    field = reader.fields.get(key)
    if field is None:
        return default
    try:
        return field.contents()
    except Exception:
        return default


def load_lfm2_gguf_state_dict(reader, gguf, dtype):
    """Dequantize an lfm2 GGUF into an HF-layout state dict. 1-D tensors and
    the depthwise conv weights stay FP32 (they are stored as F32 anyway)."""
    state_dict = {}
    qtype_counts = {}
    for tensor in reader.tensors:
        hf_name = _remap_lfm2_key(tensor.name)
        if hf_name is None:
            logging.warning("Z-Engineer LFM: skipping unmapped tensor %s", tensor.name)
            continue
        shape = tuple(reversed(tuple(int(dim) for dim in tensor.shape)))
        if tensor.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            array = np.asarray(tensor.data)
        else:
            try:
                array = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
            except Exception as exc:
                raise ValueError(
                    f"Cannot dequantize tensor '{tensor.name}' with type {tensor.tensor_type!r}: {exc}. "
                    "Use a different quant (Q3_K..Q8_0/F16 are supported by the gguf package)."
                ) from exc
        torch_tensor = torch.from_numpy(np.array(array, copy=True)).reshape(shape)
        if hf_name.endswith("conv.conv.weight") and torch_tensor.ndim == 2:
            # llama.cpp stores the depthwise Conv1d weight squeezed to (dim, L)
            torch_tensor = torch_tensor.unsqueeze(1)
        if torch_tensor.ndim <= 1:
            torch_tensor = torch_tensor.to(torch.float32)
        else:
            torch_tensor = torch_tensor.to(dtype)
        state_dict[hf_name] = torch_tensor

        type_name = getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))
        qtype_counts[type_name] = qtype_counts.get(type_name, 0) + 1
    logging.info(
        "Z-Engineer LFM: dequantized %s tensors (%s)",
        len(state_dict),
        ", ".join(f"{k} ({v})" for k, v in sorted(qtype_counts.items())),
    )
    return state_dict


def lfm2_config_from_gguf(reader, has_lm_head):
    from transformers import Lfm2Config

    kv_heads = [int(n) for n in _field(reader, "lfm2.attention.head_count_kv")]
    return Lfm2Config.from_dict(dict(
        vocab_size=int(_field(reader, "lfm2.vocab_size")),
        hidden_size=int(_field(reader, "lfm2.embedding_length")),
        # the GGUF stores the ACTUAL ffn width; disable HF's auto-adjust
        intermediate_size=int(_field(reader, "lfm2.feed_forward_length")),
        block_auto_adjust_ff_dim=False,
        num_hidden_layers=int(_field(reader, "lfm2.block_count")),
        num_attention_heads=int(_field(reader, "lfm2.attention.head_count")),
        num_key_value_heads=max(kv_heads),
        layer_types=["full_attention" if n > 0 else "conv" for n in kv_heads],
        norm_eps=float(_field(reader, "lfm2.attention.layer_norm_rms_epsilon", 1e-5)),
        rope_theta=float(_field(reader, "lfm2.rope.freq_base", 1000000.0)),
        max_position_embeddings=int(_field(reader, "lfm2.context_length", 128000)),
        conv_L_cache=int(_field(reader, "lfm2.shortconv.l_cache", 3)),
        conv_bias=False,
        tie_word_embeddings=not has_lm_head,
        use_cache=True,
        bos_token_id=int(_field(reader, "tokenizer.ggml.bos_token_id", 1)),
        eos_token_id=int(_field(reader, "tokenizer.ggml.eos_token_id", 7)),
        pad_token_id=int(_field(reader, "tokenizer.ggml.padding_token_id", 0)),
    ))


def lfm2_tokenizer_from_gguf(reader):
    """Rebuild the byte-level BPE tokenizer from GGUF metadata. Verified
    token-for-token identical to the HF tokenizer.json for the LFM2.5
    Engineer release (special tokens, unicode, emoji, whitespace runs)."""
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tokens = _field(reader, "tokenizer.ggml.tokens")
    token_types = _field(reader, "tokenizer.ggml.token_type")
    merges = _field(reader, "tokenizer.ggml.merges")
    if not tokens or not merges:
        raise ValueError("GGUF file is missing tokenizer vocab/merges metadata")

    CONTROL, USER_DEFINED = 3, 4
    vocab = {}
    specials = []
    user_defined = []
    for idx, (token, token_type) in enumerate(zip(tokens, token_types)):
        if token in vocab:
            continue  # padding entries can repeat; only the first id is reachable
        vocab[token] = idx
        if int(token_type) == CONTROL:
            specials.append(token)
        elif int(token_type) == USER_DEFINED:
            user_defined.append(token)

    backend = Tokenizer(models.BPE(
        vocab=vocab,
        merges=[tuple(m.split(" ", 1)) for m in merges],
        fuse_unk=False,
        byte_fallback=False,
        ignore_merges=False,
    ))
    backend.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=Regex(LFM2_PRETOK_REGEX), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    backend.decoder = decoders.ByteLevel(add_prefix_space=True, trim_offsets=True, use_regex=True)
    backend.add_special_tokens([AddedToken(s, special=True, normalized=False) for s in specials])
    if user_defined:
        backend.add_tokens([AddedToken(s, special=False, normalized=False) for s in user_defined])

    bos_id = int(_field(reader, "tokenizer.ggml.bos_token_id", 1))
    eos_id = int(_field(reader, "tokenizer.ggml.eos_token_id", 7))
    pad_id = int(_field(reader, "tokenizer.ggml.padding_token_id", 0))
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=tokens[bos_id],
        eos_token=tokens[eos_id],
        pad_token=tokens[pad_id],
    )


def list_lfm_safetensors_dirs():
    """Map display label -> path for HF-style LFM2 folders (config.json with
    model_type 'lfm2' + safetensors weights) under text_encoders/clip."""
    entries = {}
    for root in _model_roots(("text_encoders", "clip")):
        for dirpath, _, filenames in os.walk(root, followlinks=True):
            names = set(filenames)
            if "config.json" not in names or not any(n.endswith(".safetensors") for n in names):
                continue
            try:
                with open(os.path.join(dirpath, "config.json"), "r", encoding="utf-8") as handle:
                    if json.load(handle).get("model_type") != "lfm2":
                        continue
            except Exception:
                continue
            label = os.path.relpath(dirpath, root).replace(os.sep, "/") + "/"
            entries.setdefault(label, dirpath)
    return entries


def list_lfm_entries():
    entries = dict(list_gguf_entries())
    entries.update(list_lfm_safetensors_dirs())
    return entries


class _LFMContainer(torch.nn.Module):
    """Thin wrapper so ComfyUI's ModelPatcher can manage the HF model
    (PreTrainedModel exposes `device` as a read-only property, which the
    patcher would otherwise try to assign)."""

    def __init__(self, lm):
        super().__init__()
        self.lm = lm


class ZEngineerLFM:
    """The object passed over the ZE_LLM wire: HF LFM2.5 model under ComfyUI
    model management + its tokenizer."""

    def __init__(self, lm, tokenizer, name, device="default"):
        if device == "cpu":
            load_device = offload_device = torch.device("cpu")
        else:
            load_device = comfy.model_management.text_encoder_device()
            offload_device = comfy.model_management.text_encoder_offload_device()
        self.dtype = comfy.model_management.text_encoder_dtype(load_device)
        if load_device.type == "cpu":
            self.dtype = torch.float32
        # Uniform-cast the model, but keep buffers (rope inv_freq) at their
        # original precision - fp16 would flush the small inv_freq values to
        # subnormals.
        kept_buffers = [
            (module, key, buf.clone())
            for _, module in lm.named_modules()
            for key, buf in module._buffers.items()
            if buf is not None and buf.is_floating_point()
        ]
        lm = lm.to(dtype=self.dtype).eval()
        for module, key, buf in kept_buffers:
            module._buffers[key] = buf
        self.container = _LFMContainer(lm).to(offload_device)
        self.patcher = comfy.model_patcher.ModelPatcher(self.container, load_device, offload_device)
        self.tokenizer = tokenizer
        self.name = name

    def generate_text(self, chat_text, *, do_sample, max_tokens, temperature, top_k, top_p,
                      min_p, repetition_penalty, seed):
        comfy.model_management.load_models_gpu([self.patcher])
        device = self.patcher.load_device
        lm = self.container.lm

        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id
        ids = self.tokenizer.encode(chat_text, add_special_tokens=False)
        if bos_id is not None and (not ids or ids[0] != bos_id):
            ids = [bos_id] + ids
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        kwargs = dict(
            do_sample=bool(do_sample),
            max_new_tokens=int(max_tokens),
            repetition_penalty=float(repetition_penalty),
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            use_cache=True,
        )
        if do_sample:
            kwargs.update(
                temperature=max(float(temperature), 1e-3),
                top_k=int(top_k),
                top_p=float(top_p),
                min_p=float(min_p),
            )
        context = getattr(comfy.model_management, "cuda_device_context", None)
        ctx = context(device) if context is not None else contextlib.nullcontext()
        with ctx, torch.no_grad():
            output = lm.generate(input_ids, attention_mask=torch.ones_like(input_ids), **kwargs)
        return self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)


def load_lfm_model(path, device="default"):
    _require_transformers()
    if os.path.isdir(path):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logging.info("Z-Engineer LFM: loading HF folder %s", path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        try:
            lm = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
        except TypeError:  # transformers < 5 uses torch_dtype
            lm = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
        lm.config.use_cache = True
        return ZEngineerLFM(lm, tokenizer, os.path.basename(os.path.normpath(path)), device)

    import gguf
    from transformers import Lfm2ForCausalLM

    arch = read_gguf_architecture(path)
    if arch != "lfm2":
        raise ValueError(
            f"'{os.path.basename(path)}' has GGUF architecture {arch!r}, not 'lfm2'. "
            "Qwen3-based Z-Image-Engineer GGUFs load with 'Z-Engineer CLIP Loader (GGUF)' instead."
        )
    logging.info("Z-Engineer LFM: loading GGUF %s", path)
    reader = gguf.GGUFReader(path)
    state_dict = load_lfm2_gguf_state_dict(reader, gguf, torch.float16)
    config = lfm2_config_from_gguf(reader, has_lm_head="lm_head.weight" in state_dict)
    tokenizer = lfm2_tokenizer_from_gguf(reader)
    lm = Lfm2ForCausalLM(config)
    try:
        load_info = lm.load_state_dict(state_dict, strict=False, assign=True)
    except TypeError:  # torch without assign=
        load_info = lm.load_state_dict(state_dict, strict=False)
    unexpected = list(load_info.unexpected_keys)
    missing = [k for k in load_info.missing_keys if k != "lm_head.weight"]
    if missing or unexpected:
        raise ValueError(f"GGUF did not match the LFM2 layout (missing {missing[:4]}, unexpected {unexpected[:4]})")
    lm.tie_weights()
    del state_dict, reader
    return ZEngineerLFM(lm, tokenizer, os.path.basename(path), device)


class ZEngineerLFMLoader:
    """Load an LFM2.5-Z-Image-Engineer model (GGUF quant or HF safetensors
    folder) as a standalone prompt-enhancer LLM. NOT a CLIP/text encoder."""

    @classmethod
    def INPUT_TYPES(cls):
        entries = sorted(list_lfm_entries().keys())
        return {
            "required": {
                "model_name": (entries, {"tooltip": "An LFM2.5-Z-Image-Engineer GGUF (any quant) or HF safetensors folder under models/text_encoders. This loads a prompt WRITER - Z-Image still needs a Qwen3-4B CLIP."}),
            },
            "optional": {
                "device": (["default", "cpu"], {"advanced": True}),
            },
        }

    RETURN_TYPES = ("ZE_LLM",)
    RETURN_NAMES = ("llm",)
    FUNCTION = "load_llm"
    CATEGORY = "Z-Engineer"
    DESCRIPTION = "Loads LFM2.5-Z-Image-Engineer (1.2B) as a local prompt enhancer via torch/transformers (CUDA, ROCm, MPS or CPU - no llama.cpp needed). Not a text encoder: wire its llm output into the LFM2.5 enhancer node and keep using a Qwen3-4B CLIP for encoding."

    def load_llm(self, model_name, device="default"):
        entries = list_lfm_entries()
        path = entries.get(model_name)
        if path is None:
            raise FileNotFoundError(f"Model '{model_name}' not found under text_encoders/clip folders")
        return (load_lfm_model(path, device),)


class ZEngineerEnhanceLFM:
    """Prompt enhancer running the LFM2.5 Engineer loaded with the LFM loader.
    Same controls and post-processing as the Qwen3 enhancer node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm": ("ZE_LLM", {"tooltip": "The LFM2.5 Engineer loaded with 'Z-Engineer LFM2.5 Enhancer Loader'."}),
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
                        "default": LFM_V4_SYSTEM_PROMPT,
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
    DESCRIPTION = "Enhances a raw seed prompt into a polished Z-Image Turbo prompt using the locally loaded LFM2.5-Z-Image-Engineer (fast 1.2B prompt writer). Wire the STRING output into CLIP Text Encode - the CLIP itself must still be a Qwen3-4B model."

    def enhance(
        self,
        llm,
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
        if llm is None or not hasattr(llm, "generate_text"):
            raise RuntimeError(
                "Z-Engineer LFM Enhance: no LFM model provided. Load one with 'Z-Engineer LFM2.5 Enhancer Loader' first."
            )

        system_prompt = str(system_prompt or "").strip() or LFM_V4_SYSTEM_PROMPT
        kept_terms = parse_keep_terms(keep_terms)
        prompts = split_batch(input_prompt, bool(batch_mode), str(batch_separator or ""))
        outputs = []
        for idx, seed_prompt in enumerate(prompts):
            chat_text = build_chat_prompt(system_prompt, build_user_prompt(seed_prompt, kept_terms))
            raw_text = llm.generate_text(
                chat_text,
                do_sample=float(temperature) > 0.0,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
                min_p=float(min_p),
                repetition_penalty=float(repetition_penalty),
                seed=(int(seed) + idx) & MAX_SEED,
            )
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
