"""Fallback GGUF loader used when ComfyUI-GGUF is not installed.

Dequantizes a llama.cpp-style Qwen3 GGUF to FP16 tensors at load time and
remaps tensor names to the HuggingFace layout that ComfyUI's text-encoder
detection expects. This trades VRAM (full FP16 footprint) for zero extra
custom-node dependencies; ComfyUI-GGUF remains the recommended path.
"""

import logging

import numpy as np
import torch

# llama.cpp -> HuggingFace key fragments (qwen3 architecture)
LLAMA_TO_HF_MAP = {
    "blk.": "model.layers.",
    "attn_norm": "input_layernorm",
    "attn_q_norm.": "self_attn.q_norm.",
    "attn_k_norm.": "self_attn.k_norm.",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_norm": "post_attention_layernorm",
    "token_embd": "model.embed_tokens",
    "output_norm": "model.norm",
    "output.weight": "lm_head.weight",
}

SUPPORTED_ARCHS = {"qwen3"}


def _remap_key(name):
    for old, new in LLAMA_TO_HF_MAP.items():
        name = name.replace(old, new)
    return name


def _read_arch(reader, gguf):
    field = reader.get_field("general.architecture")
    if field is None:
        return None
    if len(field.types) != 1 or field.types[0] != gguf.GGUFValueType.STRING:
        return None
    return str(field.parts[field.data[-1]], encoding="utf-8")


def load_gguf_state_dict_dequant(path, dtype=torch.float16):
    try:
        import gguf
    except ImportError as exc:
        raise RuntimeError(
            "Loading GGUF without ComfyUI-GGUF requires the 'gguf' package. "
            "Install the ComfyUI-GGUF custom node (recommended, lower VRAM) or run: pip install gguf"
        ) from exc

    reader = gguf.GGUFReader(path)
    arch = _read_arch(reader, gguf)
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(
            f"Unsupported GGUF architecture {arch!r} for the Z-Engineer fallback loader "
            f"(expected one of {sorted(SUPPORTED_ARCHS)}). Install ComfyUI-GGUF for broader support."
        )

    state_dict = {}
    qtype_counts = {}
    for tensor in reader.tensors:
        shape = tuple(reversed(tuple(int(dim) for dim in tensor.shape)))
        tensor_type = tensor.tensor_type
        if tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            array = np.asarray(tensor.data)
        else:
            try:
                array = gguf.quants.dequantize(tensor.data, tensor_type)
            except Exception as exc:
                raise ValueError(
                    f"Cannot dequantize tensor '{tensor.name}' with type {tensor_type!r}: {exc}. "
                    "Install ComfyUI-GGUF or use a different quant."
                ) from exc
        torch_tensor = torch.from_numpy(np.array(array, copy=True)).reshape(shape)
        if torch_tensor.ndim <= 1:
            torch_tensor = torch_tensor.to(torch.float32)
        else:
            torch_tensor = torch_tensor.to(dtype)
        state_dict[_remap_key(tensor.name)] = torch_tensor

        type_name = getattr(tensor_type, "name", repr(tensor_type))
        qtype_counts[type_name] = qtype_counts.get(type_name, 0) + 1

    logging.info(
        "Z-Engineer GGUF fallback: dequantized %s tensors (%s)",
        len(state_dict),
        ", ".join(f"{k} ({v})" for k, v in qtype_counts.items()),
    )
    return state_dict
