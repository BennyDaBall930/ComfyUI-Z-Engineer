"""Torch dequantizers for the llama.cpp FP4 GGUF types (NVFP4, MXFP4).

ComfyUI-GGUF has no torch kernel for these types, so it falls back to
gguf-py's numpy dequant on every layer of every forward pass. That is fine
for a single text-encoder pass but makes the in-ComfyUI prompt enhancer
(hundreds of forwards per rewrite) crawl. These ports run on the tensor's
own device and are bit-exact with gguf-py 0.19.0 (``gguf.quants.NVFP4`` /
``gguf.quants.MXFP4``), which mirror ggml's reference decoders.

Signature matches ComfyUI-GGUF's ``dequantize_functions`` entries:
``fn(blocks, block_size, type_size, dtype=None)`` with ``blocks`` a
``(n_blocks, type_size)`` uint8 tensor.
"""

import logging

import torch

# E2M1 magnitudes doubled (ggml kvalues_mxfp4); the scale decoders below
# carry the matching 0.5 factor.
FP4_KVALUES = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)

_LUT_CACHE = {}
_WARNED_OLD_GGUF = False


def _pair_lut(device):
    """(256, 2) bf16 table: byte -> (low-nibble value, high-nibble value)."""
    lut = _LUT_CACHE.get(device)
    if lut is None:
        kv = torch.tensor(FP4_KVALUES, dtype=torch.float32)
        b = torch.arange(256)
        lut = torch.stack([kv[b & 0x0F], kv[b >> 4]], dim=-1).to(device=device, dtype=torch.bfloat16)
        _LUT_CACHE[device] = lut
    return lut


def _fp4_values(qs):
    """Decode nibble bytes (..., k) to (..., 2k) values: low nibbles fill the
    first half of each group, high nibbles the second."""
    pairs = _pair_lut(qs.device)[qs.to(torch.int32)]
    return pairs.transpose(-1, -2).reshape(*qs.shape[:-1], 2 * qs.shape[-1])


def _ue4m3_to_fp32_half(x):
    """Unsigned E4M3 (bias 7) with the 0.5 kvalues factor; 0 and 0x7F (NaN) decode to 0."""
    x = x.to(torch.int32)
    exp = (x >> 3) & 0xF
    man = (x & 0x7).to(torch.float32)
    raw = torch.where(
        exp == 0,
        man * 2.0**-9,
        (1.0 + man / 8.0) * torch.exp2(exp.to(torch.float32) - 7.0),
    )
    return torch.where((x == 0) | (x == 0x7F), torch.zeros_like(raw), raw * 0.5)


def _e8m0_to_fp32_half(x):
    """ggml_e8m0_to_fp32_half: 2^(e-127) * 0.5, built from the float32 bit pattern."""
    x = x.to(torch.int32)
    bits = torch.where(x < 2, torch.full_like(x, 0x00200000) << x, (x - 1) << 23)
    return bits.view(torch.float32)


# Both formats multiply an FP4 value (at most 2 significant bits) by a scale
# with at most 4 significant bits, so every product is exact in bf16; the
# math runs in bf16 and a single cast yields the same bits as fp32 math.


def dequantize_blocks_NVFP4(blocks, block_size, type_size, dtype=None):
    # block: 4 UE4M3 sub-block scales + 32 bytes of nibbles = 64 values (4 x 16)
    n_blocks = blocks.shape[0]
    d = _ue4m3_to_fp32_half(blocks[:, :4]).to(torch.bfloat16).unsqueeze(-1)
    vals = _fp4_values(blocks[:, 4:].reshape(n_blocks, 4, 8))
    out = (d * vals).reshape(n_blocks, block_size)
    return out if dtype is None else out.to(dtype)


def dequantize_blocks_MXFP4(blocks, block_size, type_size, dtype=None):
    # block: 1 E8M0 scale + 16 bytes of nibbles = 32 values
    d = _e8m0_to_fp32_half(blocks[:, :1]).to(torch.bfloat16)
    out = d * _fp4_values(blocks[:, 1:])
    return out if dtype is None else out.to(dtype)


def fp4_dequantize_functions(gguf):
    """{GGMLQuantizationType: fn} for the FP4 types this gguf package knows."""
    funcs = {}
    qtypes = gguf.GGMLQuantizationType
    if hasattr(qtypes, "NVFP4"):
        funcs[qtypes.NVFP4] = dequantize_blocks_NVFP4
    if hasattr(qtypes, "MXFP4"):
        funcs[qtypes.MXFP4] = dequantize_blocks_MXFP4
    return funcs


_KERNELS_REGISTERED = False


def ensure_fp4_kernels(dequant_module):
    """Register the FP4 kernels with ComfyUI-GGUF once per session. Called by
    the Z-Engineer GGUF loader; nothing is registered at import time."""
    global _KERNELS_REGISTERED
    if not _KERNELS_REGISTERED:
        register_comfyui_gguf_fp4(dequant_module)
        _KERNELS_REGISTERED = True


def register_comfyui_gguf_fp4(dequant_module):
    """Add the FP4 kernels to ComfyUI-GGUF's dequant table (never overriding
    a kernel ComfyUI-GGUF ships itself). Returns the type names added."""
    try:
        import gguf
    except ImportError:
        return []
    table = getattr(dequant_module, "dequantize_functions", None)
    if not isinstance(table, dict):
        return []
    added = []
    for qtype, fn in fp4_dequantize_functions(gguf).items():
        if qtype not in table:
            table[qtype] = fn
            added.append(qtype.name)
    if added:
        logging.info("Z-Engineer: registered torch dequant for %s in ComfyUI-GGUF", ", ".join(added))
    global _WARNED_OLD_GGUF
    if not hasattr(gguf.GGMLQuantizationType, "NVFP4") and not _WARNED_OLD_GGUF:
        _WARNED_OLD_GGUF = True
        logging.warning(
            "Z-Engineer: installed gguf %s cannot read NVFP4 GGUFs; update with: pip install -U \"gguf>=0.19.0\"",
            gguf_version(),
        )
    return added


def gguf_version():
    try:
        from importlib.metadata import version

        return version("gguf")
    except ImportError:  # includes importlib.metadata.PackageNotFoundError
        return "(unknown version)"
