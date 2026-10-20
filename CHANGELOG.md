# Changelog

## v2.3.2 (2026-10-20)

- 2.3.2: packaging only. Dev test harness excluded from the registry package; no behavior change.

## v2.3.1 (2026-10-19)

- Registry-safe loading, no behavior change: nothing runs at import beyond registering the nodes. The FP4 dequant kernels are registered once, by the Z-Engineer GGUF loader when it runs; ComfyUI-GGUF is located through ComfyUI's list of loaded node packs; no prompt hooks; specific exception handling with warnings instead of silent catch-alls.

## v2.3.0 (2026-10-19)

- NVFP4 GGUF support: loads `Z-Image-Engineer-V6-NVFP4.gguf` (RTX 50 / Blackwell) through the Z-Engineer GGUF loader and the stock `CLIPLoaderGGUF`.
- Fast GPU dequant for NVFP4 and MXFP4, registered into ComfyUI-GGUF (which otherwise falls back to numpy for these types). Bit-exact with the gguf / llama.cpp reference decoders.
- Prompt enhancer: GGUF weights are dequantized once per run instead of on every token. Same output, about 33 tok/s for every quant on an RTX 5090 (was ~5 tok/s NVFP4, ~14 tok/s Q8_0). Skipped automatically when VRAM is tight.
- Requires `gguf>=0.19.0`. Older versions now fail with a clear "update gguf" message on NVFP4 files; other quants keep working.
- NVFP4 GGUF load in ComfyUI: about 12 s (same as Q8_0), down from about 33 s on the numpy fallback.
- Two-level NVFP4 GGUFs (`.scale` tensors) are refused with a clear message instead of loading with wrong weights.
- LFM2.5 enhancer: load fully onto the GPU, which fixes a "tensors on different devices" crash when VRAM is tight. The node now shows the enhanced-prompt preview and resizable text boxes.
- API node: a failed request is re-run on the next queue instead of being cached. Only connection errors, timeouts and 5xx responses are retried. `api_url` must be a plain http(s) URL.
- An empty generation keeps the seed prompt and logs a warning, instead of being padded into a boilerplate prompt.
- `repetition_penalty` minimum is 0.01 (0 broke sampling). The batch separator handles a trailing backslash and non-ASCII text.
- FP4 dequant registration no longer depends on custom-node load order. The old-gguf warning is logged once. Lower peak RAM in the FP16 fallback loader.

## v2.2.1 (2026-08-19)

- Bundled LFM2.5 example workflow, verified end-to-end.

## v2.2.0 (2026-08-19)

- LFM2.5-Z-Image-Engineer support (standalone prompt enhancer).

## v2.1.1 (2026-07-17)

- Fix ComfyUI-GGUF detection under new ComfyUI module naming (#13).

## v2.1.0 (2026-06-12)

- `keep_terms` trigger-word field (#8) and per-box resizable text areas (#9).

## v2.0.0 (2026-06-12)

- Local model loading (sharded safetensors + GGUF) and in-ComfyUI prompt enhancement.
