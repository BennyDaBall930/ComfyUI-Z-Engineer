# V6 Workflow Notes

## Recommended: fully local path (no external server)

1. Place the model under `ComfyUI/models/text_encoders/`:
   - sharded HF release folder `Z-Image-Engineer-V6/` (3-piece safetensors + index), or
   - a GGUF quant such as `Z-Image-Engineer-V6-Q4_K_M.gguf`.
2. Load it with **Z-Engineer CLIP Loader (Safetensors / Shards)** or **Z-Engineer CLIP Loader (GGUF)**.
3. Use the resulting `CLIP` in two places at once:
   - as the Z-Image Turbo text encoder via the standard **CLIP Text Encode** node, and
   - as the generator behind **Z-Engineer Prompt Enhancer (Local)**.
4. Wire the enhancer's `prompt` output into CLIP Text Encode's `text` input. The enhanced
   prompt is previewed directly on the node after each run.

Because both jobs share one loaded model, the V6 fine-tune binds its own enhanced prompt
into the conditioning — this is the intended V6 deployment.

### Settings

`temperature 0.20 / top_p 0.9 / top_k 40 / min_p 0.03 / repetition_penalty 1.05 / max_tokens 320`,
`enforce_seed_terms`, `strip_reasoning`, and `sanitize_output` all enabled.

### A/B testing the text encoder

Compare against stock `qwen_3_4b.safetensors` with identical seed, sampler, CFG,
resolution, scheduler, and raw prompt. Arms worth recording:

- Raw prompt + base Z-Image-Turbo text encoder (baseline).
- Raw prompt + V6 text encoder.
- Enhanced prompt + base text encoder.
- Enhanced prompt + V6 text encoder (full V6 lane).

Record: raw seed prompt, enhanced prompt, model file/quant, node sampling settings, and
the full ComfyUI sampler settings for each arm.

## Legacy: OpenAI-compatible API path

The **Z-Engineer Prompt Enhancer (API)** node still talks to any
`/chat/completions` server (LM Studio, llama.cpp server, Ollama) at
`http://localhost:1234/v1` by default. Set `model` to `auto` to pick a
Z-Image-Engineer model from `/v1/models` automatically.

- `error_mode=return_input` is safest for production: rendering continues with the raw
  prompt if the server is offline. Use `return_error` while debugging.
- Some thinking models spend the first part of the token budget in `reasoning_content`.
  If the node reports that no final `content` was returned, raise `max_tokens` or switch
  to a non-thinking model build.

## Quant guidance

| Quant | File size | Notes |
| --- | --- | --- |
| Q3_K_M | 2.1 GB | smallest, noticeable quality loss |
| Q4_K_M | 2.5 GB | recommended default |
| Q5_K_M / Q6_K | 2.9 / 3.3 GB | better fidelity, still small |
| Q8_0 | 4.3 GB | near-lossless |
| F16 | 8.1 GB | full fidelity, same as safetensors release |

With ComfyUI-GGUF installed the quant stays quantized in VRAM; without it the GGUF is
dequantized to FP16 at load time (full FP16 footprint, still functional).
