# V6 Workflow Notes

## Direct LM Studio Path

1. In LM Studio, start the local server at `http://localhost:1234/v1`.
2. Load `qwen3-4b-z-image-engineer-v4` for continuity, or a stronger local teacher model for prompt rewriting.
3. In the Z-Engineer node, set `model` to the exact LM Studio model id or `auto`.
4. Keep `strip_reasoning` and `sanitize_output` enabled.
5. Feed the returned `prompt` string into the Z-Image text encode path.

## Reproducible Comparison

Use the same Z-Image seed, sampler, resolution, checkpoint, and workflow settings for each arm:

- Raw prompt with no enhancement.
- Base Z-Image-Turbo text encoder.
- V6 SMART-LoRA adapter used as the text encoder.
- Upgraded Z-Engineer node output routed into the normal Z-Image workflow.

Record:

- Raw seed prompt.
- Z-Engineer output prompt.
- LM Studio model id.
- Node seed, temperature, top_p, and max_tokens.
- Z-Image checkpoint and adapter path.
- ComfyUI seed, sampler, steps, CFG, resolution, scheduler, and denoise.
- Image outputs at 500, 1000, and 1500 training steps when those checkpoints exist.

## Error Handling

`error_mode=return_input` is safest for production workflows because ComfyUI can continue rendering the original prompt if LM Studio is offline. Use `return_error` while debugging node/server issues.

Some LM Studio thinking models spend the first part of the token budget in `reasoning_content`. If the node reports that no final `content` was returned, raise `max_tokens` or switch to a non-thinking prompt adapter.
