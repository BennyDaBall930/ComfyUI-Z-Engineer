# ComfyUI Z-Engineer

OpenAI-compatible ComfyUI prompt-engineering node for Z-Image Turbo workflows.

V6 keeps the same integration path as the original node: it calls a local OpenAI-compatible `/chat/completions` server such as LM Studio and returns a clean prompt string for downstream Z-Image text/CLIP encode nodes.

## V6 Features

- Z-Image-Engineer V6 system prompt as the default.
- LM Studio `/v1/models` discovery with `model=auto` support.
- Prompt-only output sanitizer.
- `<think>...</think>` reasoning strip control.
- Batch mode for newline-separated or separator-separated prompt lists.
- Reproducible `seed`, `temperature`, `top_p`, and `max_tokens`.
- Retry, timeout, and error handling controls.
- Same OpenAI-compatible API path: `http://localhost:1234/v1/chat/completions`.

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/BennyDaBall930/ComfyUI-Z-Engineer.git
```

Restart ComfyUI.

## Usage

1. Start LM Studio with an OpenAI-compatible server on `http://localhost:1234/v1`.
2. Load a prompt-engineer model such as `qwen3-4b-z-image-engineer-v4`, or set `model` to `auto`.
3. In ComfyUI, add **Z-Engineer > Z-Engineer**.
4. Connect the node output `prompt` to the Z-Image text encode / CLIP text encode path.

Recommended defaults:

- `model`: `qwen3-4b-z-image-engineer-v4` or `auto`
- `temperature`: `0.55`
- `top_p`: `0.9`
- `max_tokens`: `1200` for thinking models, lower for prompt-only adapters
- `strip_reasoning`: `true`
- `sanitize_output`: `true`
- `error_mode`: `return_input`

## Batch Mode

Enable `batch_mode` to process several prompts in one node call. The node splits by `batch_separator` when present, otherwise by lines, and returns prompts joined by the same separator.

Default separator:

```text
\n---\n
```

## V6 System Prompt

The default system prompt is tuned for the Tongyi-MAI Z-Image-Turbo Qwen text encoder and asks for one natural-language prompt-only paragraph with preserved constraints, visual hierarchy, semantic cinematography, lighting, lens/depth, texture, composition, and style control.

## Requirements

- ComfyUI
- `requests`
- A running local OpenAI-compatible chat server, usually LM Studio
