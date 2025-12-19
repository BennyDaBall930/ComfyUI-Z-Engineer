# ComfyUI Z-Engineer

A custom node for ComfyUI that integrates a local LLM (via OpenAI-compatible API) to engineer optimal prompts for Z-Image Turbo workflows.

## Features

-   **Local LLM Integration**: Connects to your local LLM server (e.g., LM Studio, Oobabooga, vLLM) via standard OpenAI API format.
-   **System Prompting**: Dedicated system prompt input to guide the LLM's behavior.
-   **Z-Image Optimization**: Designed to work with models like [Qwen3-4b-Z-Image-Engineer](https://huggingface.co/BennyDaBall/qwen3-4b-Z-Image-Engineer).

## Installation

1.  Clone this repository into your ComfyUI `custom_nodes` directory:
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/BennyDaBall/ComfyUI-Z-Engineer.git
    ```
2.  Restart ComfyUI.

## Usage

1.  Find the node under **Z-Engineer > Z-Engineer**.
2.  Connect the output `prompt` to your CLIP Text Encode node.
3.  **Configuration**:
    -   **Input Prompt**: Your raw idea or base prompt.
    -   **System Prompt**: Instructions for the LLM (e.g., "Convert this to a Z-Image Turbo prompt...").
    -   **API URL**: Your local LLM endpoint (default: `http://localhost:1234/v1`).
    -   **Model**: The model name string (default: `local-model`).
    -   **Seed/Temperature**: Standard LLM generation parameters.

## Recommended Model

For best results, use the **Z-Image Engineer** model:
[**BennyDaBall/qwen3-4b-Z-Image-Engineer**](https://huggingface.co/BennyDaBall/qwen3-4b-Z-Image-Engineer)

## Requirements

-   A running local LLM server compatible with OpenAI's `chat/completions` endpoint.
-   ComfyUI installed.
