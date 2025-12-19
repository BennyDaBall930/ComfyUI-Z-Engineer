import requests
import json
import logging

class ZEngineer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter your prompt here..."
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant.",
                    "placeholder": "Enter system prompt here..."
                }),
                "api_url": ("STRING", {
                    "multiline": False,
                    "default": "http://localhost:1234/v1",
                }),
                "model": ("STRING", {
                    "multiline": False,
                    "default": "local-model",
                }),
                "seed": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 0xffffffffffffffff
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate_prompt"
    CATEGORY = "Z-Engineer"

    def generate_prompt(self, input_prompt, system_prompt, api_url, model, seed, temperature):
        if not input_prompt:
            return ("",)

        # Ensure API URL ends with /v1 or correct path if user didn't specify
        # We'll assume the user provides the base (e.g. http://localhost:1234/v1).
        
        endpoint = f"{api_url}/chat/completions"
        
        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_prompt}
            ],
            "temperature": temperature,
            "seed": seed,
            "stream": False
        }

        try:
            print(f"Sending request to {endpoint} with model {model}...")
            response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            # OpenAI format usually: choices[0].message.content
            output_text = data['choices'][0]['message']['content']
            
            print(f"Z-Engineer Response: {output_text[:100]}...")
            
            return (output_text,)
            
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            error_msg = f"Error: {str(e)}"
            return (error_msg,)
