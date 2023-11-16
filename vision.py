import os
import base64
import requests

# https://platform.openai.com/docs/guides/vision

MODEL: str = "gpt-4-vision-preview"
MAX_TOKENS: int = 32

def encode_image_from_path(image_path: str) -> str:
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def vision(
    prompt: str,
    base64_image: str,
    vision_model: str = MODEL,
    max_tokens: int = MAX_TOKENS,
) -> str:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
    payload = {
        "model": vision_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": max_tokens,
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    return response.json()["choices"][0]["message"]["content"]
