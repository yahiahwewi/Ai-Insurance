import base64
import requests
import os
import sys
from groq import Groq

from dotenv import load_dotenv
load_dotenv()

 # Initialize Groq client
groq_api_key = os.getenv("API_KEY")

def describe_image(image_bytes: bytes) -> str:
    if len(image_bytes) > 4 * 1024 * 1024:
        raise ValueError("Image size exceeds 4MB limit.")

    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    image_data_url = f"data:image/png;base64,{image_base64}"

    client = Groq(api_key=groq_api_key)

    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the contents of this image. Analyze the image and identify the car’s make, model, approximate year, and visible condition"},
                    {"type": "image_url", "image_url": {"url": image_data_url}}
                ]
            }
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
    )

    return completion.choices[0].message.content