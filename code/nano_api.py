import os
from pathlib import Path

from google import genai
from PIL import Image

# -----------------------------------------------------------------------------
# Nano Banana (Gemini 3.1 Flash Image) - Local API Client
# -----------------------------------------------------------------------------
# This script sends an image and an editing instruction to Google's hosted
# Gemini image model. The model itself does not run on the local machine.
#
# Prerequisites:
# 1. pip install google-genai pillow
# 2. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable
# -----------------------------------------------------------------------------

def edit_image(input_path, prompt, output_path="output.png"):
    """Edit a local image through Google's hosted Nano Banana API."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY before running this script.")

    input_file = Path(input_path)
    if not input_file.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    print(f"[*] Loading image: {input_path}")
    image = Image.open(input_file)
    print("[*] Sending image to Nano Banana (Gemini 3.1 Flash Image)...")
    print(f"    Prompt: {prompt}")

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3.1-flash-image",
            contents=[prompt, image],
        )

        for part in response.candidates[0].content.parts:
            if part.text:
                print(part.text)
            elif part.inline_data:
                edited_image = part.as_image()
                edited_image.save(output_path)
                print(f"[+] Success! Edited image saved to: {output_path}")
                return

        print("[!] No image returned. Check the prompt and safety filters.")
    except Exception as e:
        print(f"[!] API Error: {e}")

if __name__ == "__main__":
    edit_image(
        input_path="reference.jpg",
        prompt="Replace the background with a neon cyberpunk city while preserving the subject's lighting and pose.",
        output_path="nano_edit_result.png",
    )
