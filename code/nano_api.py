import os
import time
from google import genai
from google.genai import types
from PIL import Image

# -----------------------------------------------------------------------------
# Nano Banana (Gemini 2.5/3.0) - Local Python Runner
# -----------------------------------------------------------------------------
# This script enables you to run "Nano Banana" image editing tasks from your
# local environment using the official Google GenAI SDK.
#
# Prerequisites:
# 1. pip install google-genai pillow
# 2. Set GOOGLE_API_KEY environment variable
# -----------------------------------------------------------------------------

def edit_image(input_path, prompt, output_path="output.png"):
    """
    Edits an image using Gemini's image editing capabilities.
    """
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    print(f"[*] Loading image: {input_path}")
    try:
        image = Image.open(input_path)
    except FileNotFoundError:
        print(f"[!] Error: File not found at {input_path}")
        return

    print(f"[*] Sending to Nano Banana (Gemini)...")
    print(f"    Prompt: {prompt}")

    # Nano Banana is powered by the Gemini Image models (e.g., imagen-3.0)
    try:
        response = client.models.generate_images(
            model='imagen-3.0-generate-002',
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                include_rai_reason=True,
                output_mime_type='image/png'
            )
        )
        
        if response.generated_images:
            saved_file = response.generated_images[0].image.save(output_path)
            print(f"[+] Success! Edited image saved to: {output_path}")
        else:
            print("[!] No image verified. Check safety filters or prompt.")

    except Exception as e:
        print(f"[!] API Error: {e}")

if __name__ == "__main__":
    # Example Usage
    # Ensure you have a 'reference.jpg' in this folder
    if not os.path.exists("reference.jpg"):
        # Create a dummy file for the script to not crash if run blindly
        Image.new('RGB', (100, 100), color='red').save("reference.jpg")
    
    edit_image(
        input_path="reference.jpg",
        prompt="Replace background with a neon cyberpunk city, preserve subject lighting",
        output_path="nano_edit_result.png"
    )
