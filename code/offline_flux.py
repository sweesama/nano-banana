import torch
from diffusers import FluxInpaintPipeline
from diffusers.utils import load_image
from PIL import Image

# -----------------------------------------------------------------------------
# Offline / Open Source Alternative Runner
# -----------------------------------------------------------------------------
# Nano Banana is a cloud model. For TRUE offline editing on your own GPU, 
# we recommend using FLUX.1 (the state-of-the-art open model).
#
# Prerequisites:
# 1. Ensure you have a GPU with 12GB+ VRAM (NVIDIA)
# 2. pip install torch diffusers transformers accelerate protobuf sentencepiece
# -----------------------------------------------------------------------------

def run_offline_edit(image_path, mask_path, prompt, output_path="offline_result.png"):
    """
    Runs local inpainting using FLUX.1-schnell (Fast, High Quality).
    """
    print("[*] Loading FLUX.1-schnell model (First run will download ~20GB)...")
    
    # Load Pipeline
    pipe = FluxInpaintPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16
    ).to("cuda")

    # Load Images
    image = load_image(image_path).resize((1024, 1024))
    mask = load_image(mask_path).resize((1024, 1024))

    print(f"[*] Processing: {prompt}")
    
    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        guidance_scale=3.5,
        num_inference_steps=4, # Schnell needs very few steps
        max_sequence_length=256,
        strength=0.9,
    ).images[0]

    result.save(output_path)
    print(f"[+] Saved offline edit to {output_path}")

if __name__ == "__main__":
    # Example check
    if torch.cuda.is_available():
        print(f"[*] GPU Detected: {torch.cuda.get_device_name(0)}")
        # run_offline_edit("photo.jpg", "mask.png", "A futuristic robot")
    else:
        print("[!] Warning: No CUDA GPU detected. FLUX requires a GPU to run efficiently.")
