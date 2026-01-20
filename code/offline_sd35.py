import torch
from diffusers import StableDiffusion3InpaintPipeline
from diffusers.utils import load_image
from PIL import Image

# -----------------------------------------------------------------------------
# Offline Alternative: Stable Diffusion 3.5 Large
# -----------------------------------------------------------------------------
# Recommendation: Best for prompt adherence and typography.
# VRAM Requirement: 16GB+ (Optimized) / 24GB (Full)
# -----------------------------------------------------------------------------

def run_sd35_edit(image_path, mask_path, prompt, output_path="sd35_result.png"):
    print("[*] Loading SD3.5 Large Inpainting model...")
    
    pipe = StableDiffusion3InpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large-turbo",
        torch_dtype=torch.bfloat16
    ).to("cuda")

    image = load_image(image_path).resize((1024, 1024))
    mask = load_image(mask_path).resize((1024, 1024))

    print(f"[*] Processing: {prompt}")
    
    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=8, # Turbo is fast
        guidance_scale=0.0,    # Turbo often uses 0 guidance
        strength=0.9,
    ).images[0]

    result.save(output_path)
    print(f"[+] Saved SD3.5 edit to {output_path}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("[*] GPU Detected. Ready for SD3.5.")
    else:
        print("[!] Warning: GPU required.")
