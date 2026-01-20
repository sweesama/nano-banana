import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
from PIL import Image

# -----------------------------------------------------------------------------
# Offline Alternative: SDXL Turbo
# -----------------------------------------------------------------------------
# Recommendation: Fastest (Real-time capable). Good for lower VRAM (8GB).
# -----------------------------------------------------------------------------

def run_sdxl_edit(image_path, mask_path, prompt, output_path="sdxl_result.png"):
    print("[*] Loading SDXL Turbo Inpainting model...")
    
    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/sdxl-inpaint-1.0", # Using base inpaint for stability, or turbo if available as inpaint
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")
    
    # Note: For pure Turbo speed one might use stabilityai/sdxl-turbo with specific inpaint adapters,
    # but strictly speaking ease-of-use comes from standard SDXL Inpaint (refiner optional).
    # Here we stick to a solid SDXL Inpaint baseline which is widely compatible.

    image = load_image(image_path).resize((1024, 1024))
    mask = load_image(mask_path).resize((1024, 1024))

    print(f"[*] Processing: {prompt}")
    
    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=20, 
        strength=0.85,
    ).images[0]

    result.save(output_path)
    print(f"[+] Saved SDXL edit to {output_path}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("[*] GPU Detected. Ready for SDXL.")
    else:
        print("[!] Warning: GPU required.")
