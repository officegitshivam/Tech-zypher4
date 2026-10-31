import gc
import torch
from typing import Optional
from PIL import Image
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

class DiffusionSynthesizer:
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.device = device
        self.dtype = dtype

        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-scribble",
            torch_dtype=self.dtype
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            torch_dtype=self.dtype,
            safety_checker=None
        )

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        if torch.cuda.is_available() and self.device == "cuda":
            if hasattr(self.pipe, "enable_model_cpu_offload"):
                self.pipe.enable_model_cpu_offload()
            else:
                self.pipe.to(self.device)

            if hasattr(self.pipe, "enable_attention_slicing"):
                self.pipe.enable_attention_slicing(1)
            if hasattr(self.pipe, "enable_vae_slicing"):
                self.pipe.enable_vae_slicing()
        else:
            self.pipe.to(self.device)

    def generate(
        self,
        sketch_condition: Image.Image,
        prompt: Optional[str] = None,
        controlnet_weight: float = 1.0,
        steps: int = 25
    ) -> Image.Image:
        if not prompt or not prompt.strip():
            base_prompt = "a clean, high-quality 3D digital product asset, neutral material finish, smooth surface"
        else:
            base_prompt = prompt.strip()

        enhanced_prompt = (
            f"{base_prompt}, canonical product view, studio lighting, "
            f"neutral solid background, high resolution, highly detailed"
        )
        negative_prompt = (
            "low quality, blurry, distorted, cluttered background, multiple objects, "
            "flat, cropped, cut off edges, extreme close-up, bad geometry"
        )

        with torch.inference_mode():
            result = self.pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                image=sketch_condition,
                num_inference_steps=steps,
                guidance_scale=7.5,
                controlnet_conditioning_scale=controlnet_weight,
            ).images[0]

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result