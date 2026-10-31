import os
import sys
import gc
import numpy as np
import torch
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIPOSR_DIR = os.path.join(CURRENT_DIR, "TripoSR")

if TRIPOSR_DIR not in sys.path:
    sys.path.append(TRIPOSR_DIR)

from tsr.system import TSR

class MeshRecontructor:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt"
        )
        self.model.to(self.device)

    def reconstruct(self, rgba_image: Image.Image):
        if rgba_image.mode != "RGBA":
            rgba_image = rgba_image.convert("RGBA")

        np_img = np.array(rgba_image, dtype=np.float32) / 255.0
        rgb = np_img[:, :, :3]
        alpha = np_img[:, :, 3:4]
        blended = rgb * alpha + 0.5 * (1.0 - alpha)
        rgb_image = Image.fromarray((blended * 255.0).astype(np.uint8))

        # Clear any residual allocations before inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        with torch.no_grad():
            scene_codes = self.model([rgb_image], device=self.device)
            
            # Using resolution=128 dramatically slashes VRAM usage on 6GB GPUs
            meshes = self.model.extract_mesh(
                scene_codes,
                resolution=128,
                has_vertex_color=True
            )
            return meshes[0]