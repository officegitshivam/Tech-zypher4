import os
import sys
import io
import gc
import traceback

# Ensure project root is prioritized in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from typing import Optional
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from modules.vision import process_sketch_contours
from modules.diffusion import DiffusionSynthesizer
from modules.segmentation import isolate_foreground
from modules.reconstruction import MeshRecontructor
from modules.validator import MeshValidatorAndExporter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

OUTPUT_DIR = os.path.abspath("generated_assets")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Initializing Pipeline engines on {DEVICE}...")
diffusion_engine = DiffusionSynthesizer(device=DEVICE, dtype=DTYPE)
reconstruction_engine = MeshRecontructor(device=DEVICE)
print("Engines Loaded and Ready.")

app = FastAPI(title="Sketch-to-3D Reconstruction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/assets", StaticFiles(directory=OUTPUT_DIR), name="assets")

@app.post("/api/generate-3d")
async def generate_3d_endpoint(
    sketch: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    control_weight: Optional[float] = Form(1.0)
):
    try:
        contents = await sketch.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        print("[ERROR] Failed to read uploaded sketch image:")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Invalid Image upload: {str(e)}")

    try:
        # Step 1: Vision / Contours
        print("[PIPELINE] 1/4 - Processing sketch contours...")
        contours = process_sketch_contours(pil_image)
        contours_path = os.path.join(OUTPUT_DIR, "latest_contours.png")
        contours.save(contours_path)

        # Step 2: 2D Diffusion Synthesis
        print(f"[PIPELINE] 2/4 - Running diffusion (Prompt: '{prompt}')...")
        rendered_2d = diffusion_engine.generate(contours, prompt=prompt, controlnet_weight=control_weight)
        render_path = os.path.join(OUTPUT_DIR, "latest_render.png")
        rendered_2d.save(render_path)

        # Clear PyTorch VRAM before 3D reconstruction to prevent OOM errors
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 3: Foreground Segmentation
        print("[PIPELINE] 3/4 - Isolating foreground with rembg...")
        isolated_2d = isolate_foreground(rendered_2d, OUTPUT_DIR)
        isolated_path = os.path.join(OUTPUT_DIR, "latest_isolated.png")
        isolated_2d.save(isolated_path)

        # Clear VRAM again before marching cubes / geometry extraction
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 4: 3D Reconstruction (TripoSR)
        print("[PIPELINE] 4/4 - Reconstructing 3D mesh...")
        raw_mesh = reconstruction_engine.reconstruct(isolated_2d)

        # Step 5: Mesh Validation & Export
        print("[PIPELINE] Validating and exporting mesh (.glb / .obj)...")
        obj_path, glb_path, report = MeshValidatorAndExporter.validate_and_export(raw_mesh, OUTPUT_DIR)
        glb_filename = os.path.basename(glb_path)
        obj_filename = os.path.basename(obj_path)

        print("[PIPELINE] 3D Generation completed successfully!")
        return {
            "status": "success",
            "validation_report": report,
            "prompt_applied": prompt if prompt and prompt.strip() else "Neutral fallback (Unconditioned)",
            "mesh_glb_url": f"/assets/{glb_filename}",
            "mesh_obj_url": f"/assets/{obj_filename}",
            "diagnostics": {
                "contours_url": "/assets/latest_contours.png",
                "render_url": "/assets/latest_render.png",
                "isolated_url": "/assets/latest_isolated.png"
            }
        }
    except Exception as e:
        print("\n[CRITICAL ERROR] Pipeline failed with the following traceback:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

@app.get("/api/download/{filename}")
async def download_asset(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)
    raise HTTPException(status_code=404, detail="Requested asset does not exist.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)