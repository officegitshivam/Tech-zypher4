import os 
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import torch
from PIL import Image
import gradio as gr

from modules.vision import process_sketch_contours
from modules.diffusion import DiffusionSynthesizer
from modules.segmentation import isolate_foreground
from modules.reconstruction import MeshRecontructor
from modules.validator import MeshValidatorAndExporter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
OUTPUT_DIR = "generated_assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading standalone Gradio studio on {DEVICE}...")
diffusion_engine = DiffusionSynthesizer(device=DEVICE, dtype=DTYPE)
reconstruction_engine = MeshRecontructor(device=DEVICE)
print("Studio ready.")

def pipeline_orchestrator(input_sketch: Image.Image, prompt: str, control_weight: float):
    if input_sketch is None:
        raise gr.Error("Please upload an image or sketch on the canvas.")

    contours = process_sketch_contours(input_sketch)

    rendered_2d = diffusion_engine.generate(contours, prompt=prompt, controlnet_weight=control_weight)
    isolated_2d = isolate_foreground(rendered_2d, OUTPUT_DIR)
    raw_mesh = reconstruction_engine.reconstruct(isolated_2d)
    obj_path, glb_path, report = MeshValidatorAndExporter.validate_and_export(raw_mesh, OUTPUT_DIR)

    return (
        glb_path,       
        obj_path,       
        glb_path,       
        contours,       
        rendered_2d,    
        isolated_2d,    
        report          
    )

with gr.Blocks(title="Sketch-to-3D Reconstruction Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🖌️ Sketch-to-3D Reconstruction Pipeline")
    gr.Markdown("Convert 2D line sketches into watertight, textured 3D digital assets with automated validation.")

    with gr.Row():
        with gr.Column(scale=1):
            sketch_input = gr.Image(
                type="pil",
                sources=["upload", "clipboard"],
                label="Sketch Input (Upload or Paste)"
            )
            text_prompt = gr.Textbox(
                label="Optional Appearance Prompt",
                placeholder="Optional: e.g., Matte dark walnut wood, brass joints, modern style"
            )
            weight_slider = gr.Slider(
                minimum=0.6, maximum=1.4, value=1.0, step=0.05,
                label="Contour Strictness (ControlNet Scale)"
            )
            generate_btn = gr.Button("Generate 3D Asset", variant="primary")

        with gr.Column(scale=1):
            with gr.Tab("3D Asset Inspection"):
                model_viewer = gr.Model3D(label="Interactive 3D Inspector")
                with gr.Row():
                    download_obj = gr.File(label="Download .OBJ")
                    download_glb = gr.File(label="Download .GLB")
                validation_display = gr.Textbox(label="Mesh Validation Report", lines=6)

            with gr.Tab("Intermediate Diagnostics"):
                diag_contours = gr.Image(type="pil", label="Extracted Contours")
                diag_render = gr.Image(type="pil", label="Synthesized 2D Render")
                diag_isolated = gr.Image(type="pil", label="Isolated Foreground")

    generate_btn.click(
        fn=pipeline_orchestrator,
        inputs=[sketch_input, text_prompt, weight_slider],
        outputs=[
            model_viewer,
            download_obj,
            download_glb,
            diag_contours,
            diag_render,
            diag_isolated,
            validation_display
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)