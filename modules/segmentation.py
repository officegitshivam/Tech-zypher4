import os
import subprocess
from PIL import Image

def isolate_foreground(image: Image.Image, output_dir: str, target_ratio: float = 0.85) -> Image.Image:
    temp_input = os.path.join(output_dir, "temp_render.png")
    temp_output = os.path.join(output_dir, "temp_isolate.png")
    image.save(temp_input)

    try:
        subprocess.run(
            ["rembg", "i", temp_input, temp_output],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except FileNotFoundError:
        raise RuntimeError("rembg executable not found. Ensure 'pip install rembg[cli]' is run.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"rembg execution failed: {e.stderr.decode('utf-8')}")

    rgba = Image.open(temp_output).convert("RGBA")

    bbox = rgba.getbbox()
    if bbox:
        cropped = rgba.crop(bbox)
        canvas_dim = 512
        max_dim = int(canvas_dim * target_ratio)
        cropped.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

        padded = Image.new("RGBA", (canvas_dim, canvas_dim), (0, 0, 0, 0))
        offset = ((canvas_dim - cropped.width) // 2, (canvas_dim - cropped.height) // 2)
        padded.paste(cropped, offset)
        return padded

    # Fallback if bbox is None
    return rgba