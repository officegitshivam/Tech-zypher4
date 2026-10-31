[project_readme (1).md](https://github.com/user-attachments/files/34224759/project_readme.1.md)
# 🖌️ Sketch2Shape AI

### *From a 2D Sketch to an Interactive 3D Digital Asset*

*An end-to-end generative AI system that converts rough hand-drawn 2D sketches and optional natural language prompts into watertight, textured, and inspectable 3D meshes (.GLB / .OBJ) in under 2 minutes on consumer hardware.*

[Quick Start for Judges](#-quick-start-for-judges) • [Problem Statement](#-problem-statement--hackathon-alignment) • [System Architecture](#-system-architecture) • [Pipeline Stages](#-stage-by-stage-pipeline-breakdown) • [Hardware Benchmarks](#-hardware--computational-benchmarks) • [Installation](#-installation--setup) • [Team](#-team-members)

## 📑 Table of Contents

 1. [Problem Statement & Hackathon Alignment](#-problem-statement--hackathon-alignment)
 2. [Quick Start for Judges](#-quick-start-for-judges)
 3. [Key Features](#-key-features)
 4. [System Architecture](#-system-architecture)
 5. [Stage-by-Stage Pipeline Breakdown](#-stage-by-stage-pipeline-breakdown)
 6. [Models & Third-Party Components](#-models--third-party-components)
 7. [Tech Stack](#-tech-stack)
 8. [Hardware & Computational Benchmarks](#-hardware--computational-benchmarks)
 9. [Installation & Setup](#-installation--setup)
10. [Running the Application](#-running-the-application)
11. [Input & Output Specifications](#-input--output-specifications)
12. [Sample Results & Validation Reports](#-sample-results--validation-reports)
13. [Mesh Validation & Topological Repair](#-mesh-validation--topological-repair)
14. [Project Structure](#-project-structure)
15. [Known Limitations](#-known-limitations)
16. [Troubleshooting & FAQs](#-troubleshooting--faqs)
17. [Future Scope](#-future-scope)
18. [Team Members](#-team-members)
19. [Acknowledgments & License](#-acknowledgments--license)

## 🎯 Problem Statement & Hackathon Alignment

### Problem Background (TechZephyr 4.0 | IIT Bhubaneswar)

Converting a creative idea into a detailed three-dimensional digital asset is technically demanding, requiring expertise in geometry creation, mesh editing, texturing, lighting, and rendering. While a simple 2D sketch is the natural first step in expressing an object concept (products, furniture, vehicles, toys), a single-view sketch contains **no explicit depth, volume, or back-surface geometry**. Traditional edge-detection algorithms cannot infer hidden surfaces, while standard generative image models often hallucinate visually appealing 2D outputs without preserving original sketch silhouettes.

### The Challenge

To bridge this gap by building a unified, accessible, and functional end-to-end application that transforms an in-app drawn sketch or an uploaded image (with optional text prompt conditioning) into a plausible, watertight, textured 3D digital model exportable in universal 3D formats (`.glb` / `.obj`).

### Our Solution

**Sketch2Shape AI** implements an asymmetric multi-stage pipeline:

* **Zero-Latency In-App Canvas & Upload Suite**: Interactive HTML5 drawing tool with dynamic brush/eraser controls, paired with an image upload processor featuring contrast enhancement and background isolation.
* **Controlled Cross-Attention 2D Synthesis**: ControlNet scribble conditioning anchored on Stable Diffusion v1.5 to project photorealistic, canonical views while respecting structural contours.
* **Feed-Forward Implicit 3D Field Reconstruction**: High-speed image-to-3D synthesis via TripoSR (Transformer-based triplane query with marching cubes).
* **Robust Native Isosurface Extraction**: Native Windows-compatible isosurface polygonization using `skimage.measure.marching_cubes` to bypass C++/CUDA compilation failures associated with `torchmcubes`.
* **Automated Topological Validation**: Automated mesh manifold verification, normal unification, and watertight sealing via Trimesh.
* **Bespoke 3D WebGL Studio**: Direct browser-native inspection with floor-plane alignment, sRGB vertex color rendering, and beveled UI design.

## ⚡ Quick Start for Judges

Follow these steps to reproduce the working application locally:

### 1. Prerequisites Check

Ensure your host machine runs **Windows OS**, **Python 3.10.11**, and an NVIDIA GPU with **CUDA 12.1+** installed.

### 2. Step 1: Launch Backend Engine (PowerShell)

```powershell
# 1. Clone repository
git clone https://github.com/siddhartha420/Sketch2Shape.git
cd Sketch2Shape

# 2. Create and activate Python 3.10.11 environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install PyTorch (CUDA 12.1) & dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e ./TripoSR

# 4. Start the FastAPI server
python server.py
```

*Wait until the console outputs:*

```
Initializing Pipeline engines on cuda...
Engines Loaded and Ready.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 3. Step 2: Open Client Interface (VS Code / Live Server)

1. Open the project folder in **Visual Studio Code**.
2. Right-click `interface.html` $\rightarrow$ **Open with Live Server** (or open `http://127.0.0.1:5500/interface.html`).
3. Click **CREATE**, sketch an object (e.g., a ceramic mug or chair), type an optional prompt (e.g., `"Teal glazed ceramic, matte rim, studio lighting"`), and click **Generate 3D**.
4. The system automatically executes all 5 stages, verifies the mesh, and redirects straight to `viewer.html` for 360° interactive inspection.

## 🚀 Key Features

* **Dual Input Modes**: Direct HTML5 touch/mouse drawing canvas with dynamic brush/eraser controls or upload existing image sketches (`.png`, `.jpg`, `.jpeg`).
* **Multi-Modal Text Conditioning**: Augment geometric contours with optional descriptive prompts specifying materials, surface roughness, lighting, and color.
* **Engineered for 6 GB VRAM GPUs**: Features model CPU offloading, attention slicing, VAE slicing, and intermediate VRAM cache flushing (`torch.cuda.empty_cache()`), running entirely on laptop-class GPUs (RTX 4050).
* **Guaranteed Volumetric Mesh Output**: Does not output flat cards or depth-map planes. Produces complete, watertight 3D hulls using feed-forward implicit reconstruction.
* **Cross-Platform Marching Cubes (`skimage.measure`)**: Eliminates build-time C++/MSVC compilation dependencies by using `scikit-image` for isosurface extraction instead of unstable `torchmcubes` wheels.
* **Automated Mesh Repair**: Automatically repairs winding inversions, unifies vertex normals, fills open holes, and calculates bounding box volume metrics.
* **Interactive 3D WebGL Turntable**: Orbit controls, camera centering, floor-plane alignment, wireframe modes, and direct `.GLB` / `.OBJ` downloads styled in `#014BAA` (Deep Blue) and `#F8F3F0` (Warm Cream).

## 🏗️ System Architecture

```
                                  USER INTERFACE (HTML5 / CSS / Vanilla JS)
                                     │
                 ┌───────────────────┴────────────────────┐
                 ▼                                        ▼
         [ In-App Canvas ]                       [ File Uploader ]
         (Interactive Drawing)               (.png, .jpg, .jpeg + Sliders)
                 │                                        │
                 └───────────────────┬────────────────────┘
                                     │ + Optional Text Prompt
                                     ▼
                       [ POST /api/generate-3d ]
                                     │
═════════════════════════════════════╪═════════════════════════════════════════════════
                               FASTAPI SERVER (Python 3.10.11)
                                     │
 ┌───────────────────────────────────▼─────────────────────────────────────────────────┐
 │ STAGE 1: Computer Vision Contour Preprocessing (modules/vision.py)                  │
 │ • Grayscale conversion • Bilateral edge-preserving filtering                       │
 │ • Adaptive thresholding • Morphological kernel dilation (512x512)                   │
 └───────────────────────────────────┬─────────────────────────────────────────────────┘
                                     │ Normalized Contours
 ┌───────────────────────────────────▼─────────────────────────────────────────────────┐
 │ STAGE 2: Controlled Diffusion Synthesis (modules/diffusion.py)                      │
 │ • Stable Diffusion v1.5 + ControlNet Scribble (25 Steps, UniPC)                     │
 │ • Sequential CPU Model Offloading + VAE Slicing (VRAM Optimization)                │
 └───────────────────────────────────┬─────────────────────────────────────────────────┘
                                     │ Synthetic 2D Product Render
 ┌───────────────────────────────────▼─────────────────────────────────────────────────┐
 │ STAGE 3: Foreground Matting & Alpha Masking (modules/segmentation.py)               │
 │ • U^2-Net Background Removal (rembg) • Non-zero Bounding Box Crop                   │
 │ • Aspect Ratio Rescaling (85% canonical area) & Centering on 512x512                │
 └───────────────────────────────────┬─────────────────────────────────────────────────┘
                                     │ Alpha-Isolated Subject
 ┌───────────────────────────────────▼─────────────────────────────────────────────────┐
 │ STAGE 4: Feed-Forward 3D Reconstruction (modules/reconstruction.py)                 │
 │ • Alpha-to-RGB Gray Neutral Composition (RGB x a + 0.5 x (1 - a))                   │
 │ • TripoSR Vision Transformer Encoder • Triplane NeRF Density Field Query            │
 │ • Isosurface Extraction: skimage.measure.marching_cubes (Bypassing torchmcubes)     │
 │ • Marching Cubes grid resolution=128 tuned for 6 GB VRAM                           │
 └───────────────────────────────────┬─────────────────────────────────────────────────┘
                                     │ Raw Extracted Mesh
 ┌───────────────────────────────────▼─────────────────────────────────────────────────┐
 │ STAGE 5: Topological Mesh Repair & Export (modules/validator.py)                    │
 │ • Trimesh normal alignment • Inversion correction • Boundary hole filling           │
 │ • Extents depth check (Z > 0.0005) • Export to generated_assets/ (.glb & .obj)      │
 └───────────────────────────────────┬─────────────────────────────────────────────────┘
                                     │
═════════════════════════════════════╪═════════════════════════════════════════════════
                                     │ JSON Response + URLs
                                     ▼
                   [ Interactive WebGL Viewer (viewer.html) ]
                   • Automatic Camera Target & Geometry Normalization
                   • Snapped to Ground Plane Floor (Y = 0)
                   • sRGB Color Encoding with Baked Vertex Colors
```

## 🔬 Stage-by-Stage Pipeline Breakdown

### Stage 1: Computer Vision Contour Preprocessing (`modules/vision.py`)

This stage ingests raw canvas strokes or uploaded raster sketches and standardizes them into structured geometric contours. The module converts inputs to grayscale, applies bilateral filtering to eliminate high-frequency noise while preserving boundary edges, and computes mean luminance to automatically invert dark-on-light drawings. Adaptive Gaussian thresholding and morphological dilation are applied to generate clean, consistent line strokes formatted into a $512 \times 512$ RGB conditioning tensor.

### Stage 2: Controlled 2D Diffusion Synthesis (`modules/diffusion.py`)

This stage translates sparse line structures into a photorealistic, studio-lit canonical product render using Stable Diffusion v1.5 conditioned with ControlNet Scribble. The module injects positive prompt anchors (canonical product view, studio lighting, neutral background) and negative prompts to eliminate artifacts. To ensure compatibility with 6 GB VRAM GPUs, inference runs in FP16 precision using the UniPC Multistep Scheduler (25 steps) alongside model CPU offloading, attention slicing, and VAE slicing.

### Stage 3: Foreground Matting & Alpha Masking (`modules/segmentation.py`)

To isolate the synthesized subject from artificial backgrounds prior to geometric projection, this stage runs automated foreground extraction using the open-source $U^2$-Net architecture (`rembg`). The stage processes the image into an isolated RGBA representation, calculates the exact non-zero bounding box of the foreground object, crops out ambient margins, rescales the subject to maintain an 85% canvas ratio, and centers it on a $512 \times 512$ transparent canvas.

### Stage 4: Feed-Forward 3D Reconstruction (`modules/reconstruction.py`)

This stage reconstructs explicit 3D geometry from the isolated foreground image using TripoSR. The 4-channel RGBA input is first composited over a neutral gray background ($\text{RGB} \times \alpha + 0.5 \times (1.0 - \alpha)$) to avoid tensor dimension mismatches in TripoSR's image tokenizer. The model queries NeRF density fields at an optimized grid resolution of $128^3$. Because `torchmcubes` frequently fails on Windows environments due to missing CUDA/C++ compilers, isosurface polygonization is executed seamlessly via `skimage.measure.marching_cubes` to yield stable, portable triangle meshes.

### Stage 5: Mesh Validation, Repair & Export (`modules/validator.py`)

The extracted polygon mesh undergoes topological validation and geometric repair using Trimesh. The pipeline recalculates and unifies vertex normals, resolves face winding inversions, seals open boundary seams through hole-filling algorithms, and validates non-degeneracy by checking vertex counts ($>100$) and volumetric bounding spans ($Z_{\text{depth}} > 0.0005$). The verified geometry is exported as both binary GLTF (`.glb`) and Wavefront (`.obj`) files with embedded vertex color buffers.

## 📦 Models & Third-Party Components

| Component | Repository / Model Identifier | Source / Provider | License | Purpose in Pipeline | 
| ----- | ----- | ----- | ----- | ----- | 
| **ControlNet** | `lllyasviel/sd-controlnet-scribble` | Hugging Face Hub | Apache 2.0 | Enforces sketch boundary structural adherence | 
| **Diffusion Backbone** | `runwayml/stable-diffusion-v1-5` | Hugging Face Hub | CreativeML OpenRAIL-M | Generates 2D canonical appearance & lighting | 
| **Diffusion Scheduler** | `UniPCMultistepScheduler` | Hugging Face Diffusers | Apache 2.0 | Enables fast diffusion sampling in 25 steps | 
| **Matting Model** | `rembg` ($U^2$-Net) | Open Source / PyPI | MIT License | Isolates foreground object from background | 
| **3D Reconstruction** | `stabilityai/TripoSR` | Stability AI / VAST | MIT License | Transformer-based triplane NeRF to 3D mesh | 
| **Isosurface Extraction** | `skimage.measure.marching_cubes` | Scikit-Image | BSD-3-Clause | Pure-Python/NumPy marching cubes bypass for `torchmcubes` |
| **Geometry Engine** | `trimesh` | mikedh / PyPI | MIT License | Normal repair, watertight sealing, GLB/OBJ export | 
| **API Framework** | `FastAPI` + `Uvicorn` | encode / PyPI | BSD-3-Clause | Asynchronous REST server orchestration | 
| **3D Client Engine** | `Three.js` (r128) | CDN (mrdoob) | MIT License | Interactive WebGL rendering & OrbitControls | 

*Note: No paid or external commercial APIs are used. Weights download automatically on first initialization and are cached locally in `~/.cache/huggingface/` for offline execution.*

## 🛠️ Tech Stack

* **Runtime & Environment**: Python **3.10.11**, CUDA 12.1, Windows PowerShell
* **Deep Learning Framework**: PyTorch 2.1+, Torchvision, Hugging Face `diffusers`, `transformers`, `accelerate`
* **Vision & Geometric Computing**: OpenCV (`opencv-python`), Pillow (PIL), NumPy, Scikit-Image (`skimage.measure`)
* **Mesh Processing**: Trimesh, SciPy, NetworkX
* **Backend Architecture**: FastAPI, Starlette, Uvicorn, Python-Multipart
* **Frontend & Graphics**: HTML5 Canvas API, Vanilla ES6+ JavaScript, Three.js (r128), Tailwind CSS (CDN)

## 💻 Hardware & Computational Benchmarks

The entire system was developed, benchmarked, and stress-tested on an entry-level consumer GPU environment:

* **Target System**: Laptop PC
* **Operating System**: Windows 11 64-bit
* **Graphics Card**: **NVIDIA GeForce RTX 4050 Laptop GPU (6 GB GDDR6 VRAM)**
* **System Memory**: **16 GB DDR5 RAM**
* **Processor**: Intel Core i5 / AMD Ryzen 7 Series

### Execution Time Breakdown (per 3D Model)

| Pipeline Step | Module | Average Duration | Peak VRAM Footprint | 
| ----- | ----- | ----- | ----- | 
| **1. Contour Preprocessing** | `modules/vision.py` | $\approx 0.8\text{ s}$ | $0\text{ MB}$ (CPU execution) | 
| **2. 2D Diffusion Synthesis** | `modules/diffusion.py` | $\approx 42.0\text{ s}$ | $\approx 4.4\text{ GB}$ (FP16 + Slicing) | 
| **3. Foreground Matting** | `modules/segmentation.py` | $\approx 4.5\text{ s}$ | $\approx 0.6\text{ GB}$ (CLI Isolation) | 
| **4. 3D Mesh Reconstruction** | `modules/reconstruction.py` | $\approx 55.0\text{ s}$ | $\approx 4.8\text{ GB}$ (Res: $128^3$, `skimage`) | 
| **5. Topological Mesh Repair** | `modules/validator.py` | $\approx 1.5\text{ s}$ | $0\text{ MB}$ (CPU execution) | 
| **Total Generation Cycle** | **End-to-End** | $\approx 1\text{m } 45\text{s} - 2\text{m } 05\text{s}$ | **Peak:** $\approx 5.1\text{ GB VRAM}$ | 

## 📥 Installation & Setup

### 1. System Requirements

* Windows 10/11 (64-bit)
* Python **3.10.11** installed and added to system `PATH`
* NVIDIA GPU Driver supporting CUDA 12.1+
* Internet connection (required **only once** during initial setup to download model weights $\approx 7\text{ GB}$)

### 2. Environment Setup Steps

```powershell
# Open Windows PowerShell as Administrator or standard developer mode
git clone https://github.com/siddhartha420/Sketch2Shape.git
cd Sketch2Shape

# Create Python 3.10.11 virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install PyTorch with CUDA 12.1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install core system dependencies (including scikit-image for marching cubes)
pip install fastapi uvicorn python-multipart pillow numpy opencv-python diffusers transformers accelerate rembg trimesh scipy scikit-image

# Install TripoSR module
git clone https://github.com/VAST-AI-Research/TripoSR.git
pip install -e ./TripoSR
```

## 🖥️ Running the Application

### 1. Start the Backend Server (Terminal 1)

```powershell
.\.venv\Scripts\Activate.ps1
python server.py
```

*Wait until the models are loaded. You will see:*

```
Initializing Pipeline engines on cuda...
Engines Loaded and Ready.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 2. Start the Frontend Client (Terminal 2 or VS Code)

* **Option A (VS Code Live Server - Recommended)**: Right-click `interface.html` $\rightarrow$ **Open with Live Server**.
* **Option B (Python HTTP Server)**:
  ```powershell
  python -m http.server 5500
  ```
  Open your web browser and navigate to: `http://127.0.0.1:5500/interface.html`

## 🎨 Input & Output Specifications

### Supported Inputs

1. **In-App Digital Canvas**: Draw freely with dynamic pen strokes, responsive eraser sizes, and canvas clear buttons.
2. **File Upload**: Existing sketch files in `.png`, `.jpg`, `.jpeg` formats with adjustable contrast and background noise sliders.
3. **Optional Text Prompt**: Free-form text conditioning (e.g., `"Matte dark walnut wood, brushed brass details, studio lighting"`). *Note: The prompt is entirely optional; sketch geometry alone is fully sufficient for 3D reconstruction.*

### Exported Outputs

1. **Binary GLTF (`.glb`)**: Full standard 3D asset including vertices, indices, normals, and baked vertex colors (`COLOR_0`). Supported natively by Blender, Unity, Unreal Engine, and WebGL viewers.
2. **Wavefront (`.obj`)**: Universal geometry file with vertices and face connectivity (`v`, `f`), compatible with CAD and legacy 3D tools.
3. **Diagnostics Folder (`generated_assets/`)**:
   * `latest_contours.png` (Processed sketch line structure)
   * `latest_render.png` (Synthesized 2D diffusion image)
   * `latest_isolated.png` (Foreground alpha-matted PNG)
   * `reconstructed_asset.glb` & `reconstructed_asset.obj` (Final 3D models)

## 📊 Sample Results & Validation Reports

Every generated model is automatically evaluated prior to export. Below is an authentic sample report generated by `modules/validator.py`:

```
=== MESH VALIDATION REPORT ===
Status: VALID VOLUMETRIC GEOMETRY
Vertices: 20,741
Faces: 41,484
Dimensions (X x Y x Z): 1.14 x 1.28 x 1.05
Watertight Solid: Yes
Formats: .OBJ (Wavefront) | .GLB (Binary GLTF)
Color Space: sRGB Baked Vertex Colors (Full Spectrum Enabled)
```

## 🛡️ Mesh Validation & Topological Repair

Single-view implicit neural field generation can occasionally output non-manifold faces, open boundaries, or inverted surface normals. The `MeshValidatorAndExporter` engine in `modules/validator.py` applies automated remediation:

1. **Normal Unification**: Recomputes vertex normals using `trimesh.repair.fix_normals(mesh)` to ensure uniform light reflection across curved surfaces.
2. **Winding Inversion Correction**: Detects and flips negatively wound triangular faces via `trimesh.repair.fix_inversion(mesh)` so back-faces do not render transparently.
3. **Topological Hole Sealing**: Automatically stitches open boundaries using `trimesh.repair.fill_holes(mesh)`.
4. **Volumetric Extents Verification**: Asserts that the geometry contains genuine depth ($Z_{\text{depth}} > 0.0005$ and $X/Y/Z > 0$) to discard degenerate 2D planes.

## 📂 Project Structure

```
Sketch2Shape/
├── generated_assets/           # Automated storage for .glb, .obj, and diagnostic PNGs
│   ├── latest_contours.png     # Stage 1 output (extracted contours)
│   ├── latest_render.png       # Stage 2 output (diffusion render)
│   ├── latest_isolated.png     # Stage 3 output (alpha-matted subject)
│   ├── reconstructed_asset.glb # Stage 5 output (binary GLTF)
│   └── reconstructed_asset.obj # Stage 5 output (wavefront OBJ)
├── modules/
│   ├── vision.py               # Stage 1: OpenCV filtering & contour preprocessing
│   ├── diffusion.py            # Stage 2: Stable Diffusion + ControlNet scribble engine
│   ├── segmentation.py         # Stage 3: rembg U^2-Net foreground isolation
│   ├── reconstruction.py       # Stage 4: TripoSR implicit field reconstruction (skimage fallback)
│   └── validator.py            # Stage 5: Trimesh repair, validation & multi-format export
├── TripoSR/                    # TripoSR neural architecture submodule
│   ├── tsr/                    # Transformers, NeRF renderer, tokenizers
│   └── ...
├── brain_for_create.js         # Canvas UI logic, IndexedDB storage, and API caller
├── design1.css                 # 3D beveled UI styling (#014BAA & #F8F3F0 palette)
├── interface.html              # Main dashboard (Canvas, upload dialog, history gallery)
├── viewer.html                 # Standalone Three.js WebGL 3D inspection turntable
├── server.py                   # FastAPI REST backend with VRAM optimization
├── app.py                      # Alternative Gradio studio interface
├── requirements.txt            # Pinned Python package dependencies
└── README.md                   # System documentation & hackathon submission report
```

## ⚠️ Known Limitations

1. **Single-View Ambiguity**: While TripoSR infers plausible occluded surfaces, the back side of complex asymmetric objects (e.g., a chair with intricate rear carvings) is estimated based on learned priors rather than ground-truth visibility.
2. **Thin Geometry Quantization**: At a marching cubes resolution of $128^3$, ultra-thin features (such as thin antennae, needles, or paper edges) can experience minor topological bridging.
3. **GPU VRAM Ceiling**: Running concurrent background processes that consume VRAM (e.g., local LLMs or games) may push memory usage beyond 6 GB, triggering a PyTorch OOM exception during the diffusion-to-reconstruction handoff.

## 🔧 Troubleshooting & FAQs

### Q1: `torchmcubes` fails to install or is not recognized during TripoSR execution.
* **Cause**: On Windows systems, `torchmcubes` requires Visual Studio C++ build tools and matching CUDA compiler binaries (`nvcc`), which often fail to link.
* **Solution**: We bypassed `torchmcubes` entirely by routing isosurface extraction through `skimage.measure.marching_cubes` (`pip install scikit-image`). This runs natively across all Windows environments without compilation issues.

### Q2: The server crashes with `RuntimeError: The size of tensor a (4) must match the size of tensor b (3)`.
* **Cause**: An unflattened 4-channel RGBA image from `rembg` was passed directly to TripoSR's 3-channel tokenizer.
* **Solution**: Composite the alpha channel against a neutral gray background ($0.5$) before passing it to `self.model([rgb_image])` as implemented in `modules/reconstruction.py`.

### Q3: CUDA throws `OutOfMemoryError: CUDA out of memory`.
* **Cause**: Residual VRAM cache from Stable Diffusion was not cleared before running the TripoSR NeRF grid sampler.
* **Solution**: Ensure `resolution=128` in `modules/reconstruction.py` and confirm `gc.collect()` + `torch.cuda.empty_cache()` execute between stages.

### Q4: The 3D model loads but appears pitch black or unlit in `viewer.html`.
* **Cause**: Three.js default materials multiply vertex colors by black if the material diffuse color is not explicitly set to white, or if tone mapping is unconfigured.
* **Solution**: Set `child.material.color.setHex(0xffffff)` with `vertexColors: true` and enforce `renderer.outputEncoding = THREE.sRGBEncoding`.

### Q5: The 3D model is floating or clipping into the turntable floor.
* **Solution**: In `viewer.html`, compute the bounding box after normalization, subtract `box.min.y` from the mesh's vertical position to snap the base to $Y = 0$, and place the grid helper flush at $Y = 0$.

## 🔮 Future Scope

* **Adaptive Octree Voxel Sampling**: Implement adaptive hierarchical chunking to allow $256^3$ or $512^3$ resolution extraction on GPUs with $\ge 8\text{ GB}$ VRAM.
* **Automated UV Unwrapping & PBR Baking**: Transition from vertex color buffers to dedicated $2048 \times 2048$ PBR texture atlases (Diffuse, Roughness, Normal maps).
* **Multi-View Sketch Ingestion**: Allow users to sketch both front and side profiles to reduce single-view occlusion ambiguity.
* **WebXR AR Passthrough**: Enable WebXR in `viewer.html` for zero-install mobile Augmented Reality inspection on iOS and Android devices.

## 👥 Team Members

Developed for **TechZephyr 4.0 | IIT Bhubaneswar** by:

* **Siddhartha Sinha** — [@siddhartha420](https://github.com/siddhartha420)  
  *Machine Learning Pipeline, ControlNet Integration, TripoSR Reconstruction Engine, VRAM Optimization & Export Logic.*

* **Shivam Kumar** — [@officegitshivam](https://github.com/officegitshivam)  
  *Frontend Architecture, Interactive HTML5 Canvas, Three.js WebGL Turntable Viewer, 3D Beveled UI Design.*

* **Shreyansh Padhan** — [@shreyanshpadhan8-hub](https://github.com/shreyanshpadhan8-hub)  
  *AI & Geometric Research, Logic Formulation, Vision Preprocessing & Contour Filtration Algorithms.*

* **Mithles Majhi** — [@mithlesmajhi242-dev](https://github.com/mithlesmajhi242-dev)  
  *System Documentation, API Architecture, Validation Metrics, Benchmarking & Reproducibility Analysis.*

## 📜 Acknowledgments & License

* **Stability AI & VAST AI Research** for the open-source release of [TripoSR](https://github.com/VAST-AI-Research/TripoSR).
* **RunwayML & CompVis** for [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5).
* **Lvmin Zhang** for [ControlNet](https://github.com/lllyasviel/ControlNet).
* **Daniel Gatis** for the [rembg](https://github.com/danielgatis/rembg) background matting tool.
* **Scikit-Image** developers for the robust marching cubes algorithm implementation.
* **IIT Bhubaneswar** for hosting the TechZephyr 4.0 Hackathon.

This project is licensed under the **MIT License**. Pretrained weights are governed by their respective licenses (CreativeML OpenRAIL-M and Apache 2.0).
