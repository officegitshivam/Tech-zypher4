[project_readme (1).md](https://github.com/user-attachments/files/34226002/project_readme.1.md)
# 🖌️ Sketch2Shape AI: From a 2D Sketch to a 3D Asset
> **TechZephyr 4.0 | 24-Hour Live Hackathon | Computer Vision & Generative AI**  
> *Developed for IIT Bhubaneswar Hackathon Challenge*

[![Python Version](https://img.shields.io/badge/Python-3.10.11-blue.svg)](https://www.python.org/downloads/release/python-31011/)
[![Framework](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Three.js](https://img.shields.io/badge/Three.js-r128-black.svg)](https://threejs.org/)
[![Hardware Tested](https://img.shields.io/badge/Hardware-RTX%204050%20(6GB%20VRAM)-green.svg)]()
[![Platform](https://img.shields.io/badge/Platform-Windows%20OS-blue.svg)]()

---

## 📋 Table of Contents
1. [Executive Summary & Problem Statement](#-executive-summary--problem-statement)
2. [Quick Start for Judges](#-quick-start-for-judges)
3. [Key System Features](#-key-system-features)
4. [System Architecture](#-system-architecture)
5. [Stage-by-Stage Technical Breakdown](#-stage-by-stage-technical-breakdown)
6. [Models & Third-Party Components](#-models--third-party-components)
7. [Technology Stack](#-technology-stack)
8. [Hardware Constraints & Benchmarks](#-hardware-constraints--benchmarks)
9. [Installation & Environment Setup](#-installation--environment-setup)
10. [Running the Application](#-running-the-application)
11. [Input & Output Specifications](#-input--output-specifications)
12. [Mesh Validation & Geometry Integrity](#-mesh-validation--geometry-integrity)
13. [Project Directory Structure](#-project-directory-structure)
14. [Limitations & Trade-Offs](#-limitations--trade-offs)
15. [Troubleshooting Guide](#-troubleshooting-guide)
16. [Future Scope](#-future-scope)
17. [Team Members & Contributions](#-team-members--contributions)
18. [Acknowledgments & Licensing](#-acknowledgments--licensing)

---

## 🎯 Executive Summary & Problem Statement

Transforming a rough conceptual idea into an interactive 3D asset typically requires specialized modeling skills, surface topology reconstruction, UV texturing, lighting mastery, and heavy compute resources. Hand-drawn sketches represent the fastest and most intuitive medium for conceptual prototyping (household objects, furniture, architecture, automotive forms, and toys). However, single-view 2D drawings inherently lack 3D spatial depth, present self-occluded rear geometry, and feature sparse visual contour information.

**Sketch2Shape AI** bridges this gap by delivering an automated, lightweight, end-to-end generative AI pipeline. The system ingests single-view freehand sketches (sketched directly on an in-app canvas or uploaded as images) along with optional natural-language appearance prompts, synthesizes canonical studio-lit product representations via controlled diffusion, reconstructs watertight and textured 3D meshes using fast implicit neural field transformers, and renders them inside a custom, interactive Three.js WebGL inspection studio.

---

## ⚡ Quick Start for Judges

To test and reproduce the application on Windows:

```powershell
# 1. Clone the repository
git clone https://github.com/officegitshivam/Tech-zypher4.git
cd Tech-zypher4

# 2. Set up virtual environment (Python 3.10.11 recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install PyTorch with CUDA 12.1 and project dependencies
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 4. Start the FastAPI backend server
python server.py
```

Open a second terminal or your editor to serve the web interface:
* **Option A (VS Code Live Server - Recommended):** Open the project directory in VS Code, right-click `interface.html`, and click **Open with Live Server**.
* **Option B (Python HTTP Server):**
  ```powershell
  python -m http.server 5500
  ```
  Open `http://127.0.0.1:5500/interface.html` in your browser.

---

## ✨ Key System Features

- **Interactive Canvas with 3D Depth UI:** Desktop sketching canvas designed with a high-contrast `#014BAA` (deep blue) and `#F8F3F0` (warm cream) palette, embossed physical buttons, and a dedicated right-side tool dock.
- **Image Preprocessing Suite:** Ingests `.png`, `.jpg`, and `.jpeg` sketch uploads with client-side aspect-ratio locking, contrast adjustment, noise reduction, and background transparency isolation.
- **Controlled Conditioning:** Employs ControlNet Scribble paired with Stable Diffusion v1.5 to strictly preserve input sketch contours rather than hallucinating unrelated structures.
- **VRAM-Optimized Reconstruction (6 GB Limit):** Engineered with sequential CPU model offloading, attention slicing, VAE slicing, and intermediate VRAM cache clearing (`torch.cuda.empty_cache()`) to guarantee execution on 6 GB VRAM GPUs without memory crashes.
- **Native Isosurface Extraction (`skimage.measure`):** Replaces problematic C++/CUDA compiler dependencies (such as `torchmcubes` on Windows) with pure-Python/NumPy marching cubes via `skimage.measure.marching_cubes`.
- **Standalone WebGL Turntable Viewer:** Custom Three.js viewer with dual-sided PBR lighting, orbit controls, wireframe toggling, coordinate normalization, floor alignment ($Y = 0$), and direct `.glb` and `.obj` downloads.

---

## 🏗️ System Architecture

```
[ In-App Canvas / Image Upload ] + [ Optional Text Prompt ]
                       │
                       ▼
    [ Stage 1: Computer Vision Preprocessing ]
       • Bilateral Filter Denoising
       • Adaptive Gaussian Thresholding
       • Morphological Contour Dilation
                       │
                       ▼
    [ Stage 2: Generative 2D Synthesis ]
       • Stable Diffusion v1.5 + ControlNet Scribble
       • UniPC Multistep Scheduler (25 steps)
       • Sequential CPU Offloading & VAE Slicing
                       │
                       ▼
    [ Stage 3: Foreground Isolation ]
       • rembg U2-Net Alpha Masking
       • 512x512 Canvas Centering & Normalization
                       │
                       ▼
    [ Stage 4: 3D Geometry Extraction ]
       • TripoSR Feed-Forward Reconstruction
       • NeRF-Style Density & RGB Field Estimation
       • skimage.measure.marching_cubes Extraction
                       │
                       ▼
    [ Stage 5: Mesh Validation & Export ]
       • Trimesh Normal Repair, Inversion & Hole Filling
       • Volumetric Depth & Vertex Checks
       • Dual Export: Binary .GLB & Wavefront .OBJ
                       │
                       ▼
    [ WebGL 3D Turntable Viewer (viewer.html) ]
```

---

## 🔬 Stage-by-Stage Technical Breakdown

### Stage 1: Vision Contours & Preprocessing (`modules/vision.py`)
The pipeline standardizes raw drawing strokes or uploaded raster sketches into 512×512 boundary representations. OpenCV converts images to single-channel grayscale, evaluates background luminance to invert dark-on-light drawings, and applies bilateral filtering to suppress jitter while maintaining sharp edge gradients. Adaptive Gaussian thresholding and morphological dilation generate continuous line contours that serve as conditioning tensors for the diffusion model.

### Stage 2: Sketch-Guided Generative Synthesis (`modules/diffusion.py`)
The extracted contours condition Stable Diffusion v1.5 through `lllyasviel/sd-controlnet-scribble`. In tandem with the sketch structure and an optional user prompt, the UniPC Multistep scheduler generates a high-resolution, canonical studio product render in 25 inference steps. Memory management hooks (`enable_model_cpu_offload`, attention slicing, and VAE slicing) keep peak VRAM under 5 GB throughout generation.

### Stage 3: Silhouette Segmentation & Canvas Padding (`modules/segmentation.py`)
Before 3D reconstruction, raw 2D renders are isolated from their generated backgrounds to prevent geometric distortion. Using `rembg` (powered by U2-Net), an alpha matte isolates the subject. The object's bounding box is cropped, resized with aspect-ratio preservation, and placed onto a transparent 512×512 canvas with a uniform 0.85 scaling factor to ensure ideal camera framing for the 3D encoder.

### Stage 4: 3D Triplane Reconstruction (`modules/reconstruction.py`)
The foreground asset enters TripoSR, a feed-forward vision transformer capable of estimating density fields and surface geometry in under 5 seconds. The 4-channel RGBA input is first composited over a neutral gray background ($\text{RGB} \times \alpha + 0.5 \times (1.0 - \alpha)$) to avoid channel mismatches. To ensure cross-platform stability on Windows, isosurface extraction is performed via `skimage.measure.marching_cubes` at an optimized resolution of $128^3$, yielding a clean triangle mesh with baked per-vertex RGB values.

### Stage 5: Mesh Validation, Repair & Export (`modules/validator.py`)
The polygon mesh is processed through `trimesh` to ensure manifold integrity. The validator recalculates vertex normals, corrects winding inversions, fills boundary holes, and checks volumetric depth along the Z-axis ($Z_{\text{depth}} > 0.0005$) to reject flat degenerate geometry. The validated model is exported as both binary GLTF (`.glb`) and Wavefront (`.obj`) into the `generated_assets/` folder.

---

## 📦 Models & Third-Party Components

| Component / Model | Provider / Repository | Purpose | Access / Weights |
| :--- | :--- | :--- | :--- |
| **Stable Diffusion v1.5** | RunwayML / Hugging Face | Base 2D latent diffusion model | Open access (`runwayml/stable-diffusion-v1-5`) |
| **ControlNet Scribble** | Lllyasviel / Hugging Face | Spatial edge conditioning | Open access (`lllyasviel/sd-controlnet-scribble`) |
| **TripoSR** | Stability AI / VAST-AI | Feed-forward image-to-3D reconstruction | Open weights (`stabilityai/TripoSR`) |
| **U2-Net** | rembg / Daniel Gatis | Background isolation & segmentation | Bundled via `rembg[cli]` |
| **Three.js** | Mr.doob / GitHub | Client-side WebGL 3D rendering | MIT License (CDN hosted) |
| **Trimesh** | Mike Dawson-Haggerty | Mesh repair, topology checks & export | MIT License (PyPI) |

*Note: No commercial API keys or paid third-party endpoints are used. All model weights are downloaded from Hugging Face Hub during first-time execution.*

---

## 🛠️ Technology Stack

- **Core Runtime:** Python 3.10.11, CUDA 12.1 / 11.8, Windows OS
- **Backend API:** FastAPI, Uvicorn (ASGI), Python-Multipart
- **Generative AI & ML:** PyTorch, Torchvision, Diffusers, Transformers, Accelerate
- **Computer Vision:** OpenCV (`opencv-python-headless`), Pillow, NumPy
- **Foreground Removal:** Rembg (`u2net`)
- **3D Processing:** Trimesh, SciPy, Scikit-Image (`skimage.measure`)
- **Frontend UI & Visuals:** HTML5 Canvas, CSS3 (Custom 3D Beveled Design), JavaScript (ES6+), Three.js (r128), OrbitControls, GLTFLoader, OBJLoader
- **Alternative GUI:** Gradio (available in `app.py`)

---

## 💻 Hardware Constraints & Benchmarks

The entire system was developed and benchmarked on consumer-tier laptop hardware:

| Parameter | Development & Testing Environment |
| :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 4050 Laptop GPU (6 GB GDDR6 VRAM) |
| **System RAM** | 16 GB DDR5 |
| **Operating System** | Windows 11 Home (x64) |
| **Python Environment** | Python 3.10.11 |

### Performance Metrics (Average Single Run)

| Stage | Operation | Runtime | Peak Resource Usage |
| :--- | :--- | :--- | :--- |
| **Stage 1** | Contour Extraction & Denoising | ~0.8 s | < 250 MB RAM |
| **Stage 2** | ControlNet + SD v1.5 Diffusion (25 steps) | ~150 s | ~4.8 GB VRAM (CPU Offload) |
| **Stage 3** | Rembg Foreground Isolation | ~4.5 s | ~1.2 GB RAM |
| **Stage 4** | TripoSR 3D Mesh Reconstruction | ~20 s | ~3.8 GB VRAM |
| **Stage 5** | Trimesh Repair, Validation & Export | ~1.5 s | ~450 MB RAM |
| **Total** | **End-to-End Pipeline Execution** | **~180 s** | **< 5.2 GB Peak VRAM** |

---

## ⚙️ Installation & Environment Setup

Follow these steps to set up the environment on a clean Windows machine:

### 1. Prerequisites
- **Python 3.10.11** installed and added to your system `PATH`.
- **Git** installed on your system.
- An active internet connection for the initial model downloads (~6 GB total).

### 2. Clone the Repository
```powershell
git clone https://github.com/officegitshivam/Tech-zypher4.git
cd Tech-zypher4
```

### 3. Create a Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install PyTorch with CUDA Support
Install PyTorch with CUDA 12.1 support before installing other dependencies:
```powershell
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

### 5. Install Project Dependencies
Install the remaining pinned packages from `requirements.txt`:
```powershell
pip install -r requirements.txt
```

### 6. Verify Installation
Run this command to verify CUDA acceleration:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 🚀 Running the Application

### Method 1: FastAPI + (Ctrl + F5) (Recommended)

1. **Start the API Server:**
   ```powershell
   python server.py
   ```
   *The server starts at `http://127.0.0.1:8000`. Keep this PowerShell window open.*

2. **Serve the Frontend:**
   - In **VS Code**, right-click `interface.html` and press **Ctrl + F5**.
   - Alternatively, start a local Python HTTP server in a second terminal:
     ```powershell
     python -m http.server 5500
     ```
     Navigate to `http://127.0.0.1:5500/interface.html`.

3. **Interact & Generate:**
   - Click **CREATE** to draw on the canvas, or click **UPLOAD** to submit an existing sketch (`.png`, `.jpg`, `.jpeg`).
   - Add an optional style or material prompt in the right-side control dock.
   - Click **Generate 3D**. Once complete, you will be automatically redirected to `viewer.html` to inspect the 3D model.

### Method 2: Gradio Studio (Alternative Standalone GUI)

If you prefer a single-script evaluation interface:
```powershell
python app.py
```
Open `http://127.0.0.1:7860` in your browser to access the Gradio-based studio.

---

## 📥 Input & Output Specifications

### Supported Inputs
- **In-App Drawing:** HTML5 canvas with real-time brush/eraser tools, variable stroke diameters, and canvas reset.
- **File Uploads:** `.png`, `.jpg`, `.jpeg` formats with interactive contrast adjustment, noise reduction, and background transparency controls.
- **Optional Text Prompt:** Natural-language descriptions of materials, colors, styles, or finishes (e.g., *"Vintage leather armchair, brass studs, studio lighting"*). Prompts are optional; the pipeline falls back to an unconditioned canonical representation if none is provided.

### Generated Outputs
- **Binary GLTF (`.glb`):** Self-contained 3D asset with embedded geometry, normals, and vertex colors.
- **Wavefront OBJ (`.obj`):** Universal 3D format compatible with Blender, Maya, Unity, and Unreal Engine.
- **Diagnostic Artifacts:** Intermediate contour masks, diffusion renders, and foreground-isolated PNGs saved in `generated_assets/`.

---

## 🛡️ Mesh Validation & Geometry Integrity

Before exporting any asset, `modules/validator.py` performs quality and integrity checks:

- **Geometry Thresholding:** Requires a minimum of 100 vertices and 100 faces to reject empty or corrupt outputs.
- **Volumetric Depth Check:** Analyzes the bounding box extents ($X \times Y \times Z$). Flat or 2D-planar surfaces ($Z\text{-span} < 0.0005$) trigger validation warnings to ensure non-degenerate geometry.
- **Topology Repair:** Executes `trimesh.repair.fix_normals()`, `fix_inversion()`, and `fill_holes()` to fix face winding, ensure outward-pointing normals, and improve manifold structure.

---

## 📂 Project Directory Structure

```text
Tech-zypher4/
│
├── generated_assets/           # Pipeline output directory (.glb, .obj, diagnostic images)
│   ├── reconstructed_asset.glb
│   ├── reconstructed_asset.obj
│   └── latest_render.png
│
├── modules/                    # Core technical processing pipeline
│   ├── __init__.py
│   ├── diffusion.py            # Stable Diffusion + ControlNet synthesis
│   ├── reconstruction.py       # TripoSR inference & marching cubes integration
│   ├── segmentation.py         # rembg foreground isolation & canvas padding
│   ├── validator.py            # Mesh repair, volumetric checks & export
│   └── vision.py               # Bilateral filtering & adaptive contour extraction
│
├── app.py                      # Standalone Gradio GUI interface
├── brain_for_create.js         # Frontend drawing engine & API communication
├── design1.css                 # Custom responsive stylesheet (#014BAA & #F8F3F0)
├── interface.html              # Main drawing, upload & project dashboard
├── viewer.html                 # Interactive WebGL 3D turntable viewer (Three.js)
├── requirements.txt            # Pinned Python package dependencies
├── server.py                   # FastAPI backend orchestration server
└── README.md                   # Project documentation
```

---

## ⚠️ Limitations & Trade-Offs

- **Single-View Ambiguity:** Single-view reconstruction cannot fully determine hidden, rear-facing geometry. The system infers plausible symmetric structures, but occluded details may appear simplified.
- **Resolution vs. Memory Trade-Off:** Mesh reconstruction resolution is set to `128` to maintain compatibility with 6 GB VRAM GPUs. While this ensures stable execution without OOM crashes, fine structural details can appear soft.
- **Vertex Colors vs. UV Texture Atlases:** TripoSR assigns color data directly to mesh vertices rather than generating UV texture unwraps. Complex materials with microscopic detail require higher polygon densities.
- **Generation Latency:** The 2-minute processing time is primarily dominated by the 25-step diffusion stage under CPU offload constraints. On systems with 12+ GB VRAM, disabling offloading drops execution time to under 25 seconds.
- **Live Server Issue**: Live server often serves the project from a sub-folder path or appends query params. Constructing a url or navigating directly while the modal/form event is firing triggers Live Server’s reload listener before browser finishes navigation.

---

## 🔧 Troubleshooting Guide

### Q1: `torchmcubes` build or compilation failure on Windows
**Solution:** The pipeline uses `skimage.measure.marching_cubes` to bypass `torchmcubes`. This provides cross-platform isosurface extraction without requiring Visual Studio C++ build tools.

### Q2: CUDA Out-Of-Memory (OOM) during diffusion
**Solution:** Verify that `enable_model_cpu_offload()`, `enable_attention_slicing(1)`, and `enable_vae_slicing()` are active in `modules/diffusion.py`. Explicit `torch.cuda.empty_cache()` and `gc.collect()` calls are included between pipeline stages to clear memory buffers.

### Q3: Model appears invisible or off-center in `viewer.html`
**Solution:** TripoSR coordinates default to an offset bounding box. The `viewer.html` script applies `geometry.center()` directly to the vertex buffers, scales the object to fit the camera frustum, and anchors the base of the mesh to $Y = 0$.

### Q4: CORS errors when accessing assets
**Solution:** The FastAPI backend in `server.py` includes `CORSMiddleware` configured with `allow_origins=["*"]`. Ensure requests are routed through `http://127.0.0.1:8000`.

### Q5: Live server issue
**Solution:** except of starting the program via live server , the alternative used is hard rest by going to interface.html when the server.py is online and then pressing Ctrl + F5
---

## 🔮 Future Scope

- **UV Unwrapping & Texture Maps:** Integrate xatlas or neural texture decoders to bake diffuse, normal, and roughness maps onto lower-poly base meshes.
- **Multi-View Synthesis:** Add zero123-style multi-view generation to reduce rear-surface occlusion ambiguity before running 3D reconstruction.
- **Direct 3D Printing Export:** Integrate automatic mesh manifold verification, wall thickness analysis, and direct `.stl` export for slicing software.
- **Expanded Brush Tools:** Support layered brush opacity, pressure sensitivity via the Pointer Events API, and vector line smoothing.

---

## 👥 Team Members & Contributions

* **Siddhartha Sinha** ([@siddhartha420](https://github.com/siddhartha420))  
  *Machine Learning Pipeline, 3D Reconstruction & Asset Export Engine*  
  Architected the end-to-end ML pipeline, integrating Stable Diffusion ControlNet conditioning, VRAM offload strategies, TripoSR reconstruction, and Trimesh geometric repair.

* **Shivam Kumar** ([@officegitshivam](https://github.com/officegitshivam))  
  *Frontend UI, Systems Integration & WebGL Viewer*  
  Developed the custom user interface, HTML5 canvas drawing engine, responsive layout, Three.js 3D viewer, and client-server API integration.

* **Shreyansh Padhan** ([@shreyanshpadhan8-hub](https://github.com/shreyanshpadhan8-hub))  
  *AI & Geometric Research, Logic Formulation & Vision Algorithms*  
  Researched model candidates, designed contour processing workflows, and contributed to pipeline evaluation and geometric consistency logic.

* **Mithles Majhi** ([@mithlesmajhi242-dev](https://github.com/mithlesmajhi242-dev))  
  *Technical Documentation, System Benchmarking & Validation*  
  Authored project documentation, verified setup instructions across fresh environments, tracked performance metrics, and coordinated submission materials.

---

## 📜 Acknowledgments & Licensing

- **Stability AI & VAST-AI Research:** For the open-source release of [TripoSR](https://github.com/VAST-AI-Research/TripoSR).
- **RunwayML & Hugging Face:** For [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) and the [Diffusers](https://github.com/huggingface/diffusers) ecosystem.
- **IIT Bhubaneswar & TechZephyr 4.0:** For organizing the hackathon and providing the problem statement.

This project is licensed under the **MIT License**. Pretrained model weights remain subject to their respective original licenses (CreativeML Open RAIL-M for Stable Diffusion; Apache 2.0 for TripoSR).
