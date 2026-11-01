const BACKEND_API_URL = "http://127.0.0.1:8000";

const createButton = document.getElementById("create-button");
const eraseButton = document.getElementById("erase-button");
const eraserToolButton = document.getElementById("eraser-tool");
const brushSizeInput = document.getElementById("brush-size");
const brushSizeValue = document.getElementById("brush-size-value");
const eraserSizeInput = document.getElementById("eraser-size");
const eraserSizeValue = document.getElementById("eraser-size-value");
const editorDialog = document.getElementById("editor-dialog");
const closeEditorButton = document.getElementById("close-editor");
const submitDrawingButton = document.getElementById("submit-drawing");
const canvas = document.getElementById("drawing-canvas");
const brushCursor = document.getElementById("brush-cursor");
const saveStatus = document.getElementById("save-status");
const historyGrid = document.getElementById("history-grid");
const historyMessage = document.getElementById("history-message");
const imageUpload = document.getElementById("image-upload");
const uploadsGrid = document.getElementById("uploads-grid");
const uploadsMessage = document.getElementById("uploads-message");
const processingDialog = document.getElementById("processing-dialog");
const closeProcessingButton = document.getElementById("close-processing");
const cancelProcessingButton = document.getElementById("cancel-processing");
const saveProcessedImageButton = document.getElementById("save-processed-image");
const generate3dFromUploadButton = document.getElementById("generate-3d-from-upload");
const processingCanvas = document.getElementById("processing-preview");
const processingPreviewInfo = document.getElementById("processing-preview-info");
const processingStatus = document.getElementById("processing-status");
const resizeWidthInput = document.getElementById("resize-width");
const resizeHeightInput = document.getElementById("resize-height");
const keepAspectRatioInput = document.getElementById("keep-aspect-ratio");
const contrastInput = document.getElementById("image-contrast");
const contrastValue = document.getElementById("image-contrast-value");
const noiseReductionInput = document.getElementById("noise-reduction");
const noiseReductionValue = document.getElementById("noise-reduction-value");
const removeLightBackgroundInput = document.getElementById("remove-light-background");
const backgroundThresholdControl = document.getElementById("background-threshold-control");
const backgroundThresholdInput = document.getElementById("background-threshold");
const backgroundThresholdValue = document.getElementById("background-threshold-value");

// 3D UI Elements
const modelSection = document.getElementById("model-section");
const main3dViewer = document.getElementById("main-3d-viewer");
const downloadGlbBtn = document.getElementById("download-glb-btn");
const downloadObjBtn = document.getElementById("download-obj-btn");
const validationReportLog = document.getElementById("validation-report-log");
const modelStatusText = document.getElementById("model-status-text");
const editorPromptInput = document.getElementById("editor-prompt-input");
const processingPromptInput = document.getElementById("processing-prompt-input");

const context = canvas.getContext("2d");

const MAX_IMAGE_DIMENSION = 8192;
const MAX_IMAGE_PIXELS = 16_000_000;
let isDrawing = false;
let isErasing = false;
let resizeObserver;
let historyDatabasePromise;
let historyObjectUrls = [];
let uploadObjectUrls = [];
let processingImage;
let processingImageUrl;
let processingFile;
let sourceAspectRatio;

// ==========================================
// 3D Backend Pipeline Communication
// ==========================================
async function sendTo3DPipeline(imageBlob, promptText, statusElement, triggerButton) {
  const originalText = triggerButton.textContent;
  triggerButton.disabled = true;
  triggerButton.textContent = "Reconstructing 3D...";
  if (statusElement) statusElement.textContent = "Synthesizing visual and building 3D mesh...";

  const formData = new FormData();
  formData.append("sketch", imageBlob, "sketch.png");
  if (promptText && promptText.trim() !== "") {
    formData.append("prompt", promptText.trim());
  }

  try {
    const response = await fetch(`${BACKEND_API_URL}/api/generate-3d`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Server rejected 3D reconstruction request.");
    }

    const data = await response.json();
    console.log("[PIPELINE SUCCESS] 3D Model created:", data);

    if (statusElement) statusElement.textContent = "3D Asset Generated! Redirecting...";

    // Save paths for viewer.html
    sessionStorage.setItem("last_glb_url", `${BACKEND_API_URL}${data.mesh_glb_url}`);
    sessionStorage.setItem("last_obj_url", `${BACKEND_API_URL}${data.mesh_obj_url}`);

    // Resolve absolute URL relative to current directory and force redirect
    const targetUrl = new URL("viewer.html", window.location.href).href;
    console.log("Navigating to:", targetUrl);
    window.location.assign(targetUrl);

  } catch (error) {
    if (statusElement) statusElement.textContent = `3D Error: ${error.message}`;
    console.error("3D Pipeline Error:", error);
    alert(`3D Generation Failed: ${error.message}`);
  } finally {
    triggerButton.disabled = false;
    triggerButton.textContent = originalText;
  }
}

// ==========================================
// IndexedDB & History Management
// ==========================================
function openHistoryDatabase() {
  if (!("indexedDB" in window)) {
    return Promise.reject(new Error("This browser does not support local drawing history."));
  }
  if (historyDatabasePromise) return historyDatabasePromise;

  historyDatabasePromise = new Promise((resolve, reject) => {
    const request = indexedDB.open("sketch2shape-history", 1);
    request.onupgradeneeded = () => {
      request.result.createObjectStore("drawings", { keyPath: "id" });
    };
    request.onsuccess = () => {
      const database = request.result;
      database.onversionchange = () => {
        database.close();
        historyDatabasePromise = undefined;
      };
      resolve(database);
    };
    request.onerror = () => {
      historyDatabasePromise = undefined;
      reject(request.error || new Error("Could not open drawing history."));
    };
    request.onblocked = () => {
      historyDatabasePromise = undefined;
      reject(new Error("Drawing history is blocked by another open page."));
    };
  });
  return historyDatabasePromise;
}

async function saveDrawingToHistory(drawing) {
  const database = await openHistoryDatabase();
  return new Promise((resolve, reject) => {
    const transaction = database.transaction("drawings", "readwrite");
    transaction.objectStore("drawings").put(drawing);
    transaction.oncomplete = resolve;
    transaction.onerror = () => {
      reject(transaction.error || new Error("Could not save this drawing to history."));
    };
  });
}

async function getSavedDrawings() {
  const database = await openHistoryDatabase();
  return new Promise((resolve, reject) => {
    const request = database.transaction("drawings", "readonly")
      .objectStore("drawings")
      .getAll();
    request.onsuccess = () => {
      resolve(request.result.sort((first, second) =>
        second.createdAt.localeCompare(first.createdAt)
      ));
    };
    request.onerror = () => {
      reject(request.error || new Error("Could not load saved drawings."));
    };
  });
}

async function renderHistory() {
  const drawings = (await getSavedDrawings()).filter((drawing) => drawing.kind !== "upload");
  historyObjectUrls.forEach((url) => URL.revokeObjectURL(url));
  historyObjectUrls = [];
  historyGrid.replaceChildren();

  if (drawings.length === 0) {
    historyMessage.textContent = "No saved drawings yet.";
    return;
  }

  historyMessage.textContent = `${drawings.length} saved drawing${drawings.length === 1 ? "" : "s"}`;
  drawings.forEach((drawing) => {
    const imageUrl = URL.createObjectURL(drawing.image);
    historyObjectUrls.push(imageUrl);

    const card = document.createElement("article");
    card.className = "history-card";

    const image = document.createElement("img");
    image.src = imageUrl;
    image.alt = `Sketch saved ${new Date(drawing.createdAt).toLocaleString()}`;

    const details = document.createElement("div");
    details.className = "history-card-details";

    const timestamp = document.createElement("time");
    timestamp.dateTime = drawing.createdAt;
    timestamp.textContent = new Date(drawing.createdAt).toLocaleString();

    const downloadLink = document.createElement("a");
    downloadLink.href = imageUrl;
    downloadLink.download = `sketch2shape-${drawing.id}.png`;
    downloadLink.textContent = "Download";

    details.append(timestamp, downloadLink);
    card.append(image, details);
    historyGrid.append(card);
  });
}

function formatFileSize(size) {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

async function renderUploadedFiles() {
  const uploads = (await getSavedDrawings()).filter((file) => file.kind === "upload");
  uploadObjectUrls.forEach((url) => URL.revokeObjectURL(url));
  uploadObjectUrls = [];
  uploadsGrid.replaceChildren();

  if (uploads.length === 0) {
    uploadsMessage.textContent = "No uploaded files yet.";
    return;
  }

  uploadsMessage.textContent = `${uploads.length} uploaded file${uploads.length === 1 ? "" : "s"}`;
  uploads.forEach((file) => {
    const fileUrl = URL.createObjectURL(file.image);
    uploadObjectUrls.push(fileUrl);

    const card = document.createElement("article");
    card.className = "history-card";

    const preview = document.createElement("img");
    preview.src = fileUrl;
    preview.alt = file.name;

    const details = document.createElement("div");
    details.className = "history-card-details";

    const fileInfo = document.createElement("span");
    fileInfo.className = "uploaded-file-info";
    fileInfo.textContent = `${file.name} · ${formatFileSize(file.image.size)}`;

    const downloadLink = document.createElement("a");
    downloadLink.href = fileUrl;
    downloadLink.download = file.name;
    downloadLink.textContent = "Download";

    details.append(fileInfo, downloadLink);
    card.append(preview, details);
    uploadsGrid.append(card);
  });
}

function updateProcessingValueLabels() {
  contrastValue.value = `${contrastInput.value}%`;
  contrastValue.textContent = `${contrastInput.value}%`;
  noiseReductionValue.value = Number(noiseReductionInput.value) === 0
    ? "Off"
    : `${noiseReductionInput.value} px`;
  noiseReductionValue.textContent = noiseReductionValue.value;
  backgroundThresholdValue.value = backgroundThresholdInput.value;
  backgroundThresholdValue.textContent = backgroundThresholdInput.value;
  backgroundThresholdControl.hidden = !removeLightBackgroundInput.checked;
}

function renderProcessedPreview() {
  const width = Number(resizeWidthInput.value);
  const height = Number(resizeHeightInput.value);
  if (!processingImage || !Number.isInteger(width) || !Number.isInteger(height) ||
      width < 1 || height < 1) {
    processingStatus.textContent = "Enter a valid width and height.";
    saveProcessedImageButton.disabled = true;
    generate3dFromUploadButton.disabled = true;
    return;
  }

  try {
    processingCanvas.width = width;
    processingCanvas.height = height;
    const previewContext = processingCanvas.getContext("2d", {
      willReadFrequently: removeLightBackgroundInput.checked
    });
    previewContext.clearRect(0, 0, width, height);
    previewContext.imageSmoothingEnabled = true;
    previewContext.imageSmoothingQuality = "high";
    previewContext.filter =
      `contrast(${contrastInput.value}%) blur(${noiseReductionInput.value}px)`;
    previewContext.drawImage(processingImage, 0, 0, width, height);
    previewContext.filter = "none";

    if (removeLightBackgroundInput.checked) {
      const imageData = previewContext.getImageData(0, 0, width, height);
      const pixels = imageData.data;
      const threshold = Number(backgroundThresholdInput.value);
      for (let index = 0; index < pixels.length; index += 4) {
        if (pixels[index] >= threshold &&
            pixels[index + 1] >= threshold &&
            pixels[index + 2] >= threshold) {
          pixels[index + 3] = 0;
        }
      }
      previewContext.putImageData(imageData, 0, 0);
    }

    const effects = [];
    if (Number(contrastInput.value) !== 100) effects.push(`contrast ${contrastInput.value}%`);
    if (Number(noiseReductionInput.value) > 0) effects.push("noise reduction");
    if (removeLightBackgroundInput.checked) effects.push("transparent background");
    processingPreviewInfo.textContent =
      `${width} × ${height} px · PNG${effects.length ? ` · ${effects.join(" · ")}` : ""}`;
    processingStatus.textContent = "";
    saveProcessedImageButton.disabled = false;
    generate3dFromUploadButton.disabled = false;
  } catch (error) {
    processingStatus.textContent = `Could not process this image: ${error.message}`;
    saveProcessedImageButton.disabled = true;
    generate3dFromUploadButton.disabled = true;
  }
}

async function openImageProcessor(file) {
  if (!file.type.startsWith("image/")) {
    throw new Error(`${file.name} is not a supported image file.`);
  }

  const imageUrl = URL.createObjectURL(file);
  const image = new Image();
  try {
    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = () => reject(new Error("The selected image could not be opened."));
      image.src = imageUrl;
    });

    processingFile = file;
    processingImage = image;
    processingImageUrl = imageUrl;
    sourceAspectRatio = image.naturalWidth / image.naturalHeight;

    const scale = Math.min(
      1,
      4096 / Math.max(image.naturalWidth, image.naturalHeight),
      Math.sqrt(MAX_IMAGE_PIXELS / (image.naturalWidth * image.naturalHeight))
    );
    resizeWidthInput.value = String(Math.max(1, Math.round(image.naturalWidth * scale)));
    resizeHeightInput.value = String(Math.max(1, Math.round(image.naturalHeight * scale)));
    contrastInput.value = "100";
    noiseReductionInput.value = "0";
    removeLightBackgroundInput.checked = false;
    backgroundThresholdInput.value = "240";
    saveProcessedImageButton.disabled = false;
    generate3dFromUploadButton.disabled = false;
    processingStatus.textContent = "";
    updateProcessingValueLabels();
    processingDialog.showModal();
    renderProcessedPreview();
  } catch (error) {
    URL.revokeObjectURL(imageUrl);
    throw error;
  }
}

function closeImageProcessor() {
  if (processingImageUrl) URL.revokeObjectURL(processingImageUrl);
  processingImageUrl = undefined;
  processingImage = undefined;
  processingFile = undefined;
  sourceAspectRatio = undefined;
  processingCanvas.width = 0;
  processingCanvas.height = 0;
  processingStatus.textContent = "";
}

function getProcessedPng() {
  return new Promise((resolve, reject) => {
    processingCanvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("Could not create the processed PNG."));
    }, "image/png");
  });
}

function processedFileName(fileName) {
  const baseName = fileName.replace(/\.[^.]+$/, "") || "image";
  const safeBaseName = baseName.replace(/[<>:"/\\|?*\x00-\x1f]/g, "-").trim();
  return `${safeBaseName || "image"}-processed.png`;
}

function downloadDrawing(blob, id) {
  const downloadUrl = URL.createObjectURL(blob);
  const downloadLink = document.createElement("a");
  downloadLink.href = downloadUrl;
  downloadLink.download = `sketch2shape-${id}.png`;
  document.body.appendChild(downloadLink);
  downloadLink.click();
  downloadLink.remove();
  window.setTimeout(() => URL.revokeObjectURL(downloadUrl), 1000);
}

function resizeCanvas() {
  const bounds = canvas.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;

  const pixelRatio = window.devicePixelRatio || 1;
  const previousCanvas = document.createElement("canvas");
  previousCanvas.width = canvas.width;
  previousCanvas.height = canvas.height;
  if (previousCanvas.width && previousCanvas.height) {
    previousCanvas.getContext("2d").drawImage(canvas, 0, 0);
  }

  canvas.width = Math.round(bounds.width * pixelRatio);
  canvas.height = Math.round(bounds.height * pixelRatio);
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.globalCompositeOperation = "source-over";
  context.fillStyle = "#fff";
  context.fillRect(0, 0, bounds.width, bounds.height);

  if (previousCanvas.width && previousCanvas.height) {
    context.drawImage(
      previousCanvas,
      0,
      0,
      previousCanvas.width,
      previousCanvas.height,
      0,
      0,
      bounds.width,
      bounds.height
    );
  }
}

function openEditor() {
  if (!editorDialog.open) editorDialog.showModal();
  saveStatus.textContent = "";
  requestAnimationFrame(() => {
    resizeCanvas();
    if (resizeObserver) resizeObserver.observe(canvas);
    canvas.focus();
  });
}

function clearCanvas(statusMessage = "Canvas reset.") {
  const bounds = canvas.getBoundingClientRect();
  context.globalCompositeOperation = "source-over";
  context.fillStyle = "#fff";
  context.fillRect(0, 0, bounds.width, bounds.height);
  context.beginPath();
  isErasing = false;
  eraserToolButton.setAttribute("aria-pressed", "false");
  brushCursor.classList.remove("is-eraser", "is-drawing");
  saveStatus.textContent = statusMessage;
}

createButton.addEventListener("click", openEditor);
eraseButton.addEventListener("click", () => clearCanvas());
closeEditorButton.addEventListener("click", () => editorDialog.close());

editorDialog.addEventListener("close", () => {
  isDrawing = false;
  if (resizeObserver) {
    resizeObserver.disconnect();
    resizeObserver = undefined;
  }
});

function getCanvasPoint(event) {
  const bounds = canvas.getBoundingClientRect();
  return {
    x: event.clientX - bounds.left,
    y: event.clientY - bounds.top
  };
}

function updateBrushCursor(event) {
  const point = getCanvasPoint(event);
  brushCursor.style.left = `${point.x}px`;
  brushCursor.style.top = `${point.y}px`;
  const diameter = isErasing ? Number(eraserSizeInput.value) : Number(brushSizeInput.value);
  brushCursor.style.width = `${diameter}px`;
  brushCursor.style.height = `${diameter}px`;
  brushCursor.classList.toggle("is-eraser", isErasing);
  brushCursor.classList.add("is-visible");
}

canvas.addEventListener("pointerdown", (event) => {
  if (event.button !== 0 && event.pointerType === "mouse") return;
  updateBrushCursor(event);
  isDrawing = true;
  brushCursor.classList.add("is-drawing");
  canvas.setPointerCapture(event.pointerId);
  const point = getCanvasPoint(event);
  context.beginPath();
  context.moveTo(point.x, point.y);
  context.lineWidth = isErasing ? Number(eraserSizeInput.value) : Number(brushSizeInput.value);
  context.lineCap = "round";
  context.lineJoin = "round";
  context.globalCompositeOperation = isErasing ? "destination-out" : "source-over";
  context.strokeStyle = isErasing ? "rgba(0, 0, 0, 1)" : "#172033";
});

canvas.addEventListener("pointermove", (event) => {
  updateBrushCursor(event);
  if (!isDrawing) return;
  const point = getCanvasPoint(event);
  context.lineTo(point.x, point.y);
  context.stroke();
});

function stopDrawing() {
  isDrawing = false;
  brushCursor.classList.remove("is-drawing");
  context.closePath();
}

canvas.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointercancel", stopDrawing);
canvas.addEventListener("pointerleave", () => {
  if (!isDrawing) brushCursor.classList.remove("is-visible");
});

eraserToolButton.addEventListener("click", () => {
  isErasing = !isErasing;
  eraserToolButton.setAttribute("aria-pressed", String(isErasing));
  saveStatus.textContent = isErasing ? "Eraser selected." : "Pen selected.";
});

brushSizeInput.addEventListener("input", () => {
  brushSizeValue.value = `${brushSizeInput.value} px`;
  brushSizeValue.textContent = `${brushSizeInput.value} px`;
  if (!isErasing && brushCursor.classList.contains("is-visible")) {
    brushCursor.style.width = `${brushSizeInput.value}px`;
    brushCursor.style.height = `${brushSizeInput.value}px`;
  }
});

eraserSizeInput.addEventListener("input", () => {
  eraserSizeValue.value = `${eraserSizeInput.value} px`;
  eraserSizeValue.textContent = `${eraserSizeInput.value} px`;
  if (brushCursor.classList.contains("is-visible")) {
    brushCursor.style.width = `${isErasing ? eraserSizeInput.value : 18}px`;
    brushCursor.style.height = `${isErasing ? eraserSizeInput.value : 18}px`;
  }
});

submitDrawingButton.addEventListener("click", () => {
  canvas.toBlob(async (blob) => {
    if (!blob) {
      saveStatus.textContent = "Could not export canvas drawing.";
      return;
    }

    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    downloadDrawing(blob, id);

    try {
      await saveDrawingToHistory({
        id,
        createdAt: new Date().toISOString(),
        image: blob
      });
      await renderHistory();
    } catch (e) {
      console.warn("History save error:", e);
    }

    const prompt = editorPromptInput.value;
    
    // Send to pipeline (window.location.assign inside will redirect when complete)
    await sendTo3DPipeline(blob, prompt, saveStatus, submitDrawingButton);
  }, "image/png");
});

imageUpload.addEventListener("change", async () => {
  const file = imageUpload.files?.[0];
  if (!file) return;
  imageUpload.value = "";
  try {
    await openImageProcessor(file);
  } catch (error) {
    uploadsMessage.textContent = `Could not open selected image: ${error.message}`;
  }
});

function handleResizeInput(changedInput) {
  if (keepAspectRatioInput.checked && sourceAspectRatio) {
    const changedValue = Number(changedInput.value);
    if (Number.isFinite(changedValue) && changedValue > 0) {
      if (changedInput === resizeWidthInput) {
        resizeHeightInput.value = String(Math.max(1, Math.round(changedValue / sourceAspectRatio)));
      } else {
        resizeWidthInput.value = String(Math.max(1, Math.round(changedValue * sourceAspectRatio)));
      }
    }
  }
  renderProcessedPreview();
}

resizeWidthInput.addEventListener("input", () => handleResizeInput(resizeWidthInput));
resizeHeightInput.addEventListener("input", () => handleResizeInput(resizeHeightInput));
keepAspectRatioInput.addEventListener("change", renderProcessedPreview);
contrastInput.addEventListener("input", () => {
  updateProcessingValueLabels();
  renderProcessedPreview();
});
noiseReductionInput.addEventListener("input", () => {
  updateProcessingValueLabels();
  renderProcessedPreview();
});
removeLightBackgroundInput.addEventListener("change", () => {
  updateProcessingValueLabels();
  renderProcessedPreview();
});
backgroundThresholdInput.addEventListener("input", () => {
  updateProcessingValueLabels();
  renderProcessedPreview();
});

closeProcessingButton.addEventListener("click", () => processingDialog.close());
cancelProcessingButton.addEventListener("click", () => processingDialog.close());
processingDialog.addEventListener("close", closeImageProcessor);

saveProcessedImageButton.addEventListener("click", async () => {
  if (!processingFile || saveProcessedImageButton.disabled) return;
  saveProcessedImageButton.disabled = true;

  try {
    const imageBlob = await getProcessedPng();
    const fileName = processedFileName(processingFile.name);
    await saveDrawingToHistory({
      id: `upload-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      kind: "upload",
      name: fileName,
      createdAt: new Date().toISOString(),
      image: imageBlob
    });
    await renderUploadedFiles();
    processingStatus.textContent = "Saved to browser history.";
  } catch (error) {
    processingStatus.textContent = `Save failed: ${error.message}`;
  } finally {
    saveProcessedImageButton.disabled = false;
  }
});

generate3dFromUploadButton.addEventListener("click", async (event) => {
  event.preventDefault();
  if (!processingFile || generate3dFromUploadButton.disabled) return;

  try {
    processingStatus.textContent = "Preparing uploaded image...";
    const imageBlob = await getProcessedPng();
    const prompt = processingPromptInput.value;

    const fileName = processedFileName(processingFile.name);
    await saveDrawingToHistory({
      id: `upload-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      kind: "upload",
      name: fileName,
      createdAt: new Date().toISOString(),
      image: imageBlob
    });
    await renderUploadedFiles();

    await sendTo3DPipeline(imageBlob, prompt, processingStatus, generate3dFromUploadButton);
    processingDialog.close();
  } catch (err) {
    processingStatus.textContent = `3D error: ${err.message}`;
    console.error("Upload 3D Generation Error:", err);
  }
});

if ("ResizeObserver" in window) {
  resizeObserver = new ResizeObserver(() => {
    if (editorDialog.open) resizeCanvas();
  });
  resizeObserver.observe(canvas);
} else {
  window.addEventListener("resize", () => {
    if (editorDialog.open) resizeCanvas();
  });
}

Promise.all([renderHistory(), renderUploadedFiles()]).catch((error) => {
  historyMessage.textContent = `Could not load saved drawings: ${error.message}`;
  uploadsMessage.textContent = `Could not load uploaded files: ${error.message}`;
});

window.addEventListener("beforeunload", () => {
  historyObjectUrls.forEach((url) => URL.revokeObjectURL(url));
  uploadObjectUrls.forEach((url) => URL.revokeObjectURL(url));
});
