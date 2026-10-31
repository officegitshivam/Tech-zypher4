import cv2
import numpy as np
from PIL import Image

def process_sketch_contours(input_image: Image.Image, target_size = (512,512)) -> Image.Image:
    if input_image.mode =="RGBA":
        background = Image.new("RGB", input_image.size, (255, 255, 255))
        background.paste(input_image, mask=input_image.split()[3])
        input_image = background

    img_np = np.array(input_image.convert("RGB"))
    gray = cv2.cvtColor(img_np,cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, target_size, interpolation= cv2.INTER_AREA)

    if np.mean(resized) > 127:
        resized = 255 - resized

    norm = cv2.normalize(resized, None, alpha=0, beta = 255,norm_type = cv2.NORM_MINMAX)
    denoised = cv2.bilateralFilter(norm,5, 75, 75)

    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11, 2
    )

    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)

    edges_rgb = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)