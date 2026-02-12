"""
Stage 2 Inference Script for AI2D Dataset Using Pix2Struct Model

This script performs structured semantic predictions on a set of input images
using a pre-trained Pix2Struct model (Stage 2 AI2D generic model). The workflow
includes loading the model and processor, preprocessing images, generating
predictions, and saving the results to a JSONL file.

Workflow:
1. Set up paths for the model, input images, and output file.
2. Load the pre-trained Pix2Struct model and processor from the specified directory.
3. Collect all PNG, JPG, and JPEG images from the input directory.
4. For each image:
   - Open and convert the image to RGB.
   - Resize to 224x224 pixels and normalize pixel values to [0,1].
   - Process the image through the Pix2Struct processor.
   - Generate structured semantic predictions using the model.
   - Decode the model output to a text representation.
5. Save all image predictions as JSON objects, one per line, in a JSONL output file.

Output:
- A JSONL file where each line corresponds to an image and contains:
  {
      "image": "<image filename>",
      "prediction": "<model prediction>"
  }

Dependencies:
- torch
- transformers
- numpy
- PIL (Pillow)
- tqdm
- json

Hardware:
- Supports GPU acceleration via CUDA if available.

Example usage:
$ python stage1_pix2struct_visual_test.py
"""

import os
import torch, json
from pathlib import Path
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from tqdm import tqdm
from PIL import Image
import numpy as np

# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
AI_DIR = SCRIPT_DIR.parent        # Reading4All/ai
# PROJECT_ROOT = AI_DIR             # clarity

MODEL_DIR = AI_DIR / "model" / "pix2struct_scicap_stage1_20260122_1946"
MODEL_DIR = MODEL_DIR.as_posix()

IMAGE_DIR = AI_DIR / "inference" / "images"
output_file = AI_DIR / "test" / "alt_text_predictions.jsonl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD MODEL ----------------
processor = Pix2StructProcessor.from_pretrained(MODEL_DIR)
model = Pix2StructForConditionalGeneration.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

# Find all image files
image_paths = sorted([p for p in IMAGE_DIR.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
assert len(image_paths) > 0, f"No images found in {IMAGE_DIR}"

print(f"Found {len(image_paths)} images for inference.")

results = []
# FIXED: Iterate over image_paths, not IMAGE_DIR
for img_path in tqdm(image_paths, desc="Stage 1 inference"):
    with Image.open(img_path) as img:
        # Remove unnecessary copying
        img = img.convert("RGB")
        img = img.resize((224, 224))
        img_np = np.array(img).astype(np.float32) / 255.0
    
    inputs = processor(images=[img_np], return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        # Added dtype specification for autocast
        with torch.autocast("cuda" if DEVICE=="cuda" else "cpu", dtype=torch.float16):
            outputs = model.generate(**inputs, max_length=128, num_beams=5)
        alt_text = processor.decode(outputs[0], skip_special_tokens=True)
    
    results.append({"image": img_path.name, "alt_text": alt_text})

# Save
with open(output_file, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Processing complete. Saved {len(results)} predictions to {output_file}")