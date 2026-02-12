"""
Stage 2 Pix2Struct Inference for AI2D Dataset – Structured Diagram Extraction

This script generates structured semantic representations for AI2D dataset images 
using a pre-trained Stage 2 Pix2Struct model. The output is a CSV file mapping each 
image to its structured textual representation, which can be used for downstream 
tasks such as alt-text generation, diagram understanding, or knowledge graph creation.

Workflow:
1. Set up paths for AI2D images, Stage 2 model, and output CSV.
2. Load the Pix2Struct processor and model in inference mode.
3. Define a safe image loader that resizes, pads, and converts images to RGB.
4. Collect all image paths (PNG format) from the AI2D dataset.
5. Process images in batches (default batch size = 8):
   - Load and preprocess images safely.
   - Generate structured representations using the Pix2Struct model.
   - Decode model outputs to text.
6. Aggregate results and save as a CSV file with columns:
   ["image_id", "structured_text"]

Features:
- Safe image loading with resizing and padding.
- Batch processing for faster inference.
- GPU support if available.

Dependencies:
- torch
- transformers
- PIL (Pillow)
- pandas
- tqdm
- os, pathlib

Outputs:
- CSV file containing structured text for each AI2D image:
  image_id | structured_text

Example usage:
$ python cache_stage2_structured_ai2d.py
"""
import os, sys, torch
from pathlib import Path
from PIL import Image, ImageOps
import pandas as pd
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from tqdm import tqdm

# -------------------------------------------------
# PATHS
# -------------------------------------------------
BASE_DIR = r"C:/Users/nawaa/OneDrive/Desktop/Reading4All/ai"
MODEL_DIR = os.path.join(BASE_DIR, "model")

AI2D_ROOT = Path(r"C:/Users/nawaa/Downloads/ai2d-all/ai2d")
IMAGE_DIR = AI2D_ROOT / "images"

STAGE2_MODEL = os.path.join(MODEL_DIR, "stage2_semantic")
OUT_CSV = os.path.join(MODEL_DIR, "ai2d_structured_stage2.csv")

BATCH_SIZE = 8
MAX_LEN = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

# -------------------------------------------------
# LOAD MODEL (INFERENCE ONLY)
# -------------------------------------------------
print("Loading Stage 2 Pix2Struct...")
processor = Pix2StructProcessor.from_pretrained(STAGE2_MODEL)
model = Pix2StructForConditionalGeneration.from_pretrained(
    STAGE2_MODEL, torch_dtype=torch.float32
).to(DEVICE)
model.eval()

# -------------------------------------------------
# SAFE IMAGE LOADER
# -------------------------------------------------
def safe_image(path, size=(224, 224)):
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail(size, Image.LANCZOS)
        pad_w, pad_h = size[0] - img.width, size[1] - img.height
        return ImageOps.expand(
            img,
            (pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2),
            fill=(255,255,255)
        )
    except:
        return None

# -------------------------------------------------
# LOAD IMAGE PATHS
# -------------------------------------------------
image_paths = sorted(IMAGE_DIR.glob("*.png"))
print(f"Found {len(image_paths)} AI2D images")

records = []

# -------------------------------------------------
# INFERENCE LOOP (CACHED)
# -------------------------------------------------
for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Caching structured AI2D"):
    batch_paths = image_paths[i:i + BATCH_SIZE]

    images = []
    names = []

    for p in batch_paths:
        img = safe_image(p)
        if img is not None:
            images.append(img)
            names.append(p.name)

    if not images:
        continue

    inputs = processor(images=images, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_LEN
        )

    texts = processor.batch_decode(outputs, skip_special_tokens=True)

    for name, text in zip(names, texts):
        records.append({
            "image_id": name,
            "structured_text": text
        })

# -------------------------------------------------
# SAVE CSV
# -------------------------------------------------
df = pd.DataFrame(records)
df.to_csv(OUT_CSV, index=False)

print(f"\n DONE")
print(f"Saved {len(df)} structured samples to:")
print(OUT_CSV)