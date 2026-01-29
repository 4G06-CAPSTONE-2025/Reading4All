import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration, AutoTokenizer
from tqdm import tqdm
import json

# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
AI_DIR = SCRIPT_DIR.parent  # Reading4All/ai
MODEL_DIR = AI_DIR / "model" / "stage2_ai2d_generic/epoch_3"
MODEL_DIR = MODEL_DIR.as_posix()

IMAGE_DIR = AI_DIR / "inference" / "images"
OUTPUT_FILE = AI_DIR / "pix2struct_stage2_GENpredictions_v1.jsonl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# LOAD MODEL + PROCESSOR
# ============================================================

assert os.path.isdir(MODEL_DIR), f"Model dir not found: {MODEL_DIR}"
print("Model directory contents:", os.listdir(MODEL_DIR))

print("Loading model...")
model = Pix2StructForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

processor = Pix2StructProcessor.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model.to(DEVICE)
model.eval()
print(f"Model loaded on {DEVICE}")

# ============================================================
# LOAD IMAGES
# ============================================================

image_paths = sorted([p for p in IMAGE_DIR.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
assert len(image_paths) > 0, f"No images found in {IMAGE_DIR}"

print(f"Found {len(image_paths)} images for inference.")

# ============================================================
# INFERENCE
# ============================================================

results = []

for img_path in tqdm(image_paths, desc="Running Stage2 inference"):
    # Open image
    with Image.open(img_path) as img:
        orig_img = img.copy()
        img = img.convert("RGB")
        img = img.resize((224, 224))
        img_np = np.array(img).astype(np.float32) / 255.0

    # Processor
    inputs = processor(images=[img_np], return_tensors="pt").to(DEVICE)

    # Generate structured semantic target
    with torch.no_grad(), torch.autocast("cuda" if DEVICE=="cuda" else "cpu"):
        outputs = model.generate(
            **inputs,
            max_length=512,  # same as tokenizer max_length
            do_sample=False,  # deterministic for testing
            num_beams=3
        )

    # Decode
    prediction = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

    results.append({
        "image": str(img_path.name),
        "prediction": prediction
    })

# ============================================================
# SAVE TO JSONL
# ============================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Inference complete. Predictions saved to {OUTPUT_FILE}")