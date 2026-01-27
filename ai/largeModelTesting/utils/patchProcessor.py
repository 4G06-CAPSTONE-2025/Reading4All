'''
Description:
Offline preprocessing utility for the SciCap dataset, designed to prepare image-caption
pairs for Pix2Struct model training. The script performs the following key tasks:

1. Loads filtered JSON files containing valid image paths and captions for each dataset split 
   ('train', 'val', 'test').
2. Maps preprocessed image folders for quick lookup and validation.
3. Converts images to RGB and processes them into flattened patches using the Pix2StructProcessor,
   with optional GPU acceleration.
4. Handles memory efficiently by saving preprocessed data in batches to disk as .pt files.
5. Includes error handling for missing or corrupted images to ensure robust preprocessing.

Configurable parameters include image size, maximum patches, batch size, and device selection
(CPU/GPU). The output is a set of .pt files ready for downstream model training or evaluation.
'''

import os
import json
import torch
from pathlib import Path
from PIL import Image
from transformers import Pix2StructProcessor
from tqdm import tqdm

# ==================== CONFIG ===============================
BASE_DATA_DIR = r"C:\Users\nawaa\Downloads\scicap_data_preprocessed"
IMAGE_DIRS = {
    "train": [
        BASE_DATA_DIR + r"\SciCap-Yes-Subfig-Img\train",
        BASE_DATA_DIR + r"\Scicap-No-Subfig-Img\train"
    ],
    "val": [
        BASE_DATA_DIR + r"\Scicap-No-Subfig-Img\val"
    ],
    "test": [
        BASE_DATA_DIR + r"\Scicap-No-Subfig-Img\test"
    ]
}
JSON_DIR = BASE_DATA_DIR + r"\Filtered-Captions"

OUTPUT_DIR = BASE_DATA_DIR + r"\Preprocessed-Patches"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = 512
MAX_PATCHES = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== LOAD PROCESSOR ======================
processor = Pix2StructProcessor.from_pretrained("google/pix2struct-base")
processor.image_processor.max_patches = MAX_PATCHES

# ==================== LOAD JSON ===========================
def load_json(split):
    json_file = os.path.join(JSON_DIR, f"{split}_filtered.json")
    if not os.path.exists(json_file):
        print(f"JSON not found for {split}: {json_file}")
        return []
    with open(json_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    return records

# ==================== PREPROCESS (BATCH + GPU SAFE) ==========================
def preprocess_split(split, batch_size=1000):
    print(f"\nPreprocessing {split} split...")

    records = load_json(split)
    if not records:
        return

    # Map image filenames to full paths for fast lookup
    img_paths = {}
    for folder in IMAGE_DIRS.get(split, []):
        for img_file in Path(folder).glob("*.png"):
            img_paths[img_file.name] = str(img_file)

    processed_data = []
    batch_count = 0

    for idx, rec in enumerate(tqdm(records, desc=f"{split}")):
        img_path = rec.get("image_path", "")
        caption = str(rec.get("caption", ""))

        if not img_path or not os.path.exists(img_path):
            print(f"[WARNING] Missing image: {img_path}")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except:
            print(f"[WARNING] Failed to open image: {img_path}")
            continue

        try:
            # Process image and move to device if GPU is available
            image_input = processor(images=img, return_tensors="pt", max_patches=MAX_PATCHES)
            if DEVICE == "cuda":
                image_input = image_input.to(DEVICE)

            flattened_patches = image_input.flattened_patches.squeeze(0)  # [patches, dim]
        except Exception as e:
            print(f"[WARNING] Failed to process image: {img_path} | Error: {e}")
            continue

        processed_data.append({
            "flattened_patches": flattened_patches.cpu(),  # move back to CPU before saving
            "caption": caption
        })

        # Save in batches to avoid memory blow-up
        if (idx + 1) % batch_size == 0:
            batch_file = os.path.join(OUTPUT_DIR, f"{split}_processed_batch_{batch_count}.pt")
            torch.save(processed_data, batch_file)
            print(f"[INFO] Saved batch {batch_count}: {len(processed_data)} samples -> {batch_file}")
            processed_data = []  # free memory
            batch_count += 1

    # Save any remaining data
    if processed_data:
        batch_file = os.path.join(OUTPUT_DIR, f"{split}_processed_batch_{batch_count}.pt")
        torch.save(processed_data, batch_file)
        print(f"[INFO] Saved final batch {batch_count}: {len(processed_data)} samples -> {batch_file}")

# ==================== MAIN ================================
if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        preprocess_split(split, batch_size=1000)