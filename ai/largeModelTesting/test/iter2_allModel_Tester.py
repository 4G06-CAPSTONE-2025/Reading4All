"""
Iterative Multi-Model Inference Script for Pix2Struct and BLIP Models

This script performs inference across multiple trained models (Pix2Struct and BLIP) 
on a set of images, generating structured representations or captions depending on 
the model type. It dynamically resolves the latest usable checkpoint within each 
model directory, supports different architectures, and outputs a CSV summarizing 
the generated text for each model-image pair.

Workflow:
1. Set up paths for model directories, inference images, and output CSV.
2. Resolve the actual model directory containing a checkpoint by locating the latest 'config.json'.
3. Load models and processors for Pix2Struct or BLIP architectures:
   - Pix2Struct: Generates structured semantic text from diagrams.
   - BLIP: Generates image captions.
4. Define a helper function `run_inference` to generate text for a single image given a model and processor.
5. Iterate over all models (excluding iteration 1 models) and all images:
   - Generate text for each image-model pair.
   - Collect results and track failures (e.g., missing processor or checkpoint).
   - Free GPU memory between models to manage resources.
6. Save all results to a CSV file with columns:
   ["model", "image", "generated_text"]
7. Print a summary of any model load failures.

Features:
- Automatic checkpoint resolution and architecture detection.
- Supports both Pix2Struct (structured outputs) and BLIP (captions) models.
- GPU support if available.
- Graceful handling of missing images or inference errors.
- CSV output for downstream analysis.

Dependencies:
- torch
- transformers
- PIL (Pillow)
- csv
- json
- pathlib
- os, sys

Outputs:
- CSV file containing generated text for each model-image pair:
  model | image | generated_text

Example usage:
$ python iter2_allModel_Tester.py
"""
import os
import csv
import json
from pathlib import Path
from PIL import Image
import torch
from transformers import (
    Pix2StructProcessor,
    Pix2StructForConditionalGeneration,
    BlipProcessor,
    BlipForConditionalGeneration,
)

# ---------------- PATHS ----------------
SCRIPT_DIR = Path(__file__).resolve().parent
AI_ROOT = SCRIPT_DIR.parent
MODELS_ROOT = AI_ROOT / "model"
IMAGES_DIR = AI_ROOT / "inference" / "images"
OUTPUT_CSV = SCRIPT_DIR / "iter2_inferenceResults.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- MODEL RESOLUTION ----------------
def resolve_actual_model_dir(run_dir: Path):
    candidates = [p.parent for p in run_dir.rglob("config.json")]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

# ---------------- MODEL LOADING ----------------
def load_model(model_dir: Path):
    try:
        config_path = model_dir / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        model_type_str = config.get("model_type", "").lower()
        archs = config.get("architectures", [])

        # -------- PIX2STRUCT --------
        if (
        "pix2struct" in model_type_str
        or any("Pix2Struct" in a for a in archs)
        or any("T5ForConditionalGeneration" in a for a in archs)
        or (not model_type_str and not archs)
        ):
            try:
                processor = Pix2StructProcessor.from_pretrained(model_dir)
            except:
                print(f"  Processor not found in {model_dir.name}, using base processor.")
                processor = Pix2StructProcessor.from_pretrained("google/pix2struct-base")
            model = Pix2StructForConditionalGeneration.from_pretrained(model_dir)
            return model, processor, "pix2struct", config

        # -------- BLIP --------
        if "blip" in model_type_str or any("Blip" in a for a in archs):
            try:
                processor = BlipProcessor.from_pretrained(model_dir)
            except:
                print(f"  Processor not found in {model_dir.name}, using base BLIP processor.")
                processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained(model_dir)
            return model, processor, "blip", config
    except Exception as e:
        print(f"  Model load error: {e}")
        return None, None, None, None

# ---------------- INFERENCE ----------------
def run_inference(model, processor, model_type, image_path: Path):
    try:
        if not image_path.exists():
            return "[IMAGE NOT FOUND]"

        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=256)

        if model_type == "pix2struct":
            return processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        elif model_type == "blip":
            return processor.decode(output_ids[0], skip_special_tokens=True).strip()
        else:
            return "[UNSUPPORTED MODEL TYPE]"
    except Exception as e:
        return f"[INFERENCE ERROR] {str(e)}"

# ---------------- MAIN ----------------
# ---------------- MAIN ----------------
def main():
    if not MODELS_ROOT.exists() or not IMAGES_DIR.exists():
        print("Models or images directory not found.")
        return

    # Filter out iteration 1 models
    model_dirs = [d for d in MODELS_ROOT.iterdir() if d.is_dir() and "iteration 1 models" not in d.name.lower()]
    image_files = [f for f in IMAGES_DIR.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}]

    print(f"Found {len(model_dirs)} models (excluding iteration 1 models), {len(image_files)} images\n")

    results = []
    failures = []

    for model_dir in model_dirs:
        print(f"\n{'='*60}\nModel: {model_dir.name}\n{'='*60}")
        actual_model_dir = resolve_actual_model_dir(model_dir)
        if actual_model_dir is None:
            print("  No loadable checkpoint found")
            failures.append((model_dir.name, "No loadable checkpoint"))
            continue

        print(f"  Using model path: {actual_model_dir.name}")
        model, processor, model_type, config = load_model(actual_model_dir)

        if model is None:
            failures.append((model_dir.name, "Unsupported or missing processor"))
            continue

        model.to(DEVICE).eval()

        for image_path in image_files:
            print(f"→ Image: {image_path.name}")
            generated_text = run_inference(model, processor, model_type, image_path)
            results.append({
                "model": f"{model_dir.name}/{actual_model_dir.name}",
                "image": image_path.name,
                "generated_text": generated_text
            })

        # free GPU memory between models
        del model
        torch.cuda.empty_cache()

    # SAVE CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "image", "generated_text"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nInference complete. Results saved to: {OUTPUT_CSV}")

    # SUMMARY
    if failures:
        print("\n--- MODEL LOAD FAILURES SUMMARY ---")
        for name, reason in failures:
            print(f"{name}: {reason}")
    else:
        print("\nAll models loaded successfully!")

if __name__ == "__main__":
    main()