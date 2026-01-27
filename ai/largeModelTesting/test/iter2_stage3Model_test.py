"""
STEM Diagram Alt-Text Generation Pipeline using Stage 2 Pix2Struct and Majunga v3 (FLAN-T5)

This script performs a two-stage inference pipeline to generate concise, 
accessible alt-text for STEM diagrams. It combines a Pix2Struct model 
(Stage 2) to extract structured representations of diagrams with a 
Majunga v3 FLAN-T5 model (Stage 3) to generate natural language alt-text 
from the structured output.

Workflow:
1. Set up project paths for models, unseen images, and output CSV.
2. Initialize logger for progress and error reporting.
3. Load images safely, resizing and padding as necessary.
4. Load Stage 2 Pix2Struct model and processor for structured diagram extraction.
5. Load Majunga v3 FLAN-T5 model and tokenizer for alt-text generation.
6. For each image in the unseen directory:
   - Generate structured representation using Stage 2 Pix2Struct.
   - Build a prompt including the structured output.
   - Generate alt-text using Majunga v3.
   - Record processing time and results.
7. Save results as a CSV file with columns:
   ["image_path", "stage2_structured", "majunga_alt_text", "time_seconds"]

Features:
- Safe image loading with verification, resizing, and padding.
- Two-stage pipeline for structured understanding → natural language alt-text.
- Progress tracking and logging for monitoring.
- GPU support if available.

Dependencies:
- torch
- transformers
- PIL (Pillow)
- pandas
- utils.logging_utils (custom)
- utils.progress_utils (custom)

Outputs:
- CSV file containing alt-text generation results and timing metrics for each image.

Example usage:
$ python iter2_stage3Model_test.py
"""

import sys, os, torch, time
from pathlib import Path
from PIL import Image, ImageOps
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM

# Add ai/ to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from utils.logging_utils import setup_logger
from utils.progress_utils import progress

# ---------------- PATHS ----------------
BASE_DIR = r"C:/Users/nawaa/OneDrive/Desktop/Reading4All/ai"
MODEL_DIR = os.path.join(BASE_DIR, "model")
STAGE2_MODEL = os.path.join(MODEL_DIR, "stage2_semantic")
MAJUNGA_MODEL = os.path.join(MODEL_DIR, "stage3_majunga")
UNSEEN_DIR = r"C:/Users/nawaa/OneDrive/Desktop/Reading4All/ai/inference/images"  # put any unseen, new images here
OUT_FILE = os.path.join(MODEL_DIR, "majunga_v3_inference_results.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

# ---------------- LOGGER ----------------
logger = setup_logger("inference", MODEL_DIR)

# ---------------- SAFE IMAGE LOADING ----------------
def safe_image(img_path, size=(224,224)):
    try:
        with Image.open(img_path) as img:
            img.verify()
        img = Image.open(img_path).convert("RGB")
        img.thumbnail(size, Image.LANCZOS)
        pad_w, pad_h = size[0]-img.width, size[1]-img.height
        return ImageOps.expand(
            img,
            (pad_w//2, pad_h//2, pad_w-pad_w//2, pad_h-pad_h//2),
            fill=(255,255,255)
        )
    except Exception as e:
        logger.info(f"Skipping {img_path} | {e}")
        return None

# ---------------- LOAD STAGE2 PIX2STRUCT ----------------
logger.info("Loading Stage 2 Pix2Struct...")
p2s_processor = Pix2StructProcessor.from_pretrained(STAGE2_MODEL)
p2s_model = Pix2StructForConditionalGeneration.from_pretrained(STAGE2_MODEL).to(DEVICE)
p2s_model.eval()

# ---------------- LOAD MAJUNGA v3 MODEL ----------------
logger.info("Loading Majunga v3 FLAN-T5...")
tokenizer = AutoTokenizer.from_pretrained(MAJUNGA_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MAJUNGA_MODEL).to(DEVICE)
model.eval()

# ---------------- BUILD PROMPT ----------------
def build_prompt(structured):
    return (
        "Instruction: Generate concise, accessible alt text for a STEM diagram.\n\n"
        "Visual structure:\n"
        f"{structured}\n\n"
        "Alt text:"
    )

# ---------------- INFERENCE ----------------
results = []
img_paths = list(Path(UNSEEN_DIR).glob("*.png")) + list(Path(UNSEEN_DIR).glob("*.jpg"))

for img_path in progress(img_paths, desc="Running Inference"):
    start_time = time.time()  # start timing
    
    img = safe_image(img_path)
    if img is None:
        continue

    # Stage 2 → structured
    inputs = p2s_processor(images=[img], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out_ids = p2s_model.generate(**inputs, max_length=512)
    structured = p2s_processor.decode(out_ids[0], skip_special_tokens=True)

    # Majunga → alt text
    prompt = build_prompt(structured)
    tok_inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out_ids = model.generate(
            **tok_inputs,
            max_length=256,
            num_beams=4,
            early_stopping=True
        )
    alt_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)

    end_time = time.time()  # end timing
    elapsed = end_time - start_time

    results.append({
        "image_path": str(img_path),
        "stage2_structured": structured,
        "majunga_alt_text": alt_text,
        "time_seconds": elapsed
    })
    logger.info(f"{img_path.name} processed in {elapsed:.2f}s")

# ---------------- SAVE RESULTS ----------------
import pandas as pd
df = pd.DataFrame(results)
df.to_csv(OUT_FILE, index=False)
logger.info(f"Inference complete! Results saved to {OUT_FILE}")