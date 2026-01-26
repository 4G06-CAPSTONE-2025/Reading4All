'''
============================================================
Multi-Stage Pix2Struct + T5 Pipeline
============================================================

Description:
------------
This script implements a full end-to-end pipeline for training and 
inference using a three-stage workflow:

1. Stage 1: Pix2Struct visual model training and diagram description.
2. Stage 2: Pix2Struct structured model training and component/relationship extraction.
3. Stage 3: T5-based alt-text generation from structured data.

The pipeline:
- Executes training scripts for each stage sequentially.
- Loads inference images from a specified folder.
- Performs inference using the trained models for all stages.
- Saves results in JSON files for each stage.

Directory Structure:
--------------------
BASE_DIR           : Main project directory
MODEL_DIR          : Saved models for all stages
TRAIN_DIR          : Training scripts directory
TEST_DIR           : Directory where inference results are saved
DATA_DIR           : Dataset directory for structured alt-text
INFERENCE_IMAGES_DIR: Folder containing images for inference

Dependencies:
-------------
- Python >=3.8
- PyTorch
- Transformers
- Datasets
- Pillow (PIL)

Key Components:
---------------
1. load_images_from_folder(folder)
   - Loads all .png, .jpg, .jpeg images from a folder.
   - Returns a list of tuples: (filename, PIL.Image object).

2. Training Script Execution
   - Sequentially runs:
       stage1_pix2struct_visual.py
       stage2_pix2struct_structured.py
       stage3_alttext_t5.py
   - Stops pipeline if any script fails.

3. Stage 1 Inference
   - Uses Pix2Struct visual model to describe diagram structure.
   - Outputs JSON: stage1_outputs.json

4. Stage 2 Inference
   - Uses Pix2Struct structured model to extract components and relationships.
   - Outputs JSON: stage2_outputs.json

5. Stage 3 Inference
   - Uses T5 model to generate alt-text from structured data.
   - Loads structured data from structured_alttext.json.
   - Outputs JSON: stage3_outputs.json

Usage:
------
python run_all_stages.py

Outputs:
--------
- stage1_outputs.json
- stage2_outputs.json
- stage3_outputs.json

Notes:
------
- The script automatically detects GPU availability and uses it if available.
- Models are saved in the respective stage directories.
- Subprocesses inherit the project PYTHONPATH for proper module resolution.
'''
import os
import copy
import sys
import subprocess
import torch
from datasets import load_dataset
from transformers import (
    Pix2StructProcessor, Pix2StructForConditionalGeneration,
    T5Tokenizer, T5ForConditionalGeneration
)
from PIL import Image
import json

# ---------------- PATHS ----------------
BASE_DIR = r"C:\Users\nawaa\OneDrive\Desktop\Reading4All\ai"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
ENV = copy.deepcopy(os.environ)
ENV["PYTHONPATH"] = BASE_DIR

MODEL_DIR = os.path.join(BASE_DIR, "model")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
DATA_DIR = os.path.join(BASE_DIR, "data")
INFERENCE_IMAGES_DIR = os.path.join(BASE_DIR, "inference", "images")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- HELPER: LOAD IMAGES ----------------
def load_images_from_folder(folder):
    images = []
    file_names = sorted(os.listdir(folder))
    for f in file_names:
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(folder, f)
            img = Image.open(path).convert("RGB")
            images.append((f, img))
    return images

inference_images = load_images_from_folder(INFERENCE_IMAGES_DIR)
print(f"Loaded {len(inference_images)} images for inference.")

# ---------------- HELPER: RUN TRAINING SCRIPTS ----------------
TRAIN_SCRIPTS = [
    "stage1_pix2struct_visual.py",
    "stage2_pix2struct_structured.py",
    "stage3_alttext_t5.py"
]

for script in TRAIN_SCRIPTS:
    script_path = os.path.join(TRAIN_DIR, script)
    print(f"\n=== Running {script} ===\n")
    try:
        subprocess.run([sys.executable, script_path], check=True, env=ENV)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {script} failed with code {e.returncode}. Stopping pipeline.")
        sys.exit(1)

# ---------------- INFERENCE STAGE 1 ----------------
print("\n=== Stage 1 Inference ===\n")
stage1_model_dir = os.path.join(MODEL_DIR, "stage1_visual", "epoch_2")
processor1 = Pix2StructProcessor.from_pretrained("google/pix2struct-base")
model1 = Pix2StructForConditionalGeneration.from_pretrained(stage1_model_dir, torch_dtype=torch.float16).to(DEVICE)
processor1.save_pretrained(stage1_model_dir)

stage1_results = []
for fname, img in inference_images:
    inputs = processor1(img, text="describe the diagram structure", return_tensors="pt").to(DEVICE)
    out_ids = model1.generate(**inputs)
    out_text = processor1.batch_decode(out_ids, skip_special_tokens=True)[0]
    stage1_results.append({"image": fname, "stage1_desc": out_text})

with open(os.path.join(TEST_DIR, "stage1_outputs.json"), "w") as f:
    json.dump(stage1_results, f, indent=2)

# ---------------- INFERENCE STAGE 2 ----------------
print("\n=== Stage 2 Inference ===\n")
stage2_model_dir = os.path.join(MODEL_DIR, "stage2_structured")
processor2 = Pix2StructProcessor.from_pretrained("google/pix2struct-base")
model2 = Pix2StructForConditionalGeneration.from_pretrained(stage2_model_dir, torch_dtype=torch.float16).to(DEVICE)
processor2.save_pretrained(stage2_model_dir)

stage2_results = []
for fname, img in inference_images:
    inputs = processor2(img, text="list components and relationships", return_tensors="pt").to(DEVICE)
    out_ids = model2.generate(**inputs)
    out_text = processor2.batch_decode(out_ids, skip_special_tokens=True)[0]
    stage2_results.append({"image": fname, "stage2_struct": out_text})

with open(os.path.join(TEST_DIR, "stage2_outputs.json"), "w") as f:
    json.dump(stage2_results, f, indent=2)

# ---------------- INFERENCE STAGE 3 ----------------
print("\n=== Stage 3 Inference ===\n")
stage3_model_dir = os.path.join(MODEL_DIR, "stage3_alttext")
tokenizer3 = T5Tokenizer.from_pretrained("t5-base")
model3 = T5ForConditionalGeneration.from_pretrained(stage3_model_dir).to(DEVICE)
tokenizer3.save_pretrained(stage3_model_dir)

dataset3 = load_dataset("json", data_files=os.path.join(DATA_DIR, "structured_alttext.json"))["train"]

stage3_results = []
for i, sample in enumerate(dataset3):
    inp_ids = tokenizer3(sample["structured"], return_tensors="pt", truncation=True).input_ids.to(DEVICE)
    out_ids = model3.generate(inp_ids)
    out_text = tokenizer3.batch_decode(out_ids, skip_special_tokens=True)[0]
    stage3_results.append({"id": i, "alt_text": out_text})

with open(os.path.join(TEST_DIR, "stage3_outputs.json"), "w") as f:
    json.dump(stage3_results, f, indent=2)

print("\n=== ALL STAGES COMPLETE ===")
print("Inference results saved in:", TEST_DIR)