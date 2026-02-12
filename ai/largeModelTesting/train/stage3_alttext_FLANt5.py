"""
Stage 3 – MAJUNGA: Structured-to-Alt-Text Generation Training Script
===================================================================

This script implements **Stage 3 (MAJUNGA)** of a multi-stage alt-text generation
pipeline for STEM diagrams. It trains a **text-to-text language model (FLAN-T5)**
to generate concise, accessible alt text from *structured visual descriptions*
produced by a frozen vision-language model.

The pipeline architecture is:

    Diagram Image 
        to 
    Stage 2 (Frozen Pix2Struct)
        → Structured visual description
        to
    Stage 3 (FLAN-T5, trainable)
        → Accessible alt text

--------------------------------------------------------------------
Hardware & Platform Assumptions
--------------------------------------------------------------------
• Operating System: Windows 10 / 11
• GPU: NVIDIA RTX / GTX (tested on consumer GPUs)
• CUDA-enabled PyTorch
• fp32 training (safe for Windows + consumer GPUs)
• Batch size intentionally kept small for GPU stability

--------------------------------------------------------------------
Key Components
--------------------------------------------------------------------
1. Reproducibility & Safety
   - Fixed random seeds
   - Deterministic CUDA behavior
   - Periodic health checks and cooldowns to prevent GPU overheating

2. Stage 2 Model (Frozen)
   - Pix2Struct model trained in previous stages
   - Converts diagram images into structured textual descriptions
   - Used in inference-only mode (no gradients)

3. Training Data Sources
   a) AI2D Dataset
      - Loaded from preprocessed `.pt` tensor files
      - Images reconstructed from tensors
      - Uses placeholder alt text to regularize language generation

   b) Custom Annotated Dataset
      - CSV with image paths and human-written alt text
      - Provides high-quality supervision signal

4. Prompt Design
   - Structured visual description injected into a fixed instruction template
   - Encourages concise, accessible alt-text output aligned with WCAG principles

5. Stage 3 Model
   - Model: google/flan-t5-base
   - Objective: sequence-to-sequence text generation
   - Input: structured visual prompt
   - Output: alt text

--------------------------------------------------------------------
Training Configuration
--------------------------------------------------------------------
• Epochs: 3
• Batch size: 1 (memory-safe for Windows GPUs)
• Optimizer: AdamW
• Learning rate: 3e-5
• Gradient clipping enabled
• Padding tokens masked with -100 for loss computation

--------------------------------------------------------------------
Outputs
--------------------------------------------------------------------
• Trained FLAN-T5 model saved to:
      ai/model/stage3_majunga/
• Tokenizer saved alongside model
• Training metrics logged periodically
• Configuration snapshot stored for reproducibility

--------------------------------------------------------------------
Intended Use
--------------------------------------------------------------------
This script is designed for:
• Research-grade accessibility tooling
• Diagram-to-alt-text generation pipelines
• Educational and STEM accessibility applications
• Iterative multi-stage vision-language systems

This stage should be run **after** Stage 2 Pix2Struct training is complete.

--------------------------------------------------------------------
"""

import sys, os, torch, random, json
from pathlib import Path
from PIL import Image
import pandas as pd
from datasets import Dataset
from transformers import (
    Pix2StructProcessor,
    Pix2StructForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from torch.optim import AdamW

# -------------------------------------------------
# PATH SETUP
# -------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from utils.reproducibility import set_seed, save_config
from utils.logging_utils import setup_logger, log_metrics
from utils.progress_utils import progress
from utils.safety_utils import log_health, cooldown

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
BASE_DIR = r"C:/Users/nawaa/OneDrive/Desktop/Reading4All/ai"
MODEL_DIR = os.path.join(BASE_DIR, "model")

AI2D_ROOT = r"C:/Users/nawaa/Downloads/ai2d-all/ai2d"
PREPROCESS_DIR = Path(r"C:/Users/nawaa/Downloads/ai2d-all/preprocessed")
CUSTOM_CSV = r"C:/Users/nawaa/Downloads/annotated_physics_data(Sheet1)_UTF8preserved.csv"

STAGE2_MODEL = os.path.join(MODEL_DIR, "stage2_semantic")
OUT_DIR = os.path.join(MODEL_DIR, "stage3_majunga")

CFG = {
    "model": "google/flan-t5-base",
    "epochs": 3,
    "batch_size": 1,
    "lr": 3e-5,
    "seed": 42,
    "max_input_len": 512,
    "max_output_len": 256,
    "p2s_batch_size": 16
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------
# SETUP
# -------------------------------------------------
set_seed(CFG["seed"])
save_config(CFG, OUT_DIR)
logger = setup_logger("stage3_majunga", OUT_DIR)
logger.info(f"Device: {DEVICE}")

# -------------------------------------------------
# LOAD STAGE 2 (FROZEN)
# -------------------------------------------------
logger.info("Loading Stage 2 Pix2Struct...")
p2s_processor = Pix2StructProcessor.from_pretrained(STAGE2_MODEL)
p2s_model = Pix2StructForConditionalGeneration.from_pretrained(
    STAGE2_MODEL, torch_dtype=torch.float32
).to(DEVICE)
p2s_model.eval()

# -------------------------------------------------
# LOAD PREPROCESSED .pt IMAGE
# -------------------------------------------------
def load_preprocessed_pt(pt_path):
    try:
        tensor = torch.load(pt_path)           # C,H,W in [0,1]
        tensor = (tensor * 255).byte()         # uint8
        img = tensor.permute(1, 2, 0).numpy()  # H,W,C
        return Image.fromarray(img)
    except Exception as e:
        print(f"Failed to load {pt_path}: {e}")
        return None

# -------------------------------------------------
# STRUCTURED PROMPT
# -------------------------------------------------
def build_prompt(structured):
    return (
        "Instruction: Generate concise, accessible alt text for a STEM diagram.\n\n"
        "Visual structure:\n"
        f"{structured}\n\n"
        "Alt text:"
    )

# -------------------------------------------------
# PROCESS AI2D DATA (FROM .pt FILES)
# -------------------------------------------------
logger.info("Processing AI2D dataset from preprocessed tensors...")
ai2d_records = []

pt_paths = list(PREPROCESS_DIR.glob("*.pt"))
logger.info(f"Found {len(pt_paths)} preprocessed AI2D tensors")

for i in progress(
    range(0, len(pt_paths), CFG["p2s_batch_size"]),
    desc="AI2D to structured"
):
    batch_paths = pt_paths[i:i + CFG["p2s_batch_size"]]

    images = []
    for p in batch_paths:
        img = load_preprocessed_pt(p)
        if img is not None:
            images.append(img)

    if not images:
        continue

    inputs = p2s_processor(images=images, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = p2s_model.generate(
            **inputs,
            max_length=CFG["max_input_len"]
        )

    for out in outputs:
        structured = p2s_processor.decode(out, skip_special_tokens=True)
        ai2d_records.append({
            "prompt": build_prompt(structured),
            "alt_text": "Diagram containing labeled elements related to a scientific concept."
        })

# -------------------------------------------------
# PROCESS CUSTOM DATA (PNG → PIL OK)
# -------------------------------------------------
logger.info("Processing custom annotated dataset...")
custom_df = pd.read_csv(CUSTOM_CSV)
custom_records = []

for _, row in progress(custom_df.iterrows(), desc="Custom to structured"):
    try:
        img = Image.open(row["Image-Path"]).convert("RGB")
    except:
        continue

    inputs = p2s_processor(images=[img], return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out = p2s_model.generate(**inputs, max_length=CFG["max_input_len"])

    structured = p2s_processor.decode(out[0], skip_special_tokens=True)

    custom_records.append({
        "prompt": build_prompt(structured),
        "alt_text": row["Modified-Alt-Text"]
    })

# -------------------------------------------------
# MERGE DATASETS
# -------------------------------------------------
dataset = Dataset.from_list(ai2d_records + custom_records)
logger.info(f"Total training samples: {len(dataset)}")

# -------------------------------------------------
# LOAD FLAN-T5
# -------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(CFG["model"])
model = AutoModelForSeq2SeqLM.from_pretrained(
    CFG["model"], torch_dtype=torch.float32
).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=CFG["lr"])

# -------------------------------------------------
# TRAINING LOOP
# -------------------------------------------------
model.train()
step = 0

for epoch in range(CFG["epochs"]):
    logger.info(f"Epoch {epoch+1}/{CFG['epochs']}")
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    for i in progress(range(0, len(indices), CFG["batch_size"]), desc="Training"):
        batch = [dataset[j] for j in indices[i:i+CFG["batch_size"]]]

        x = tokenizer(
            [b["prompt"] for b in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CFG["max_input_len"]
        ).to(DEVICE)

        y = tokenizer(
            [b["alt_text"] for b in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CFG["max_output_len"]
        ).input_ids.to(DEVICE)

        y[y == tokenizer.pad_token_id] = -100

        loss = model(**x, labels=y).loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        step += 1
        if step % 100 == 0:
            log_metrics({"step": step, "loss": loss.item()}, OUT_DIR)
            log_health(logger)
            cooldown()

# -------------------------------------------------
# SAVE
# -------------------------------------------------
model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
logger.info("MAJUNGA v3 TRAINING COMPLETE")