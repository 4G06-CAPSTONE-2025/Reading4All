"""
Stage 2 – AI2D Semantic Parser Retraining (Pix2Struct)
====================================================

This script retrains **Stage 2** of a multi-stage diagram understanding pipeline
using the **AI2D dataset**. The goal of this stage is to teach a Pix2Struct model
to convert raw STEM diagram images into **structured semantic descriptions**
(objects + visual emphasis), which will later be consumed by a language model
for alt-text generation.

This stage focuses on **generic diagram parsing**, not final alt text.

--------------------------------------------------------------------
Pipeline Context
--------------------------------------------------------------------
Overall system architecture:

    Diagram Image
        to
    Stage 1 (SciCap-trained Pix2Struct)
        → Basic visual-text alignment
        to
    Stage 2 (THIS SCRIPT – AI2D Semantic Parser)
        → Structured semantic representation
        to
    Stage 3 (FLAN-T5 / MAJUNGA)
        → Accessible alt text

--------------------------------------------------------------------
Hardware & Platform Assumptions
--------------------------------------------------------------------
• Operating System: Windows 10 / 11
• GPU: NVIDIA RTX / GTX (consumer GPU, CUDA enabled)
• Mixed precision training (torch.amp + GradScaler)
• Small batch size with gradient accumulation for stability
• Designed to be safe on Windows systems with limited VRAM

--------------------------------------------------------------------
Model & Training Strategy
--------------------------------------------------------------------
• Base model: Pix2Struct (loaded from Stage 1 SciCap checkpoint)
• Encoder: Frozen (vision backbone not updated)
• Decoder: Trainable (learns structured semantic language)
• Optimizer: AdamW
• Learning rate: 5e-5
• Epochs: 15
• Effective batch size: BATCH_SIZE × GRAD_ACCUM

--------------------------------------------------------------------
Dataset Processing (AI2D)
--------------------------------------------------------------------
Inputs:
• Diagram images (PNG / JPG)
• AI2D JSON annotations (when available)

For each diagram:
1. Image is resized and normalized
2. Visual emphasis is heuristically detected:
   - Edge density
   - Contrast level
   - Salient region (default: center)
3. Annotation objects are extracted (if present)
4. A structured semantic target is constructed:

   OBJECTS:
   - ARROW: indicates direction
   - TEXT: label name
   VISUAL EMPHASIS: edge_density=high, high_contrast=present, ...

If annotations are missing, the model falls back to a generic description.

--------------------------------------------------------------------
Training Objective
--------------------------------------------------------------------
• Task: Image → Structured text generation
• Loss: Cross-entropy over tokenized semantic description
• Padding tokens masked with -100
• Mixed precision forward pass for efficiency

--------------------------------------------------------------------
Data Flow
--------------------------------------------------------------------
Raw image → Processor → Pix2Struct encoder (frozen)
          → Decoder (trainable)
          → Structured semantic text

--------------------------------------------------------------------
Checkpoints & Outputs
--------------------------------------------------------------------
• Model checkpoints saved after each epoch
• Saved artifacts include:
  - Pix2Struct model weights
  - Processor configuration
• Output directory:
      ai/model/stage2_ai2d_generic/

--------------------------------------------------------------------
Intended Use
--------------------------------------------------------------------
This script is intended for:
• Diagram understanding research
• Accessibility tooling pipelines
• Structured vision-language representations
• Pretraining before downstream alt-text generation

This stage should be run **after Stage 1** and **before Stage 3**.

--------------------------------------------------------------------
"""

import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration, AutoTokenizer
from PIL import Image
import cv2

# -------------------------------
# CONFIGURATION
# -------------------------------
STAGE1_MODEL = r"C:/Users/nawaa/OneDrive/Desktop/Reading4All/ai/model/pix2struct_scicap_stage1_20260122_1946"
IMAGE_DIR = r"C:/Users/nawaa/Downloads/ai2d-all/ai2d/images"
ANN_DIR = r"C:/Users/nawaa/Downloads/ai2d-all/ai2d/annotations"
PREPROCESS_DIR = r"C:/Users/nawaa/Downloads/ai2d-all/ai2d/preprocessed"
OUTPUT_DIR = r"C:/Users/nawaa/OneDrive/Desktop/Reading4All/ai/model/stage2_ai2d_generic"

BATCH_SIZE = 1
GRAD_ACCUM = 8
EPOCHS = 15
LR = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(PREPROCESS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD PROCESSOR + TOKENIZER
# ============================================================

processor = Pix2StructProcessor.from_pretrained(STAGE1_MODEL)
tokenizer = AutoTokenizer.from_pretrained(STAGE1_MODEL)
model = Pix2StructForConditionalGeneration.from_pretrained(STAGE1_MODEL)
model.to(DEVICE)

# Freeze encoder
for name, param in model.named_parameters():
    if name.startswith("encoder"):
        param.requires_grad = False

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scaler = GradScaler()

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_json_load(filepath):
    import json
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().strip()
            if content.startswith('\ufeff'):
                content = content[1:]
            return json.loads(content)
    except:
        return {}

def detect_multitype_emphasis(img: Image.Image):
    # minimal example: just detect edges and contrast

    img_np = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray,100,200)
    high_contrast = gray.std()/255.0
    region = "center"
    return {
        "edge_density": "high" if edges.mean()/255>0.05 else "low",
        "high_contrast": "present" if high_contrast>0.05 else "none",
        "region": region,
        "present": "yes"
    }

def build_semantic_target(annotation, emphasis):
    # If annotation missing, fallback to visual description
    objects = []
    if annotation:
        for obj in annotation.get("diagramElements", {}).values():
            text = obj.get("text","")
            type_ = obj.get("type","element")
            objects.append(f"{type_.upper()}: {text}" if text else type_.upper())
    if not objects:
        objects.append("Diagram with unknown objects")

    emphasis_text = ", ".join([f"{k}={v}" for k,v in emphasis.items()])
    return "OBJECTS:\n- " + "\n- ".join(objects) + f"\nVISUAL EMPHASIS: {emphasis_text}"

# ============================================================
# DATASET
# ============================================================

class AI2DDataset(Dataset):
    def __init__(self, image_dir, ann_dir):
        self.image_dir = Path(image_dir)
        self.ann_dir = Path(ann_dir)
        self.files = sorted([p for p in self.image_dir.glob("*") if p.suffix.lower() in [".png",".jpg"]])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        ann_path = self.ann_dir / f"{img_path.stem}.json"
        annotation = safe_json_load(ann_path)
        with Image.open(img_path) as img:
            orig_img = img.copy()
            img = img.resize((224,224)).convert("RGB")
            img_np = np.array(img).astype(np.float32)/255.0
        emphasis = detect_multitype_emphasis(orig_img)
        target_text = build_semantic_target(annotation, emphasis)
        labels = tokenizer(target_text, padding="max_length", truncation=True, max_length=512, return_tensors="pt").input_ids.squeeze(0)
        labels[labels==tokenizer.pad_token_id] = -100
        return {
            "pixel_values": torch.tensor(img_np).permute(2,0,1),  # HWC -> CHW
            "labels": labels,
            "target_text": target_text
        }

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "target_text": [b["target_text"] for b in batch]
    }

# ============================================================
# TRAINING LOOP
# ============================================================

dataset = AI2DDataset(IMAGE_DIR, ANN_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)

step = 0
for epoch in range(EPOCHS):
    print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
    pbar = tqdm(loader)
    optimizer.zero_grad()
    for batch in pbar:
        if batch is None:
            continue

        # batch["pixel_values"] is (B,3,H,W) in CHW format
        images = batch["pixel_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        # Processor expects HWC images, so convert
        images_for_processor = images.permute(0,2,3,1)  # CHW -> HWC

        # Encode images using processor
        encoding = processor(images=list(images_for_processor.cpu().numpy()), return_tensors="pt").to(DEVICE)

        # Forward + loss
        with autocast(device_type="cuda" if DEVICE=="cuda" else "cpu"):
            outputs = model(**encoding, labels=labels)  # <-- pass processed tensors
            loss = outputs.loss / GRAD_ACCUM

        # Backward
        scaler.scale(loss).backward()
        step += 1

        if step % GRAD_ACCUM == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        pbar.set_postfix(real_loss=(loss.item()*GRAD_ACCUM))

    # Save epoch checkpoint
    save_dir = Path(OUTPUT_DIR)/f"epoch_{epoch+1}"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    print(f"Saved checkpoint at: {save_dir}")

print("Stage 2 retraining (general AI2D parser) complete!")