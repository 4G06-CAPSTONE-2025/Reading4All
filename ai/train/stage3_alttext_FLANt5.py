import sys, os, torch, random, json
from pathlib import Path
from PIL import Image, ImageOps
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
CUSTOM_CSV = r"C:/Users/nawaa/Downloads/annotated_physics_data(Sheet1).csv"

STAGE2_MODEL = os.path.join(MODEL_DIR, "stage2_semantic")
OUT_DIR = os.path.join(MODEL_DIR, "stage3_majunga")

CFG = {
    "model": "google/flan-t5-base",
    "epochs": 3,
    "batch_size": 1,
    "lr": 3e-5,
    "seed": 42,
    "max_input_len": 512,
    "max_output_len": 256
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
# IMAGE LOADER
# -------------------------------------------------
def safe_image(path, size=(224,224)):
    try:
        with Image.open(path) as img:
            img.verify()
        img = Image.open(path).convert("RGB")
        img.thumbnail(size, Image.LANCZOS)
        pad_w, pad_h = size[0]-img.width, size[1]-img.height
        return ImageOps.expand(
            img,
            (pad_w//2, pad_h//2, pad_w-pad_w//2, pad_h-pad_h//2),
            fill=(255,255,255)
        )
    except:
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
# LOAD AI2D DATA (ON-THE-FLY STRUCTURING)
# -------------------------------------------------
logger.info("Processing AI2D dataset...")
ai2d_records = []

img_dir = Path(AI2D_ROOT) / "images"
ann_dir = Path(AI2D_ROOT) / "annotations"
q_dir = Path(AI2D_ROOT) / "questions"

for img_path in progress(list(img_dir.glob("*.png")), desc="AI2D → structured"):
    img = safe_image(img_path)
    if img is None:
        continue

    inputs = p2s_processor(images=[img], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = p2s_model.generate(**inputs, max_length=CFG["max_input_len"])

    structured = p2s_processor.decode(out[0], skip_special_tokens=True)

    # Weak supervision target (AI2D context)
    target_text = f"Diagram containing labeled elements related to a scientific concept."

    ai2d_records.append({
        "prompt": build_prompt(structured),
        "alt_text": target_text
    })

# -------------------------------------------------
# LOAD CUSTOM DATA (STRONG SUPERVISION)
# -------------------------------------------------
logger.info("Processing custom annotated dataset...")
custom_df = pd.read_csv(CUSTOM_CSV)
custom_records = []

for _, row in custom_df.iterrows():
    img = safe_image(row["Image-Path"])
    if img is None:
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