import os
import json
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # ai/test
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))     # Reading4All

INFERENCE_DIR = os.path.join(PROJECT_ROOT, "ai", "inference", "images")
TEST_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "ai", "test")

# HARD-CODED MODEL PATH (relative to NF project root)
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "ai",
    "model",
    "scicap_blip_model_final",
    "scicap_blip_finetune_crossattn_20260118_0644",
    "checkpoint-12021"
)

os.makedirs(TEST_OUTPUT_ROOT, exist_ok=True)

# ============================================================
# Safety check (fail fast)
# ============================================================
if not os.path.isdir(MODEL_PATH):
    raise RuntimeError(f"Model path does not exist:\n{MODEL_PATH}")

if not (
    os.path.exists(os.path.join(MODEL_PATH, "model.safetensors")) or
    os.path.exists(os.path.join(MODEL_PATH, "pytorch_model.bin"))
):
    raise RuntimeError(f"No model weights found in:\n{MODEL_PATH}")

print("Using model at:")
print(MODEL_PATH)

# ============================================================
# Device
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ============================================================
# Load processor + model
# ============================================================
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    MODEL_PATH
).to(DEVICE)

model.eval()

# ============================================================
# Inference function
# ============================================================
def generate_caption(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Skipping corrupted image {image_path} ({e})")
        return ""

    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=150,
            num_beams=3
        )

    return processor.decode(
        output_ids[0],
        skip_special_tokens=True
    )

# ============================================================
# Run inference
# ============================================================
outputs = {}

for fname in sorted(os.listdir(INFERENCE_DIR)):
    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(INFERENCE_DIR, fname)
        caption = generate_caption(img_path)
        outputs[fname] = caption
        print(f"{fname} -> {caption}")

# ============================================================
# Save outputs
# ============================================================
MODEL_RUN_NAME = "scicap_blip_finetune_crossattn_20260118_0644"

output_dir = os.path.join(TEST_OUTPUT_ROOT, MODEL_RUN_NAME)
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "captions.json")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(outputs, f, indent=4, ensure_ascii=False)

print(f"\nSaved captions for {len(outputs)} images to:")
print(output_file)