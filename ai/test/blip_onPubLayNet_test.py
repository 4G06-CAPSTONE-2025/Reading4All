import os
import time
import json
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# -----------------------------
# Paths
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # ai/test
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Reading4All

INFERENCE_DIR = os.path.join(PROJECT_ROOT, "ai", "inference", "images")
MODEL_ROOT = os.path.join(PROJECT_ROOT, "ai", "model")
TEST_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "ai", "test")
os.makedirs(TEST_OUTPUT_ROOT, exist_ok=True)

# -----------------------------
# Device
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# -----------------------------
# Find latest checkpoint across ALL models
# -----------------------------
latest_checkpoint_path = None
latest_checkpoint_mtime = -1
latest_model_name = None

for model_name in os.listdir(MODEL_ROOT):
    model_path = os.path.join(MODEL_ROOT, model_name)
    if not os.path.isdir(model_path):
        continue

    for ckpt in os.listdir(model_path):
        if not ckpt.startswith("checkpoint-"):
            continue

        ckpt_path = os.path.join(model_path, ckpt)

        if not (
            os.path.exists(os.path.join(ckpt_path, "pytorch_model.bin")) or
            os.path.exists(os.path.join(ckpt_path, "model.safetensors"))
        ):
            continue

        mtime = os.path.getmtime(ckpt_path)
        if mtime > latest_checkpoint_mtime:
            latest_checkpoint_mtime = mtime
            latest_checkpoint_path = ckpt_path
            latest_model_name = model_name

if latest_checkpoint_path is None:
    raise RuntimeError("No valid checkpoints found across any model directories.")

MODEL_NAME = latest_model_name
MODEL_PATH = latest_checkpoint_path

print("Selected latest model:", MODEL_NAME)
print("Selected checkpoint:", MODEL_PATH)
print("Last modified:", time.ctime(latest_checkpoint_mtime))

# -----------------------------
# Load processor and model
# -----------------------------
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

# -----------------------------
# Inference function
# -----------------------------
def generate_caption(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Skipping corrupted image {image_path} ({e})")
        return ""

    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs)

    return processor.decode(out[0], skip_special_tokens=True)

# -----------------------------
# Run inference
# -----------------------------
outputs = {}
for fname in sorted(os.listdir(INFERENCE_DIR)):
    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(INFERENCE_DIR, fname)
        caption = generate_caption(img_path)
        outputs[fname] = caption
        print(f"{fname} -> {caption}")

# -----------------------------
# Save outputs
# -----------------------------
output_dir = os.path.join(TEST_OUTPUT_ROOT, MODEL_NAME)
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "captions.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(outputs, f, indent=4, ensure_ascii=False)

print(f"Saved captions for {len(outputs)} images to {output_file}")