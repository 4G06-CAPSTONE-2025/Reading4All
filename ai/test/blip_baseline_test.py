import os
import json
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# -----------------------------
# Paths
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # ai/test
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Reading4All

# Folder containing images to test
INFERENCE_DIR = os.path.join(PROJECT_ROOT, "ai", "inference", "images")

# Folder where trained models are saved
MODEL_ROOT = os.path.join(PROJECT_ROOT, "ai", "model")

# Folder to save test outputs
TEST_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "ai", "test")
os.makedirs(TEST_OUTPUT_ROOT, exist_ok=True)

# -----------------------------
# Choose the model to test
# -----------------------------
#MODEL_NAME = "scicap_blip_model_final_2026_01_12_2100"
MODEL_NAME = sorted(os.listdir(MODEL_ROOT))[-1] #pick the last model created dynamically
MODEL_BASE_PATH = os.path.join(MODEL_ROOT, MODEL_NAME)

# Find latest checkpoint
checkpoints = [d for d in os.listdir(MODEL_BASE_PATH) if d.startswith("checkpoint-")]

if not checkpoints:
    raise RuntimeError("No checkpoints found in model directory.")

LATEST_CHECKPOINT = sorted(
    checkpoints,
    key=lambda x: int(x.split("-")[-1])
)[-1]

MODEL_PATH = os.path.join(MODEL_BASE_PATH, LATEST_CHECKPOINT)

print("Loading model from:", MODEL_PATH)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load processor and model
# -----------------------------
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")  # processor can remain base
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
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# -----------------------------
# Run inference on all images
# -----------------------------
outputs = {}
for fname in os.listdir(INFERENCE_DIR):
    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(INFERENCE_DIR, fname)
        caption = generate_caption(img_path)
        outputs[fname] = caption
        print(f"{fname} -> {caption}")

# -----------------------------
# Save outputs to JSON
# -----------------------------
output_dir = os.path.join(TEST_OUTPUT_ROOT, MODEL_NAME)
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "captions.json")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(outputs, f, indent=4, ensure_ascii=False)

print(f"Saved captions for {len(outputs)} images to {output_file}")
