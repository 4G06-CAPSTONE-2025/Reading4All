import os
import json
import yaml
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, BlipProcessor, BlipForConditionalGeneration

# ---------------------------
# Load config.yaml
# ---------------------------
config_path = os.path.abspath("./config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Get model name, dataset, output folder
model_name = config.get("model_name", "nlpconnect/vit-gpt2-image-captioning")
dataset_dir = os.path.abspath(os.path.join("..", config.get("dataset_dir", "images")))
output_dir = os.path.abspath(os.path.join("..", config.get("output_dir", "outputs")))
os.makedirs(output_dir, exist_ok=True)
output_json = os.path.join(output_dir, "captions_filled.json")

# ---------------------------
# Load model & processor
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Support both BLIP and ViT-GPT2 (for PoC purposes)
if "blip" in model_name.lower():
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
else:
    processor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

model.to(device)
model.eval()

# ---------------------------
# Loop through all images in dataset_dir
# ---------------------------
captions_data = []

for img_file in os.listdir(dataset_dir):
    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(dataset_dir, img_file)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Failed to open {img_file}: {e}")
            continue

        # Generate caption
        if "blip" in model_name.lower():
            inputs = processor(images=image, return_tensors="pt").to(device)
            output_ids = model.generate(**inputs, max_length=50)
            caption = processor.decode(output_ids[0], skip_special_tokens=True)
        else:
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            output_ids = model.generate(pixel_values, max_length=50)
            caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        captions_data.append({"image": img_file, "caption": caption})
        print(f"{img_file} : {caption}")

# ---------------------------
# Save captions to JSON
# ---------------------------
with open(output_json, "w") as f:
    json.dump(captions_data, f, indent=4)

print(f"\nAll captions saved to {output_json}")