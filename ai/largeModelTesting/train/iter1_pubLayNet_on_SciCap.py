"""
============================================================
SciCap BLIP Fine-Tuning Script
============================================================

Description:
------------
This script fine-tunes the Salesforce BLIP image captioning 
model on the SciCap dataset, focusing on cross-attention 
layers of the text decoder. The pipeline includes:

1. Loading SciCap image-caption data (train/val splits).
2. Lazy on-the-fly dataset creation with image verification.
3. Custom Trainer class (BlipTrainer) to handle loss 
   computation and batch nuances.
4. BLIP model loading and selective layer freezing/unfreezing:
   - Freeze all parameters initially.
   - Unfreeze cross-attention layers in the text decoder.
   - Unfreeze the text decoder LM head for caption generation.
5. Fine-tuning using Hugging Face Trainer and configurable 
   training arguments.
6. Saving the trained model and processor for downstream use.
7. Testing on a small subset of validation images for qualitative 
   evaluation.

File Structure & Paths:
----------------------
- BASE: Root SciCap dataset folder (extracted images and captions)
- IMG_NO: Images without subfigures
- IMG_YES: Images with subfigures
- CAP_ALL: Caption JSON files
- MODEL_BASE_DIR: Directory where checkpoints and fine-tuned models are saved
- CONFIG_PATH: JSON file specifying path configurations
- training_config.json: JSON file containing TrainingArguments for Trainer

Key Classes & Functions:
------------------------
1. BlipTrainer(Trainer):
   - Custom Trainer overriding compute_loss to safely handle 
     model inputs.
   
2. load_split(split: str) -> list:
   - Loads a train/val split, validates images, and returns 
     a list of records: {"image_path": str, "caption": str}.
   - Skips corrupted images and logs progress every 1000 samples.

3. OnTheFlyDataset(torch.utils.data.Dataset):
   - Lazy dataset that loads images on-the-fly to save memory.
   - __getitem__ returns a dictionary with keys "image" and "caption".

4. collate_fn(batch: list) -> dict:
   - Collates a batch of images and captions using BlipProcessor.
   - Prepares input tensors and labels for training.

Usage:
------
Run the script directly for training:

    python iter1_pubLayNet_on_SciCap.py

Requirements:
-------------
- Python >= 3.8
- torch, transformers, safetensors, datasets, Pillow
- SciCap dataset extracted to BASE folder
- training_config.json present in the script directory

Output:
-------
- Fine-tuned BLIP model saved in MODEL_BASE_DIR/<timestamp>
- Processor saved alongside the model
- Optional small-scale test outputs printed to console
"""



import os
import json
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, TrainingArguments, Trainer
from datetime import datetime
from safetensors.torch import load_file

# ============================================================
# Paths
# ============================================================
BASE = r"C:/Users/nawaa/Downloads/scicap_data_extracted/scicap_data"
IMG_NO = os.path.join(BASE, "Scicap-No-Subfig-Img")
IMG_YES = os.path.join(BASE, "SciCap-Yes-Subfig-Img")
CAP_ALL = os.path.join(BASE, "SciCap-Caption-All")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # ai/train
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Reading4All/ai

CONFIG_PATH = os.path.join(SCRIPT_DIR, "paths_config.json")
with open(CONFIG_PATH, "r") as f:
    paths_config = json.load(f)

MODEL_BASE_DIR = os.path.join(PROJECT_ROOT, paths_config.get("model_dir", "ai/model"))
os.makedirs(MODEL_BASE_DIR, exist_ok=True)

# ============================================================
# Custom Trainer
# ============================================================
class BlipTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs.pop("num_items_in_batch", None)
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

# ============================================================
# Load SciCap captions & image paths
# ============================================================
def load_split(split):
    split_folder = next(
        (os.path.join(CAP_ALL, candidate) for candidate in [split.lower(), split.capitalize(), split.upper()]
         if os.path.exists(os.path.join(CAP_ALL, candidate))),
        None
    )
    if split_folder is None:
        raise FileNotFoundError(f"No caption folder found for split '{split}'")

    records = []
    corrupted_images = []

    for jf in [f for f in os.listdir(split_folder) if f.endswith(".json")]:
        with open(os.path.join(split_folder, jf), "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]

            for item in data:
                img_id = os.path.splitext(item["figure-ID"])[0].lower()
                caption = item["2-normalized"]["2-1-basic-num"]["caption"]

                img_folders = [IMG_NO]
                if split.lower() == "train":
                    img_folders.append(IMG_YES)

                found = False
                for folder in img_folders:
                    folder_split = os.path.join(folder, split.lower())
                    if not os.path.exists(folder_split):
                        continue
                    for ext in [".png", ".jpg", ".jpeg"]:
                        img_path = os.path.join(folder_split, img_id + ext)
                        if os.path.exists(img_path):
                            try:
                                with Image.open(img_path) as img:
                                    img.verify()
                                records.append({"image_path": img_path, "caption": caption})
                                found = True
                            except (IOError, OSError, SyntaxError):
                                corrupted_images.append(img_path)
                            break
                    if found:
                        break

                if len(records) % 1000 == 0:
                    print(f"{len(records)} samples loaded so far for {split}")

    print(f"Loaded {len(records)} samples for {split}")
    if corrupted_images:
        print(f"Skipped {len(corrupted_images)} corrupted images")
    return records

# ============================================================
# Dataset class
# ============================================================
class OnTheFlyDataset(torch.utils.data.Dataset):
    def __init__(self, records, image_size=384):
        self.records = records
        self.image_size = image_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            img = Image.open(rec["image_path"]).convert("RGB")
        except Exception as e:
            print(f"Warning: Failed to load {rec['image_path']} ({e})")
            img = Image.new("RGB", (self.image_size, self.image_size), color=(0, 0, 0))
        return {"image": img, "caption": rec["caption"]}

# ============================================================
# Processor & collate function
# ============================================================
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

def collate_fn(batch):
    images = [x["image"] for x in batch]
    captions = [x["caption"] for x in batch]
    inputs = processor(images=images, text=captions,
                       padding="max_length", truncation=True,
                       max_length=128, return_tensors="pt")
    inputs["labels"] = inputs["input_ids"]
    return inputs

# ============================================================
# Main training + testing
# ============================================================
if __name__ == "__main__":
    print("Loading train/val splits...")
    train_data = load_split("train")
    val_data = load_split("val")

    train_ds = OnTheFlyDataset(train_data)
    val_ds = OnTheFlyDataset(val_data)

    # Load base BLIP model first
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Auto-detect latest vision checkpoint
    vision_folders = [
        f for f in os.listdir(MODEL_BASE_DIR)
        if os.path.isdir(os.path.join(MODEL_BASE_DIR, f)) and f.startswith("blip_publaynet_vision_")
    ]

    if not vision_folders:
        print("WARNING: No vision checkpoints found, training from base model.")
    else:
        latest_folder = sorted(vision_folders)[-1]
        vision_weights_path = os.path.join(MODEL_BASE_DIR, latest_folder, "model.safetensors")
        if os.path.exists(vision_weights_path):
            print(f"Using vision checkpoint: {vision_weights_path}")
            state_dict = load_file(vision_weights_path, device=device)
            model.load_state_dict(state_dict, strict=False)
        else:
            print(f"WARNING: Expected checkpoint not found: {vision_weights_path}. Using base model.")

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze cross-attention layers in the text decoder
    for name, module in model.text_decoder.named_modules():
        if "crossattention" in name.lower() or "encoder_attn" in name.lower():
            for param in module.parameters():
                param.requires_grad = True

    # Unfreeze LM head
    for param in model.text_decoder.cls.parameters():
        param.requires_grad = True


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Dynamic output directory
    MODEL_NAME = f"scicap_blip_finetune_crossattn_{datetime.now().strftime('%Y%m%d_%H%M')}"
    OUTPUT_DIR = os.path.join(MODEL_BASE_DIR, MODEL_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load training args
    training_config_path = os.path.join(SCRIPT_DIR, "training_config.json")
    with open(training_config_path, "r") as f:
        config = json.load(f)
    config["output_dir"] = OUTPUT_DIR

    training_args = TrainingArguments(**config)

    trainer = BlipTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
    )

    # Train
    print("=== STARTING SciCap Fine-Tuning (Cross-Attention Only) ===")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

    # Testing on a small subset
    print("=== TESTING ON SMALL DATASET ===")
    test_data = val_data[:5]
    model.eval()

    for rec in test_data:
        img = Image.open(rec["image_path"]).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_length=128)
        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        print(f"Image: {os.path.basename(rec['image_path'])}")
        print(f"Reference Caption: {rec['caption']}")
        print(f"Generated Caption: {caption}\n")