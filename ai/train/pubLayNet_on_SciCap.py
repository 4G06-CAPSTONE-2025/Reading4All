import os
import json
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, TrainingArguments, Trainer
from datetime import datetime

# -----------------------------
# Hardcoded dataset paths
# -----------------------------
BASE = r"C:/Users/nawaa/Downloads/scicap_data_extracted/scicap_data"
IMG_NO = os.path.join(BASE, "Scicap-No-Subfig-Img")
IMG_YES = os.path.join(BASE, "SciCap-Yes-Subfig-Img")
CAP_ALL = os.path.join(BASE, "SciCap-Caption-All")

# -----------------------------
# Paths for model, test images, and configs
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # ai/train
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Reading4All

CONFIG_PATH = os.path.join(SCRIPT_DIR, "paths_config.json")
with open(CONFIG_PATH, "r") as f:
    paths_config = json.load(f)

MODEL_ROOT = os.path.join(PROJECT_ROOT, paths_config["model_dir"])
os.makedirs(MODEL_ROOT, exist_ok=True)

TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, paths_config["test_images_dir"])

# -----------------------------
# Accessibility template
# -----------------------------
def apply_accessibility_template(raw_caption):
    raw_caption = raw_caption.strip()
    if raw_caption.lower().startswith(("figure", "fig.", "image")):
        return raw_caption
    return (
        "Figure description: "
        f"{raw_caption} "
        "Key visual elements, relationships, and trends are described for accessibility."
    )

# -----------------------------
# Custom Trainer
# -----------------------------
class BlipTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs.pop("num_items_in_batch", None)
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

# -----------------------------
# Load captions & image paths
# -----------------------------
def load_split(split, max_samples=None):
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
                caption = apply_accessibility_template(
                    item["2-normalized"]["2-1-basic-num"]["caption"]
                )

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
                            # Check if the image can be opened
                            try:
                                with Image.open(img_path) as img:
                                    img.verify()
                                records.append({"image_path": img_path, "caption": caption})
                                found = True
                            except (IOError, OSError, SyntaxError) as e:
                                corrupted_images.append(img_path)
                                print(f"Skipping corrupted image: {img_path} ({e})")
                            break
                    if found:
                        break

                if max_samples is not None and len(records) >= max_samples:
                    print(f"[{split}] stopping early at {max_samples} samples")
                    return records

                if len(records) % 1000 == 0 and len(records) > 0:
                    print(f"{len(records)} samples loaded so far for {split}")

    if not records:
        raise RuntimeError(f"No samples loaded for split '{split}'")

    print(f"Loaded {len(records)} samples for {split}")
    if corrupted_images:
        print(f"Skipped {len(corrupted_images)} corrupted images. Example: {corrupted_images[:5]}")
    return records

# -----------------------------
# Dataset class
# -----------------------------
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
        except (IOError, OSError, SyntaxError) as e:
            print(f"Warning: Failed to load image {rec['image_path']} ({e}). Using placeholder.")
            img = Image.new("RGB", (self.image_size, self.image_size), color=(0, 0, 0))
        return {"image": img, "caption": rec["caption"]}

# -----------------------------
# Processor & collate function
# -----------------------------
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

def collate_fn(batch):
    images = [x["image"] for x in batch]
    captions = [x["caption"] for x in batch]
    inputs = processor(images=images, text=captions,
                       padding="max_length", truncation=True,
                       max_length=128, return_tensors="pt")
    inputs["labels"] = inputs["input_ids"]
    return inputs

# -----------------------------
# Load PubLayNet vision encoder
# -----------------------------
def load_publaynet_vision_weights(model, ckpt_path):
    # fix Windows path for HF
    ckpt_path = ckpt_path.replace("\\", "/")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"PubLayNet checkpoint not found at {ckpt_path}")

    publaynet_model = BlipForConditionalGeneration.from_pretrained(
        ckpt_path,
        local_files_only=True
    )
    model.vision_model.load_state_dict(
        publaynet_model.vision_model.state_dict(),
        strict=True
    )
    print("Loaded PubLayNet vision encoder weights")


# -----------------------------
# Main training
# -----------------------------
if __name__ == "__main__":

    # -----------------------------
    # Load SciCap dataset with optional max_samples for quick testing
    # -----------------------------
    train_data = load_split("train", max_samples=1000)
    val_data   = load_split("val",   max_samples=200)

    print(f"Train samples: {len(train_data)}")
    print(f"Val samples:   {len(val_data)}")

    train_ds = OnTheFlyDataset(train_data)
    val_ds = OnTheFlyDataset(val_data)

    # -----------------------------
    # Load BLIP model
    # -----------------------------
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # -----------------------------
    # Freeze all parameters first
    # -----------------------------
    for param in model.parameters():
        param.requires_grad = False

    # -----------------------------
    # Load PubLayNet vision weights
    # -----------------------------
    PUBLAYNET_MODEL_PATH = r"C:/Users/nawaa/OneDrive/Desktop/Reading4All/ai/model/blip_publaynet_vision_20260117_1258"
    load_publaynet_vision_weights(model, PUBLAYNET_MODEL_PATH)

    # -----------------------------
    # Freeze vision encoder
    # -----------------------------
    for param in model.vision_model.parameters():
        param.requires_grad = False

    # -----------------------------
    # Unfreeze entire text decoder + LM head (safe approach)
    # -----------------------------
    for name, param in model.text_decoder.named_parameters():
        param.requires_grad = True

    # -----------------------------
    # Device
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # -----------------------------
    # Log trainable parameters
    # -----------------------------
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,}/{total:,} ({100 * trainable / total:.2f}%)")

    # -----------------------------
    # Dynamic output directory
    # -----------------------------
    MODEL_NAME = f"scicap_blip_model_final_{datetime.now().strftime('%Y_%m_%d_%H%M')}"
    MODEL_DIR = os.path.join(MODEL_ROOT, MODEL_NAME)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # -----------------------------
    # Load training configuration
    # -----------------------------
    training_config_path = os.path.join(SCRIPT_DIR, "training_config.json")
    with open(training_config_path, "r") as f:
        config = json.load(f)
    config["output_dir"] = MODEL_DIR

    training_args = TrainingArguments(**config)

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = BlipTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
    )

    print("\n=== STARTING SciCap CAPTION TRAINING ===\n")
    trainer.train()
    trainer.save_model(MODEL_DIR)
    print(f"\n Model saved to: {MODEL_DIR}\n")

    print("=== SANITY-CHECK INFERENCE ===\n")
    model.eval()
    sample_batch = [val_ds[i] for i in range(min(5, len(val_ds)))]  # pick first 5 samples
    images = [x["image"] for x in sample_batch]
    captions_gt = [x["caption"] for x in sample_batch]

    inputs = processor(images=images, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128, num_beams=3)
        preds = processor.batch_decode(outputs, skip_special_tokens=True)

    for i, (pred, gt) in enumerate(zip(preds, captions_gt)):
        print(f"Sample {i+1}")
        print(f"  Ground truth: {gt}")
        print(f"  Prediction  : {pred}\n")

