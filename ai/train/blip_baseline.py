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

# Model folder
MODEL_DIR = os.path.join(PROJECT_ROOT, paths_config["model_dir"])
os.makedirs(MODEL_DIR, exist_ok=True)

# Test images folder
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, paths_config["test_images_dir"])

# print("Project root:", PROJECT_ROOT)
# print("Model directory:", MODEL_DIR)
# print("Test images directory:", TEST_IMAGES_DIR)

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
                            # Check if the image can be opened
                            try:
                                with Image.open(img_path) as img:
                                    img.verify()  # verify the image is not corrupted
                                records.append({"image_path": img_path, "caption": caption})
                                found = True
                            except (IOError, OSError, SyntaxError) as e:
                                corrupted_images.append(img_path)
                                print(f"Skipping corrupted image: {img_path} ({e})")
                            break
                    if found:
                        break

                if len(records) % 1000 == 0:
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
            print(f"Warning: Failed to load image {rec['image_path']} ({e}). Skipping.")
            # Return a blank image to keep batch size consistent
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
# Main training
# -----------------------------
if __name__ == "__main__":
    print("ABOUT TO LOAD TRAIN SPLIT")
    train_data = load_split("train")
    print("TRAIN SPLIT LOADED")

    print("ABOUT TO LOAD VAL SPLIT")
    val_data = load_split("val")
    print("VAL SPLIT LOADED")

    train_ds = OnTheFlyDataset(train_data)
    val_ds = OnTheFlyDataset(val_data)

    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    for param in model.vision_model.parameters():
        param.requires_grad = False
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Load training configuration
    training_config_path = os.path.join(SCRIPT_DIR, "training_config.json")
    with open(training_config_path, "r") as f:
        config = json.load(f)
    # -----------------------------
    # Dynamic output_dir for saving this training run
    # -----------------------------
    MODEL_NAME = f"scicap_blip_model_final_{datetime.now().strftime('%Y_%m_%d_%H%M')}"
    MODEL_DIR = os.path.join(PROJECT_ROOT, "ai", "model", MODEL_NAME)
    os.makedirs(MODEL_DIR, exist_ok=True)
    config["output_dir"] = MODEL_DIR

    print("Trained model will be saved in:", MODEL_DIR)

    # -----------------------------
    # Training
    # -----------------------------
    training_args = TrainingArguments(**config)

    trainer = BlipTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
    )

    trainer.train()
