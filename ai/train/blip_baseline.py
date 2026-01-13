import os
import json
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, TrainingArguments, Trainer

print("TRAINING SCRIPT STARTED")

# -----------------------------
# Paths
# -----------------------------
BASE = r"C:/Users/nawaa/Downloads/scicap_data_extracted/scicap_data"
IMG_NO = os.path.join(BASE, "Scicap-No-Subfig-Img")
IMG_YES = os.path.join(BASE, "SciCap-Yes-Subfig-Img")
CAP_ALL = os.path.join(BASE, "SciCap-Caption-All")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "scicap_blip_model_final")
os.makedirs(MODEL_DIR, exist_ok=True)

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
    for jf in [f for f in os.listdir(split_folder) if f.endswith(".json")]:
        with open(os.path.join(split_folder, jf), "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]

            for item in data:
                img_id = os.path.splitext(item["figure-ID"])[0].lower()
                caption = item["2-normalized"]["2-1-basic-num"]["caption"]

                # Decide which image folders to search based on split
                img_folders = [IMG_NO]  # Always include No-Subfig
                if split.lower() == "train":
                    img_folders.append(IMG_YES)  # Only check Yes-Subfig for train

                found = False
                for folder in img_folders:
                    folder_split = os.path.join(folder, split.lower())
                    if not os.path.exists(folder_split):
                        continue
                    for ext in [".png", ".jpg", ".jpeg"]:
                        img_path = os.path.join(folder_split, img_id + ext)
                        if os.path.exists(img_path):
                            records.append({"image_path": img_path, "caption": caption})
                            found = True
                            break
                    if found:
                        break
                if len(records) % 1000 == 0:
                    print(f"{len(records)} samples loaded so far for {split}")

    if not records:
        raise RuntimeError(f"No samples loaded for split '{split}'")
    print(f"Loaded {len(records)} samples for {split}")
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
        img = Image.open(rec["image_path"]).convert("RGB")
        return {"image": img, "caption": rec["caption"]}

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# -----------------------------
# Collate function
# -----------------------------
def collate_fn(batch):
    images = [x["image"] for x in batch]
    captions = [x["caption"] for x in batch]
    inputs = processor(images=images, text=captions,
                       padding="max_length", truncation=True,
                       max_length=128, return_tensors="pt")
    inputs["labels"] = inputs["input_ids"]
    return inputs

# -----------------------------
# Main
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

    with open(os.path.join(os.path.dirname(__file__), "training_config.json"), "r") as f:
        config = json.load(f)

    training_args = TrainingArguments(**config)

    trainer = BlipTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
    )

    trainer.train()
