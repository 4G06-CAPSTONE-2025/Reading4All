import os
import json
import torch
from PIL import Image
from torchvision import transforms
from transformers import BlipProcessor, BlipForConditionalGeneration, TrainingArguments, Trainer

# -----------------------------
# Dataset paths
# -----------------------------
BASE = r"C:/Users/nawaa/Downloads/scicap_data_extracted/scicap_data"
IMG_NO = os.path.join(BASE, "Scicap-No-Subfig-Img")
IMG_YES = os.path.join(BASE, "SciCap-Yes-Subfig-Img")
CAP_ALL = os.path.join(BASE, "SciCap-Caption-All")

# -----------------------------
# Training parameters
# -----------------------------
MAX_CAPTION_LEN = 150
MODEL_FOLDER_PUBLAYNET = r"C:/Users/nawaa/OneDrive/Desktop/Reading4All/ai/model/blip_publaynet_vision_20260117_1258"
MODEL_FOLDER_OUT = r"C:/Users/nawaa/OneDrive/Desktop/Reading4All/ai/model/blip_sciCap_finetuned"
os.makedirs(MODEL_FOLDER_OUT, exist_ok=True)

MAX_SAMPLES_TRAIN = 5000  # Increase dataset size for better learning
MAX_SAMPLES_VAL = 500

IMAGE_SIZE = 384
BATCH_SIZE = 8  # Reduce batch size if GPU memory is limited

# -----------------------------
# Dataset class with augmentation
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
])

class OnTheFlyDataset(torch.utils.data.Dataset):
    def __init__(self, records, image_size=IMAGE_SIZE, augment=False):
        self.records = records
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            img = Image.open(rec["image_path"]).convert("RGB")
            if self.augment:
                img = transform(img)
        except (OSError, IOError, SyntaxError):
            img = Image.new("RGB", (self.image_size, self.image_size), color=(0, 0, 0))
        return {"image": img, "caption": rec["caption"]}

# -----------------------------
# Load split
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
                caption = item["2-normalized"]["2-1-basic-num"]["caption"].strip()
                if len(caption) == 0:
                    continue

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
                            except Exception:
                                corrupted_images.append(img_path)
                            break
                    if found:
                        break

                if max_samples and len(records) >= max_samples:
                    print(f"[{split}] stopping early at {max_samples} samples")
                    return records

    print(f"[{split}] Loaded {len(records)} samples. Skipped {len(corrupted_images)} corrupted images.")
    return records

# -----------------------------
# Processor & collate_fn
# -----------------------------
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

def collate_fn(batch):
    images = [x["image"] for x in batch]
    captions = [x["caption"] for x in batch]
    inputs = processor(images=images, text=captions,
                       padding="max_length", truncation=True,
                       max_length=MAX_CAPTION_LEN, return_tensors="pt")
    inputs["labels"] = inputs["input_ids"]
    return inputs

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
# Main training
# -----------------------------
if __name__ == "__main__":
    # Load datasets
    train_data = load_split("train", max_samples=MAX_SAMPLES_TRAIN)
    val_data = load_split("val", max_samples=MAX_SAMPLES_VAL)
    train_ds = OnTheFlyDataset(train_data, augment=True)
    val_ds = OnTheFlyDataset(val_data, augment=False)

    # Load pretrained PubLayNet weights
    model = BlipForConditionalGeneration.from_pretrained(MODEL_FOLDER_PUBLAYNET)

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze decoder (text) for fine-tuning
    for param in model.text_decoder.parameters():
        param.requires_grad = True

    # Unfreeze last 6 vision layers to adapt to SciCap
    for param in model.vision_model.encoder.layers[-6:].parameters():
        param.requires_grad = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=MODEL_FOLDER_OUT,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,  # simulate larger batch
        num_train_epochs=10,            # more epochs for convergence
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        logging_steps=100,
        learning_rate=1e-5,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        logging_strategy="steps",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        save_total_limit=2,
        report_to="none"
    )

    trainer = BlipTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn
    )

    print("\n=== STARTING TRAINING ===\n")
    trainer.train()
    trainer.save_model(MODEL_FOLDER_OUT)
    print(f"\nModel saved to: {MODEL_FOLDER_OUT}/model.safetensors\n")

    # -----------------------------
    # Sanity-check inference
    # -----------------------------
    print("=== SANITY-CHECK INFERENCE ===")
    model.eval()
    sample_batch = [val_ds[i] for i in range(min(5, len(val_ds)))]
    images = [x["image"] for x in sample_batch]
    captions_gt = [x["caption"] for x in sample_batch]

    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_CAPTION_LEN,
            num_beams=5,
            no_repeat_ngram_size=3,
            early_stopping=True,
            length_penalty=1.2
        )
        preds = processor.batch_decode(outputs, skip_special_tokens=True)

    for i, (pred, gt) in enumerate(zip(preds, captions_gt)):
        print(f"Sample {i+1}")
        print(f"  Ground truth: {gt}")
        print(f"  Prediction  : {pred}\n")