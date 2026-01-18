import os
import json
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, TrainingArguments, Trainer
from datetime import datetime

# -----------------------------
# Boosted training parameters
# -----------------------------
MAX_SAMPLES_TRAIN = 500
MAX_SAMPLES_VAL = 200
NUM_EPOCHS = 2
BATCH_SIZE = 8
MAX_CAPTION_LEN = 128
TRAINABLE_VISION_LAYERS = 2  # unfreeze last N layers of vision encoder

# -----------------------------
# Paths
# -----------------------------
BASE = r"C:/Users/nawaa/Downloads/scicap_data_extracted/scicap_data"
IMG_NO = os.path.join(BASE, "Scicap-No-Subfig-Img")
IMG_YES = os.path.join(BASE, "SciCap-Yes-Subfig-Img")
CAP_ALL = os.path.join(BASE, "SciCap-Caption-All")

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
        except:
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
    for jf in [f for f in os.listdir(split_folder) if f.endswith(".json")]:
        with open(os.path.join(split_folder, jf), "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            for item in data:
                img_id = os.path.splitext(item["figure-ID"])[0].lower()
                caption = item["2-normalized"]["2-1-basic-num"]["caption"].strip()
                found = False
                for folder in [IMG_NO, IMG_YES] if split.lower() == "train" else [IMG_NO]:
                    folder_split = os.path.join(folder, split.lower())
                    for ext in [".png", ".jpg", ".jpeg"]:
                        img_path = os.path.join(folder_split, img_id + ext)
                        if os.path.exists(img_path):
                            records.append({"image_path": img_path, "caption": caption})
                            found = True
                            break
                    if found:
                        break
                if max_samples and len(records) >= max_samples:
                    print(f"[{split}] stopping early at {max_samples} samples")
                    return records
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
# Main
# -----------------------------
if __name__ == "__main__":
    train_data = load_split("train", MAX_SAMPLES_TRAIN)
    val_data   = load_split("val", MAX_SAMPLES_VAL)
    train_ds = OnTheFlyDataset(train_data)
    val_ds = OnTheFlyDataset(val_data)

    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Freeze all first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze all text decoder
    for param in model.text_decoder.parameters():
        param.requires_grad = True

    # Optionally unfreeze last N vision layers
    if hasattr(model, "vision_encoder") and hasattr(model.vision_encoder, "layer"):
        vision_layers = model.vision_encoder.layer
        for layer in vision_layers[-TRAINABLE_VISION_LAYERS:]:
            for param in layer.parameters():
                param.requires_grad = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    MODEL_DIR = os.path.join("model_test", f"boosted_training_{datetime.now().strftime('%H%M')}")
    os.makedirs(MODEL_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        logging_steps=25,
        learning_rate=1e-5,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = BlipTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn
    )

    print("\n=== STARTING BOOSTED TRAINING ===\n")
    trainer.train()
    trainer.save_model(MODEL_DIR)
    print(f"\nModel saved to: {MODEL_DIR}\n")

    # -----------------------------
    # Sanity-check inference
    # -----------------------------
    print("=== SANITY-CHECK INFERENCE ===\n")
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