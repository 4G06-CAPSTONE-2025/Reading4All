import os
import time
import torch
from PIL import Image
from datetime import datetime
from collections import Counter
from datasets import load_dataset
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback
)

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

MODEL_BASE_DIR = os.path.join(PROJECT_ROOT, "ai", "model")
os.makedirs(MODEL_BASE_DIR, exist_ok=True)

# ============================================================
# Labels (PubLayNet official)
# ============================================================
LABELS = ["text", "title", "list", "table", "figure"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# ============================================================
# Custom Trainer
# ============================================================
class BlipTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ============================================================
# Heartbeat Callback
# ============================================================
class HeartbeatCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0 and state.global_step > 0:
            loss = state.log_history[-1].get("loss", "N/A")
            print(
                f"[HEARTBEAT] step={state.global_step} "
                f"epoch={state.epoch:.2f} loss={loss}"
            )

# ============================================================
# Dataset loader (HF PubLayNet)
# ============================================================
def load_publaynet_hf(split="train", max_samples=None):
    print("\n=== Loading PubLayNet (metadata only) ===\n")

    dataset = load_dataset(
        "creative-graphic-design/PubLayNet",
        split=split,
        streaming=False   # important
    )

    PUBLAYNET_ID2LABEL = {
        0: "text",
        1: "title",
        2: "list",
        3: "table",
        4: "figure",
    }

    records = []
    skipped = 0

    for idx, item in enumerate(dataset):
        if idx % 1000 == 0 and idx > 0:
            print(f"[HEARTBEAT] processed {idx} | kept {len(records)}")

        ann = item.get("annotations", {})
        category_ids = ann.get("category_id", [])

        if not category_ids:
            skipped += 1
            continue

        categories = [
            PUBLAYNET_ID2LABEL[cid]
            for cid in category_ids
            if cid in PUBLAYNET_ID2LABEL
        ]

        if not categories:
            skipped += 1
            continue

        dominant = Counter(categories).most_common(1)[0][0]

        records.append({
            "index": idx,
            "label": LABEL2ID[dominant]
        })

        if max_samples and len(records) >= max_samples:
            break

    print(f"\nLoaded {len(records)} samples | skipped {skipped}\n")
    return records

# ============================================================
# Dataset wrapper
# ============================================================
class PubLayNetDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, records):
        self.dataset = hf_dataset
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        item = self.dataset[rec["index"]]

        image = item["image"].convert("RGB")
        return {
            "image": image,
            "label": rec["label"]
        }

# ============================================================
# Processor & collate
# ============================================================
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

def collate_fn(batch):
    images = [b["image"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch])

    inputs = processor(images=images, return_tensors="pt")
    inputs["labels"] = labels
    return inputs

# ============================================================
# Trainable parameters logger
# ============================================================
def log_trainable_parameters(model):
    print("\n=== TRAINABLE PARAMETERS ===")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    print("============================\n")

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    # -----------------------------
    # Load dataset
    # -----------------------------
    hf_dataset = load_dataset(
    "creative-graphic-design/PubLayNet",
    split="train")

    records = load_publaynet_hf(max_samples=5000) #CHANGE THIS BEFORE ACTUAL TRAINING
    dataset = PubLayNetDataset(hf_dataset, records)


    # -----------------------------
    # Load BLIP
    # -----------------------------
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    # -----------------------------
    # Freeze everything
    # -----------------------------
    for param in model.parameters():
        param.requires_grad = False

    # -----------------------------
    # Unfreeze vision encoder
    # -----------------------------
    for param in model.vision_model.parameters():
        param.requires_grad = True

    # -----------------------------
    # Add classification head
    # -----------------------------
    vision_dim = model.vision_model.config.hidden_size
    model.vision_classifier = torch.nn.Linear(
        vision_dim, len(LABELS)
    )

    # -----------------------------
    # Override forward (vision-only)
    # -----------------------------
    def vision_forward(pixel_values, labels=None):
        outputs = model.vision_model(pixel_values=pixel_values)
        pooled = outputs.last_hidden_state[:, 0]
        logits = model.vision_classifier(pooled)
        return type("Out", (), {"logits": logits})

    model.forward = vision_forward

    # -----------------------------
    # Device
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    log_trainable_parameters(model)

    # -----------------------------
    # Output dir
    # -----------------------------
    MODEL_NAME = f"blip_publaynet_vision_{datetime.now().strftime('%Y%m%d_%H%M')}"
    OUTPUT_DIR = os.path.join(MODEL_BASE_DIR, MODEL_NAME)

    # -----------------------------
    # Training args
    # -----------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,
        learning_rate=1e-4,
        num_train_epochs=2,
        logging_steps=50,
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = BlipTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        callbacks=[HeartbeatCallback()],
    )

    # -----------------------------
    # Train
    # -----------------------------
    print("\n=== STARTING PubLayNet VISION PRETRAINING ===\n")
    start = time.time()

    trainer.train()

    elapsed = (time.time() - start) / 60
    print(f"\n=== TRAINING COMPLETE ({elapsed:.2f} min) ===\n")

    trainer.save_model(OUTPUT_DIR)