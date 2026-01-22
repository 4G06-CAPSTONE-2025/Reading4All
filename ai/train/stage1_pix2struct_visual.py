import os
import time
import torch
from datetime import datetime
from collections import Counter
from datasets import load_dataset
from transformers import (
    Pix2StructProcessor,
    Pix2StructForConditionalGeneration,
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

# ============================================================
# Heartbeat callback
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
# Dataset loader
# ============================================================
def load_publaynet_hf(split="train", max_samples=None):
    print("\n=== Loading PubLayNet ===\n")

    dataset = load_dataset(
        "creative-graphic-design/PubLayNet",
        split=split,
        streaming=False
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

        dominant_label = Counter(categories).most_common(1)[0][0]

        records.append({
            "index": idx,
            "text": dominant_label
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
        text = rec["text"]

        return {
            "image": image,
            "text": text
        }

# ============================================================
# Processor & collate (Pix2Struct-style)
# ============================================================
processor = Pix2StructProcessor.from_pretrained(
    "google/pix2struct-base"
)

# LIMIT visual tokens (VERY IMPORTANT)
processor.image_processor.max_patches = 512

def shift_tokens_right(labels, pad_token_id, decoder_start_token_id):
    shifted = labels.new_zeros(labels.shape)
    shifted[:, 1:] = labels[:, :-1].clone()
    shifted[:, 0] = decoder_start_token_id
    shifted.masked_fill_(shifted == -100, pad_token_id)
    return shifted


def collate_fn(batch):
    images = [b["image"] for b in batch]
    texts = [b["text"] for b in batch]

    # Pix2Struct image processing (IMPORTANT)
    image_inputs = processor.image_processor(
        images,
        return_tensors="pt"
    )

    # Tokenize labels
    labels = processor.tokenizer(
        texts,
        padding="longest",
        truncation=True,
        return_tensors="pt"
    ).input_ids

    # Ignore padding tokens in loss
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Shift decoder inputs
    decoder_input_ids = shift_tokens_right(
        labels,
        pad_token_id=-100,
        decoder_start_token_id=model.config.decoder_start_token_id
    )

    return {
        "flattened_patches": image_inputs["flattened_patches"],
        "attention_mask": image_inputs["attention_mask"],
        "labels": labels,
        "decoder_input_ids": decoder_input_ids,
    }

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
        split="train"
    )

    records = load_publaynet_hf()
    dataset = PubLayNetDataset(hf_dataset, records)

    # -----------------------------
    # Load Pix2Struct
    # -----------------------------
    model = Pix2StructForConditionalGeneration.from_pretrained(
        "google/pix2struct-base"
    )

    # -----------------------------
    # Freeze everything
    # -----------------------------
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze decoder token embeddings
    # (this learns the label vocabulary: text/title/list/table/figure):
    for param in model.decoder.embed_tokens.parameters():
        param.requires_grad = True
    
    # Unfreeze LayerNorms in encoder for stability
    # (acts like lightweight adapters)
    # -----------------------------
    for name, param in model.encoder.named_parameters():
        if "layernorm" in name.lower():
            param.requires_grad = True

    # -----------------------------
    # Device
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    log_trainable_parameters(model)

    # -----------------------------
    # Output dir
    # -----------------------------
    MODEL_NAME = f"pix2struct_publaynet_stage1_{datetime.now().strftime('%Y%m%d_%H%M')}"
    OUTPUT_DIR = os.path.join(MODEL_BASE_DIR, MODEL_NAME)

    # -----------------------------
    # Training args
    # -----------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,   # effective batch = 8
        learning_rate=1e-4,
        num_train_epochs=2,
        logging_steps=50,
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to="none",
        fp16=True,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        callbacks=[HeartbeatCallback()],
    )

    # -----------------------------
    # Train
    # -----------------------------
    print("\n=== STARTING PubLayNet PIX2STRUCT STAGE 1 (IMAGE → LABEL TEXT) ===\n")
    start = time.time()

    trainer.train()

    elapsed = (time.time() - start) / 60
    print(f"\n=== TRAINING COMPLETE ({elapsed:.2f} min) ===\n")

    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)