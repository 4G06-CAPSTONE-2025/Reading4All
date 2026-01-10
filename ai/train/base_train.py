import os
import torch
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor, BlipForConditionalGeneration, TrainingArguments, Trainer, default_data_collator


'''
This is what input dir should look like
<BASE_DIR>/
  train.csv (or train_rel.csv)
  val.csv   (or val_rel.csv)
  data/
    train/
      img_0001.png
      img_0002.png
      ...
    val/
      img_0101.png
      img_0102.png
      ...
      where the csv files have two columns: 'image' and 'text'
      'image' column has relative paths to images like 'data/train/img_0001.png
      '''
MODEL_ID = "Salesforce/blip-image-captioning-base"
BASE_DIR = "PATH/TO/YOUR/DATASET"
TRAIN_CSV = os.path.join(BASE_DIR, "train_rel.csv")
VAL_CSV = os.path.join(BASE_DIR, "val_rel.csv")
OUT_DIR = "PATH/TO/YOUR/OUTPUT/DIR"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)

ds = load_dataset("csv", data_files={"train": TRAIN_CSV, "validation": VAL_CSV})

MAX_LEN = 128

def preprocess(examples):
    image_paths = [os.path.join(BASE_DIR, p) for p in examples["image"]]
    images = [Image.open(p).convert("RGB") for p in image_paths]
    texts = ["Describe this engineering diagram: " + t for t in examples["text"]]
    enc = processor(
        images=images,
        text=texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    enc["labels"] = enc["input_ids"].clone()
    return enc

ds = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)
ds.set_format(type="torch", columns=["input_ids", "attention_mask", "pixel_values", "labels"])

class FixedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs.pop("num_items_in_batch", None)
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        return (loss, outputs) if return_outputs else loss

args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    num_train_epochs=3,
    warmup_ratio=0.05,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=25,
    remove_unused_columns=False,
    fp16=False,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
)


trainer = FixedTrainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    data_collator=default_data_collator,
)

trainer.train()
trainer.save_model(OUT_DIR)
processor.save_pretrained(OUT_DIR)
print(f"Saved to {OUT_DIR}")
