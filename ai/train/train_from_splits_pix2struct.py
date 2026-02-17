import os
import torch
from PIL import Image
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, default_data_collator, Pix2StructProcessor, Pix2StructForConditionalGeneration

'''
This is what input dir should look like
<BASE_DIR>/
    ai/ 
        train/
            train.csv (or train_rel.csv)
            val.csv   (or val_rel.csv)
            train_data/
                img_0001.png
                img_0002.png
                ...
            val_data/
                img_0101.png
                img_0102.png
                ...
where the csv files have all the columns in the original annotated csv in ai/annotations/annotated_physics_data(Sheet1).csv
'''
# TESTER=1
MODEL_ID = "google/pix2struct-base"
BASE_DIR = "C:/Users/nawaa/OneDrive/Desktop/Reading4All/ai/train"     # CHANGE THIS to your local path where your split is 
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
VAL_CSV = os.path.join(BASE_DIR, "val.csv")
OUT_DIR = "C:/Users/nawaa/OneDrive/Desktop/Reading4All/ai/model"     # CHANGE THIS to your local path 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# torch.backends.mps.enable_fallback = True

processor = Pix2StructProcessor.from_pretrained(MODEL_ID)
model = Pix2StructForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)
 
# for param in model.vision_model.parameters():
#     param.requires_grad = False


ds = load_dataset("csv", data_files={"train": TRAIN_CSV, "validation": VAL_CSV})

MAX_LEN = 256

def preprocess(examples):
    image_paths = examples["Image-Path"]
    images = []
    for p in image_paths:
        with Image.open(p) as img:
            images.append(img.convert("RGB"))


    texts = [t.strip() for t in examples["Modified-Alt-Text"]]

    enc = processor(
    images=images,
    padding="max_length",
    return_tensors="pt",
    )

    labels = processor.tokenizer(
    texts,
    padding="max_length",
    truncation=True,
    max_length=MAX_LEN,
    return_tensors="pt",
    ).input_ids

    labels[labels == processor.tokenizer.pad_token_id] = -100
    enc["labels"] = labels

    return {
        "flattened_patches": enc["flattened_patches"],
        "attention_mask": enc["attention_mask"],
        "labels": enc["labels"],
    }

ds = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)
ds.set_format(
    type="torch",
    columns=["flattened_patches", "attention_mask", "labels"]
)


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
    gradient_accumulation_steps=1,
    learning_rate=5e-5,
    num_train_epochs=5,
    warmup_ratio=0.05,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=10,
    remove_unused_columns=False,
    fp16=False,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
)


# for quick train debug

# FAST_DEBUG = True 
# args = TrainingArguments(
#     output_dir=OUT_DIR,
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=1,
#     gradient_accumulation_steps=1 if FAST_DEBUG else 8,
#     learning_rate=5e-5,
#     num_train_epochs=1 if FAST_DEBUG else 3,
#     max_steps=20 if FAST_DEBUG else -1,  
#     eval_strategy="no" if FAST_DEBUG else "epoch",
#     save_strategy="no",
#     logging_steps=1 if FAST_DEBUG else 25,
#     fp16=False,
#     dataloader_num_workers=0 if FAST_DEBUG else 2,
#     dataloader_pin_memory=False,
#     disable_tqdm=False if FAST_DEBUG else False,
#     report_to="none",
#     remove_unused_columns=False,
# )


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
