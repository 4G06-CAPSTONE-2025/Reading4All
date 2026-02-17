'''
This is the training script for the moonbeam model. 
It accepts the csv file of generated alt text by BLIP or pix2truct models as its training input. 
It outputs better alt text. 

Update: NOT NEEDED ANYMORE 
'''

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator
)

MODEL_ID = "google/gemma-2b" # try moonbeam-2b 
TRAIN_CSV = "/Users/francinebulaclac/Desktop/Reading4All/ai/train/BLIP_draftCaptions_train.csv"
VAL_CSV   = "/Users/francinebulaclac/Desktop/Reading4All/ai/train/BLIP_draftCaptions_val.csv"

OUT_DIR = "/Users/francinebulaclac/Desktop/Reading4All/ai/train/models/moonbeam"
PROMPT = (
    "Rewrite the following image description to be clear, accurate, "
    "and accessible for a screen reader:\n\n{caption}\n\nAlt text:"
)


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token  # REQUIRED

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32
).to(DEVICE)

ds = load_dataset(
    "csv",
    data_files={
        "train": TRAIN_CSV,
        "validation": VAL_CSV
    }
)

MAX_LEN = 256

def preprocess(examples):
    prompts = [
        PROMPT.format(caption=c.strip())
        for c in examples["generated_caption"]
    ]

    targets = [t.strip() for t in examples["modified_alt_text"]]

    # Tokenize prompt
    model_inputs = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )

    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        )

    # Mask padding tokens
    labels_ids = []
    for seq in labels["input_ids"]:
        labels_ids.append([
            tok if tok != tokenizer.pad_token_id else -100
            for tok in seq
        ])

    model_inputs["labels"] = labels_ids
    return model_inputs


ds = ds.map(
    preprocess,
    batched=True,
    remove_columns=ds["train"].column_names
)

ds.set_format(type="torch")


args = TrainingArguments(
    output_dir=OUT_DIR,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,   # effective batch = 8

    learning_rate=5e-6,              # 🔑 small-data friendly
    num_train_epochs=4,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",

    evaluation_strategy="epoch",
    save_strategy="no",
    logging_steps=10,

    fp16=False,                      # MPS safety
    bf16=False,
    remove_unused_columns=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    data_collator=default_data_collator,
)


trainer.train()

trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print(f"Moonbeam-2 model saved to {OUT_DIR}")
