import os, torch
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from utils.reproducibility import set_seed, save_config
from utils.logging_utils import setup_logger, log_metrics
from utils.progress_utils import progress
from utils.safety_utils import log_health, cooldown

BASE_DIR = r"C:/Users/nawaa/OneDrive/Desktop/Reading4All/ai"
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")

CFG = {
    "model": "t5-base",
    "epochs": 3,
    "batch_size": 2,
    "lr": 3e-5,
    "seed": 42,
    "data": os.path.join(DATA_DIR, "structured_alttext.json")
}

OUT = os.path.join(MODEL_DIR, "stage3_alttext")
DEVICE = "cuda"

set_seed(CFG["seed"])
save_config(CFG, OUT)
logger = setup_logger("stage3", OUT)

dataset = load_dataset("json", data_files=CFG["data"], streaming=True)

tok = T5Tokenizer.from_pretrained(CFG["model"])
model = T5ForConditionalGeneration.from_pretrained(CFG["model"]).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=CFG["lr"])

step = 0
model.train()

for epoch in range(CFG["epochs"]):
    for sample in progress(dataset, f"Epoch {epoch+1}"):

        inp = tok(sample["structured"], return_tensors="pt", truncation=True).to(DEVICE)
        lab = tok(sample["alt_text"], return_tensors="pt", truncation=True).input_ids.to(DEVICE)

        out = model(**inp, labels=lab)
        loss = out.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step += 1

        if step % 100 == 0:
            log_metrics({"step": step, "loss": loss.item()}, OUT)
            torch.cuda.empty_cache()
            cooldown()

model.save_pretrained(OUT)
logger.info("STAGE 3 COMPLETE")