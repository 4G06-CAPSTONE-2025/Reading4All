import os
import torch
from datasets import load_dataset
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration, get_scheduler
from torch.optim import AdamW
import math

from utils.reproducibility import set_seed, save_config
from utils.logging_utils import setup_logger, log_metrics
from utils.progress_utils import progress
from utils.safety_utils import log_health, cooldown

# ---------------- OPTIMIZED CONFIG ----------------
CFG = {
    "model": "google/pix2struct-base",
    "dataset": "PubLayNet",
    "epochs": 2,                    # Full epochs
    "batch_size": 4,                # Increased batch size for efficiency
    "lr": 5e-5,
    "warmup_steps": 1000,           # Learning rate warmup
    "seed": 42,
    "gradient_accumulation_steps": 2,  # Simulate larger batch size
    "save_steps": 5000,             # Save checkpoint every 5000 steps
    "logging_steps": 100            # Log every 100 steps
}

OUT = "models/stage1_visual_full"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- SYSTEM SAFETY --------
torch.set_num_threads(6)
os.environ["OMP_NUM_THREADS"] = "6"
if torch.cuda.is_available():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    torch.backends.cudnn.benchmark = True  # Enable cuDNN autotuner

# -------- SETUP --------
set_seed(CFG["seed"])
save_config(CFG, OUT)
logger = setup_logger("stage1_full", OUT)
logger.info(f"Training on device: {DEVICE}")
logger.info(f"Configuration: {CFG}")

# -------- DATA --------
# PubLayNet has ~335,703 training images
dataset = load_dataset("jordanparker6/publaynet", split="train", streaming=True)
logger.info("PubLayNet dataset loaded (streaming mode)")

# Calculate approximate total steps
# PubLayNet size: ~335,703 images
if CFG.get("max_steps"):
    total_steps = CFG["max_steps"]
else:
    total_steps = math.ceil(335703 / CFG["batch_size"]) * CFG["epochs"]
    
logger.info(f"Approximate total steps: {total_steps:,}")
logger.info(f"Training for {CFG['epochs']} epochs")
logger.info(f"Batch size: {CFG['batch_size']}")

# -------- MODEL --------
processor = Pix2StructProcessor.from_pretrained(CFG["model"])
model = Pix2StructForConditionalGeneration.from_pretrained(
    CFG["model"], 
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

optimizer = AdamW(model.parameters(), lr=CFG["lr"])

# Learning rate scheduler
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=CFG["warmup_steps"],
    num_training_steps=total_steps
)

logger.info(f"Model loaded: {CFG['model']}")
logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# -------- SAVE INITIAL MODEL --------
initial_out = f"{OUT}/initial"
os.makedirs(initial_out, exist_ok=True)
model.save_pretrained(initial_out)
processor.save_pretrained(initial_out)
logger.info(f"Saved initial model to {initial_out}")

# -------- TRAINING --------
step = 0
global_step = 0
model.train()
loss_history = []
gradient_accumulation_counter = 0

for epoch in range(CFG["epochs"]):
    epoch_loss = 0
    epoch_samples = 0
    batch_images = []
    
    logger.info(f"Starting epoch {epoch + 1}/{CFG['epochs']}")
    
    for sample in progress(dataset, f"Epoch {epoch+1}"):
        try:
            # Add to batch
            batch_images.append(sample["image"])
            
            # Process batch when full
            if len(batch_images) == CFG["batch_size"]:
                # Prepare batch inputs
                inputs = processor(images=batch_images, return_tensors="pt", padding=True)
                
                # Move to device
                if DEVICE == "cuda":
                    inputs = {k: v.to(DEVICE, dtype=torch.float16) for k, v in inputs.items()}
                else:
                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                
                # Prepare labels for each image in batch
                batch_labels = processor.tokenizer(
                    ["describe the diagram structure"] * CFG["batch_size"],
                    return_tensors="pt",
                    padding=True
                ).input_ids.to(DEVICE)
                
                # Forward pass
                out = model(**inputs, labels=batch_labels)
                loss = out.loss
                
                # Scale loss for gradient accumulation
                loss = loss / CFG["gradient_accumulation_steps"]
                loss.backward()
                
                gradient_accumulation_counter += 1
                epoch_loss += loss.item() * CFG["gradient_accumulation_steps"]
                epoch_samples += CFG["batch_size"]
                
                # Update weights when accumulation steps reached
                if gradient_accumulation_counter >= CFG["gradient_accumulation_steps"]:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    gradient_accumulation_counter = 0
                    
                    step += 1
                    global_step += 1
                    loss_history.append(loss.item() * CFG["gradient_accumulation_steps"])
                
                # Clear batch
                batch_images = []
                
                # Logging
                if step % CFG["logging_steps"] == 0 and step > 0:
                    avg_loss = sum(loss_history[-CFG["logging_steps"]:]) / min(len(loss_history), CFG["logging_steps"])
                    current_lr = optimizer.param_groups[0]["lr"]
                    
                    logger.info(
                        f"Step {step:,} | Loss: {loss.item() * CFG['gradient_accumulation_steps']:.4f} | "
                        f"Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | "
                        f"Samples: {epoch_samples:,}"
                    )
                    
                    log_health(logger)
                    log_metrics({
                        "step": step,
                        "global_step": global_step,
                        "loss": loss.item() * CFG["gradient_accumulation_steps"],
                        "avg_loss": avg_loss,
                        "learning_rate": current_lr,
                        "epoch": epoch + 1,
                        "samples_processed": epoch_samples
                    }, OUT)
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    cooldown()
                
                # Save checkpoint
                if step % CFG["save_steps"] == 0 and step > 0:
                    checkpoint_dir = f"{OUT}/step_{step}"
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    processor.save_pretrained(checkpoint_dir)
                    
                    checkpoint_info = {
                        "step": step,
                        "global_step": global_step,
                        "epoch": epoch + 1,
                        "loss": loss.item() * CFG["gradient_accumulation_steps"],
                        "avg_loss": sum(loss_history[-100:]) / min(len(loss_history), 100),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "samples_processed": epoch_samples
                    }
                    
                    import json
                    with open(os.path.join(checkpoint_dir, "checkpoint_info.json"), "w") as f:
                        json.dump(checkpoint_info, f, indent=2)
                    
                    logger.info(f"✅ Saved checkpoint at step {step:,}")
        
        except Exception as e:
            logger.error(f"Error processing batch at step {step}: {e}")
            batch_images = []  # Reset batch on error
            continue
    
    # -------- SAVE EPOCH CHECKPOINT --------
    epoch_dir = f"{OUT}/epoch_{epoch+1}"
    os.makedirs(epoch_dir, exist_ok=True)
    
    # Save model and processor
    model.save_pretrained(epoch_dir)
    processor.save_pretrained(epoch_dir)
    
    # Save epoch summary
    epoch_avg_loss = epoch_loss / max(step, 1)
    epoch_info = {
        "epoch": epoch + 1,
        "total_steps": step,
        "global_steps": global_step,
        "average_loss": epoch_avg_loss,
        "samples_processed": epoch_samples,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "config": CFG
    }
    
    import json
    with open(os.path.join(epoch_dir, "epoch_info.json"), "w") as f:
        json.dump(epoch_info, f, indent=2)
    
    logger.info(f"✅ Epoch {epoch+1} complete")
    logger.info(f"   Steps: {step:,}")
    logger.info(f"   Samples: {epoch_samples:,}")
    logger.info(f"   Average Loss: {epoch_avg_loss:.4f}")
    logger.info(f"   Saved to: {epoch_dir}")

# -------- SAVE FINAL MODEL --------
final_dir = f"{OUT}/final"
os.makedirs(final_dir, exist_ok=True)

model.save_pretrained(final_dir)
processor.save_pretrained(final_dir)

final_info = {
    "total_steps": step,
    "global_steps": global_step,
    "total_epochs": CFG["epochs"],
    "final_loss": loss_history[-1] if loss_history else 0,
    "average_loss": sum(loss_history) / max(len(loss_history), 1),
    "config": CFG,
    "device": DEVICE,
    "total_parameters": sum(p.numel() for p in model.parameters()),
    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
}

with open(os.path.join(final_dir, "final_info.json"), "w") as f:
    json.dump(final_info, f, indent=2)

logger.info("=" * 60)
logger.info("TRAINING COMPLETE!")
logger.info(f"Total steps: {step:,}")
logger.info(f"Final model saved to: {final_dir}")
logger.info("=" * 60)