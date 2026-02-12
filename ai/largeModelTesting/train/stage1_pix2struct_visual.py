"""
Stage 1 Training Script – Pix2Struct on SciCap (Preprocessed Patches)
===================================================================

This script performs **Stage 1 training** of a Pix2Struct model on the
**SciCap dataset**, using *preprocessed image patches stored as `.pt` files*.
It is explicitly engineered to be:

• Fully GPU-friendly  
• Stable on **Windows (CUDA / consumer GPUs)**  
• Memory-efficient for large SciCap corpora  
• Safe for long-running training jobs  

--------------------------------------------------------------------
High-Level Goal
--------------------------------------------------------------------
Train a Pix2Struct encoder–decoder model to map **scientific figures**
to **natural-language captions**, forming the foundation for later
semantic parsing and accessibility-focused refinement stages.

This stage learns *visual-to-language grounding* only.
No semantic graph or accessibility prompting is applied here.

--------------------------------------------------------------------
Pipeline Position
--------------------------------------------------------------------
    Raw SciCap Figures
        ↓ (offline preprocessing)
    Flattened image patches (.pt)
        ↓
    Stage 1 (THIS SCRIPT)
        → Figure captioning model
        ↓
    Stage 2 (AI2D / diagram semantics)
        ↓
    Stage 3 (LLM-based alt-text refinement)

--------------------------------------------------------------------
Key Design Constraints
--------------------------------------------------------------------
• OS: Windows 10 / 11
• GPU: NVIDIA CUDA-capable (consumer-grade)
• VRAM-conscious: supports limited memory via
  - batch size = 1
  - large gradient accumulation
  - activation checkpointing
• No multi-node or distributed assumptions

--------------------------------------------------------------------
Environment Optimizations
--------------------------------------------------------------------
The script explicitly configures runtime flags to improve stability:

• Disables tokenizer multiprocessing
• Enables expandable CUDA memory segments
• Suppresses noisy backend logs
• Avoids NCCL debug spam
• Ignores non-critical warnings

These choices reduce deadlocks and fragmentation issues common
on Windows-based CUDA setups.

--------------------------------------------------------------------
Dataset Format
--------------------------------------------------------------------
Input data consists of **batched `.pt` files**, each containing a list of
samples with the following structure:

    {
        "flattened_patches": Tensor[num_patches, patch_dim],
        "caption": str
    }

Files are named with split prefixes:
• train_*.pt
• val_*.pt

--------------------------------------------------------------------
Lazy Loading Strategy
--------------------------------------------------------------------
To avoid loading the entire SciCap corpus into memory:

• Only **one `.pt` file is loaded at a time**
• File lengths are indexed once at initialization
• Samples are accessed via cumulative offsets
• Cache is cleared automatically when switching files

This allows training on very large datasets using limited RAM.

--------------------------------------------------------------------
Data Collation
--------------------------------------------------------------------
The custom collator:
• Tokenizes captions on CPU (cheap)
• Moves flattened image patches to GPU lazily
• Constructs attention masks dynamically
• Masks padding tokens with `-100` for loss computation

The output is directly compatible with
`Pix2StructForConditionalGeneration`.

--------------------------------------------------------------------
Model Configuration
--------------------------------------------------------------------
• Base model: google/pix2struct-base
• Early encoder layers optionally frozen
• First N encoder blocks frozen for stability
• Activation checkpointing enabled (encoder + decoder)
• Mixed precision (fp16) enabled when CUDA is available

This configuration prioritizes **training stability over raw speed**.

--------------------------------------------------------------------
Training Setup
--------------------------------------------------------------------
• Optimizer: AdamW
• Effective batch size:
    BATCH_SIZE × GRAD_ACCUM_STEPS
• Evaluation & checkpointing:
    - Step-based
    - Best model selected via validation loss
• Logging:
    - Training speed (samples/sec)
    - GPU memory usage
    - Optional system health metrics

--------------------------------------------------------------------
Safety & Debugging Features
--------------------------------------------------------------------
Before full training begins, the script:
• Loads datasets and prints sample counts
• Tests the data collator on a single sample
• Runs a forward pass on GPU to validate shapes
• Aborts early if any incompatibility is detected

This prevents wasting hours on invalid training runs.

--------------------------------------------------------------------
Outputs
--------------------------------------------------------------------
Each run produces a timestamped directory containing:
• Trained Pix2Struct model
• Processor/tokenizer
• Training logs
• Intermediate checkpoints (limited retention)

--------------------------------------------------------------------
Intended Use
--------------------------------------------------------------------
This script is intended for:
• Research-grade pretraining / continued training
• Accessibility and document understanding pipelines
• Serving as a foundation model for later semantic stages

It is **not intended** as an inference-only script.

--------------------------------------------------------------------
"""

import os, time
import threading, queue
import torch, psutil
from torch.utils.data import Dataset
from pathlib import Path
from datetime import datetime
import random
import warnings

# ============================================================
# ENVIRONMENT OPTIMIZATIONS
# ============================================================
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
SCRIPT_DIR = Path(__file__).parent.absolute()
AI_DIR = SCRIPT_DIR.parent
MODEL_BASE_DIR = AI_DIR / "model"
MODEL_BASE_DIR.mkdir(exist_ok=True, parents=True)

BASE_DATA_DIR = r"C:/Users/nawaa/Downloads/scicap_data_preprocessed"
PREPROC_DIR = os.path.join(BASE_DATA_DIR, "Preprocessed-Patches")

# ============================================================
# IMPORT TRANSFORMERS
# ============================================================
from transformers import (
    Pix2StructProcessor,
    Pix2StructForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback
)

# Optional utilities
try:
    from logging_utils import setup_logger, log_metrics
    from reproducibility import set_seed, save_config
    from safety_utils import log_health
except ImportError:
    def setup_logger(name, path): return None
    def log_health(logger): pass
    def set_seed(seed): random.seed(seed); torch.manual_seed(seed)

# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    BASE_MODEL = "google/pix2struct-base"
    IMAGE_SIZE = 256
    BATCH_SIZE = 1  # inc for full training
    GRAD_ACCUM_STEPS = 16  # Increased to compensate
    LEARNING_RATE = 5e-5
    EPOCHS = 1  # Just 1 epoch for testing
    WARMUP_STEPS = 50  # Reduced
    MAX_LENGTH = 256

    FREEZE_EARLY_LAYERS = True
    NUM_FROZEN_BLOCKS = 2
    MODEL_PREFIX = "pix2struct_scicap_stage1"
    LOGGING_STEPS = 500  # More frequent logging for testing

config = Config()
logger = None

# ============================================================
# DATASET: Lazy loading PT batch files
# ============================================================
class PTBatchDatasetLazy(Dataset):
    """
    Efficient lazy-loading dataset for many PT files.
    Only loads one PT file at a time into memory.
    """
    def __init__(self, split, preproc_dir):
        self.split = split
        self.preproc_dir = preproc_dir
        self.files = sorted(
            f for f in os.listdir(preproc_dir) if f.startswith(split) and f.endswith(".pt")
        )
        
        # Only store file lengths and cumulative sizes, not data
        self.file_lengths = []
        self.cumulative_sizes = [0]
        for f in self.files:
            file_path = os.path.join(preproc_dir, f)
            # Get number of samples without loading full data
            data = torch.load(file_path, map_location="cpu")
            self.file_lengths.append(len(data))
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + len(data))
        
        self.total_samples = self.cumulative_sizes[-1]
        self.data_cache = {}  # only 1 file in memory at a time

    def _find_file_idx(self, idx):
        for i in range(len(self.cumulative_sizes) - 1):
            if self.cumulative_sizes[i] <= idx < self.cumulative_sizes[i+1]:
                return i, idx - self.cumulative_sizes[i]
        raise IndexError(f"Sample index {idx} out of range")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        file_idx, local_idx = self._find_file_idx(idx)
        if file_idx not in self.data_cache:
            # Load the PT file lazily
            file_path = os.path.join(self.preproc_dir, self.files[file_idx])
            self.data_cache.clear()  # keep only one file
            self.data_cache[file_idx] = torch.load(file_path, map_location="cpu")
        sample = self.data_cache[file_idx][local_idx]
        return {
            "flattened_patches": sample["flattened_patches"],
            "caption": sample["caption"]
        }
# ============================================================
# DATA COLLATOR
# ============================================================
class SciCapDataCollatorLazy:
    def __init__(self, tokenizer, max_length, device="cuda"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.device = device

    def __call__(self, batch):
        # Tokenize captions first (small memory)
        captions = [item["caption"] for item in batch]
        text_inputs = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True
        )

        labels = text_inputs.input_ids.clone()
        labels[labels == self.pad_token_id] = -100

        # Move everything to GPU one by one (lazy)
        flattened_patches = torch.stack([item["flattened_patches"] for item in batch]).to(self.device)
        attention_mask = torch.ones(flattened_patches.shape[:2], dtype=torch.long, device=self.device)
        decoder_mask = text_inputs.attention_mask.to(self.device)

        return {
            "flattened_patches": flattened_patches,
            "attention_mask": attention_mask,
            "labels": labels.to(self.device),
            "decoder_attention_mask": decoder_mask
        }

# ============================================================
# CALLBACK
# ============================================================
class SpeedOptimizedCallback(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.logger and self.start_time:
            elapsed = time.time() - self.start_time
            steps_per_sec = state.global_step / elapsed if elapsed > 0 else 0
            samples_per_sec = steps_per_sec * config.BATCH_SIZE * config.GRAD_ACCUM_STEPS
            
            # Log training speed
            self.logger.info(
                f"Step {state.global_step}/{state.max_steps} | "
                f"Speed: {samples_per_sec:.1f} samples/sec"
            )
            
            # ---- NEW: Log GPU memory usage ----
            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated() / 1e9
                mem_reserved = torch.cuda.memory_reserved() / 1e9
                self.logger.info(
                    f"GPU memory | Allocated: {mem_alloc:.2f} GB | Reserved: {mem_reserved:.2f} GB"
                )

            try:
                log_health(self.logger)
            except Exception:
                pass

# ============================================================
# MODEL SETUP
# ============================================================
def setup_model():
    model = Pix2StructForConditionalGeneration.from_pretrained(config.BASE_MODEL)
    
    if config.FREEZE_EARLY_LAYERS:
        # Freeze image embedding layer
        for name, param in model.named_parameters():
            if name.startswith("encoder.embed"):
                param.requires_grad = False
        
        # Freeze the first 8 encoder blocks
        for i in range(8):
            for name, param in model.named_parameters():
                if name.startswith(f"encoder.block.{i}"):
                    param.requires_grad = False

    # --- NEW: Enable activation checkpointing for memory efficiency ---
    if hasattr(model.encoder, "block"):
        for module in model.encoder.block:
            module.gradient_checkpointing_enable()

    # Optional: also for decoder if needed (memory heavy)
    if hasattr(model.decoder, "block"):
        for module in model.decoder.block:
            module.gradient_checkpointing_enable()

    return model
# ============================================================
# MAIN TRAINING
# ============================================================
def main():
    global logger
    set_seed(42)

    # Output folder
    MODEL_NAME = f"{config.MODEL_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    OUTPUT_DIR = os.path.join(MODEL_BASE_DIR, MODEL_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger = setup_logger("stage1_scicap", OUTPUT_DIR)
    if logger:
        logger.info("Logger initialized, starting training...")

    processor = Pix2StructProcessor.from_pretrained(config.BASE_MODEL)
    
    print(f"\nProcessor type: {type(processor)}")
    print(f"Tokenizer pad token id: {processor.tokenizer.pad_token_id}")

    # Load datasets WITH LIMITED BATCHES FOR TESTING
    print("\n" + "="*50)
    print("Loading train dataset...")
    train_dataset = PTBatchDatasetLazy("train", PREPROC_DIR)
    print("\n" + "="*50)
    print("Loading val dataset...")
    val_dataset   = PTBatchDatasetLazy("val", PREPROC_DIR) 
    
    # Rest of the code remains the same...

    # Print dataset info
    print(f"\nTRAIN dataset: {len(train_dataset)} samples")
    print(f"VAL dataset: {len(val_dataset)} samples")
    
    # Test one sample from data collator
    print("\nTesting one sample through data collator...")
    data_collator = SciCapDataCollatorLazy(processor.tokenizer, config.MAX_LENGTH)
    
    # Get a single sample
    single_sample = train_dataset[0]
    print(f"Single sample keys: {single_sample.keys()}")
    
    # Create a batch with just this sample
    test_batch = [single_sample]
    try:
        collated = data_collator(test_batch)
        print(f"Collated keys: {collated.keys()}")
        for key, value in collated.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    except Exception as e:
        print(f"ERROR in data collator: {e}")
        import traceback
        traceback.print_exc()
        return
    
    model = setup_model()
    
    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"\nGPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        model.to("cuda")
        
        # Test forward pass
        print("\nTesting forward pass on GPU...")
        model.eval()
        with torch.no_grad():
            test_input = {
                "flattened_patches": collated["flattened_patches"].to("cuda"),
                "attention_mask": torch.ones(
                    collated["flattened_patches"].shape[:2], 
                    dtype=torch.long, device="cuda"
                ),
                "labels": collated["labels"].to("cuda")
            }
            try:
                outputs = model(**test_input)
                print(f"Forward pass successful!")
                print(f"Loss: {outputs.loss.item() if outputs.loss is not None else 'N/A'}")
            except Exception as e:
                print(f"ERROR in forward pass: {e}")
                import traceback
                traceback.print_exc()
                return
    else:
        print("WARNING: CUDA not available, using CPU (slow!)")
        model.to("cpu")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        learning_rate=config.LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=config.WARMUP_STEPS,
        eval_strategy="steps",
        eval_steps=2000,
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_strategy="steps",
        logging_steps=config.LOGGING_STEPS,
        report_to="none",
        fp16=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        seed=42
    )

    print(f"\n" + "="*50)
    print(f"Starting training with:")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Gradient accumulation steps: {config.GRAD_ACCUM_STEPS}")
    print(f"  Effective batch size: {config.BATCH_SIZE * config.GRAD_ACCUM_STEPS}")
    print(f"  Epochs: {config.EPOCHS}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("="*50 + "\n")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[SpeedOptimizedCallback(logger)] if logger else []
    )

    try:
        start = time.time()
        sample = train_dataset[0]
        end = time.time()
        print(f"Time to load first sample: {end-start:.2f} sec")
        print("Starting training...")
        trainer.train()
        
        print(f"\nTraining complete! Saving model to: {OUTPUT_DIR}")
        trainer.save_model(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)
        
    except Exception as e:
        print(f"ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return

    return model, processor, OUTPUT_DIR

if __name__ == "__main__":
    main()