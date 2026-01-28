# SciCap BLIP Training Configuration

This README documents the purpose and usage of the JSON configuration files used 
for training and inference of the BLIP-based SciCap image captioning models.

---

## 1. `iter1_paths_config.json`

**Location:** `ai/train/iter1_paths_config.json`  

**Purpose:**  
Specifies folder paths for the first iteration of training the SciCap BLIP model.

**Contents:**

| Key | Description | Example/Value |
|-----|------------|---------------|
| `base_data_dir` | Root folder containing all SciCap dataset files | `C:/Users/nawaa/Downloads/scicap_data_extracted/scicap_data` |
| `img_no_dir` | Folder name for images without subfigures | `Scicap-No-Subfig-Img` |
| `img_yes_dir` | Folder name for images with subfigures (used during training) | `SciCap-Yes-Subfig-Img` |
| `captions_dir` | Folder containing JSON caption files | `SciCap-Caption-All` |
| `model_dir` | Directory to save the trained model | `model/scicap_blip_model_final` |
| `test_images_dir` | Directory containing images for inference/testing | `inference/images` |

**Usage:**  
This file is loaded in the training scripts to locate data, captions, and save outputs.

---

## 2. `iter1_training_config.json`

**Location:** `C:/Users/nawaa/OneDrive/Desktop/Reading4All/ai/train/iter1_training_config.json`  

**Purpose:**  
Specifies training hyperparameters and strategies for the first iteration of SciCap BLIP fine-tuning.

**Contents:**

| Key | Description | Example/Value |
|-----|------------|---------------|
| `per_device_train_batch_size` | Batch size per GPU/CPU device | 16 |
| `per_device_eval_batch_size` | Evaluation batch size per device | 16 |
| `gradient_accumulation_steps` | Steps to accumulate gradients before backprop | 1 |
| `num_train_epochs` | Number of training epochs | 1 |
| `eval_strategy` | Evaluation strategy (e.g., steps or epoch) | `steps` |
| `eval_steps` | Frequency of evaluation in steps | 1000 |
| `save_strategy` | Model checkpoint saving strategy | `steps` |
| `save_steps` | Frequency of saving model checkpoints | 1000 |
| `logging_steps` | Frequency of logging | 1000 |
| `fp16` | Use mixed-precision training | true |
| `remove_unused_columns` | Whether to remove unused columns from dataset | false |
| `logging_strategy` | Logging strategy (steps/epoch) | `steps` |
| `log_level` | Logging verbosity | `info` |
| `disable_tqdm` | Disable progress bars | false |
| `dataloader_num_workers` | Number of subprocesses for data loading | 4 |
| `dataloader_pin_memory` | Pin memory in data loader for faster transfer to GPU | true |

**Usage:**  
Loaded in training scripts via `TrainingArguments` from Hugging Face Transformers.

---

## 3. `paths_config.json`

**Location:** `C:/Users/nawaa/OneDrive/Desktop/Reading4All/ai/train/paths_config.json`  

**Purpose:**  
General paths configuration for multi-iteration or preprocessed SciCap dataset usage.

**Contents:**

| Key | Description | Example/Value |
|-----|------------|---------------|
| `base_data_dir` | Folder containing preprocessed SciCap data | `C:/Users/nawaa/Downloads/scicap_data_preprocessed` |
| `og_base_dir` | Original raw SciCap dataset folder | `C:/Users/nawaa/Downloads/scicap_data_extracted/scicap_data` |
| `img_no_dir` | Folder name for images without subfigures | `Scicap-No-Subfig-Img` |
| `img_yes_dir` | Folder name for images with subfigures | `SciCap-Yes-Subfig-Img` |
| `captions_dir` | Folder containing all caption JSONs | `SciCap-Caption-All` |
| `model_dir` | Folder to save or load trained models | `model` |
| `data_dir` | Folder for processed datasets (PyTorch/JSON) | `data` |
| `test_images_dir` | Folder for images used in inference/testing | `inference/images` |

**Usage:**  
Provides a consistent way to locate all dataset, caption, model, and test files across scripts.

---

### Notes:
- `iter1_paths_config.json` and `iter1_training_config.json` are specific to the first training iteration.  
- `paths_config.json` is more general and can be used for future iterations or preprocessed datasets.  
- All paths are absolute to avoid ambiguity and allow scripts to run from any working directory.  
- Ensure consistency between `paths_config.json` and `training_config.json` when modifying batch sizes or dataset paths.