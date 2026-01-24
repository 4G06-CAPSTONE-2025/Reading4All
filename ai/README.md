# BLIP Single-Image Inference (Backend)


## Definitions (Repo Folders / Purpose)

- **Inference**  
  Used for **pretrained models** to run as a **baseline**.

- **Model**  
  **OUTPUT only**. Storage location for saved model checkpoints / exported models.

- **Test**  
  Testing happens here (accuracy checks, evaluation scripts, metrics, etc.).

- **Train**  
  Where we train **multiple versions** of our model to compare against inference baselines.

## What this script does (Batch Inference + Logging)

This script runs a saved BLIP captioning model on **every image in a given folder** and writes the generated captions to a log file.

### High-level flow
1. Loads the BLIP **processor** and **model** from `MODEL_DIR`
2. Scans `IMAGE_DIR` for image files (`.png`, `.jpg`, `.jpeg`, `.webp`)
3. For each image:
   - opens it with PIL
   - runs BLIP generation using the text prompt in `PROMPT`
   - decodes the output into a caption string
   - appends a line to the log: `<image_name>: <caption>`
4. If any image fails, it logs `ERROR` plus a stack trace (and continues to the next image)

All results are appended to: `ai/logger/log.txt`

---

### Changeable paths (what they mean + examples)

These variables at the top control where the script reads from and writes to. 
To run BLIP inference on **one image at a time** and return a caption.
Users must provide:

```python
MODEL_DIR = "ai/models/TesterOneFrozen"
LOGGER_PATH = "ai/logger"
IMAGE_DIR = "ai/data/tester_1/val_data"
PROMPT = "Describe this physics diagram:"
```

## Requirements

- Python 3.9+ recommended

Install dependencies:

```bash
pip install torch transformers pillow
```

