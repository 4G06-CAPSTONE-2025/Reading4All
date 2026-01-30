# BLIP Single-Image Inference


## Definitions (Repo Folders / Purpose)

- **Inference**  
  Used for **pretrained models** to run as a **baseline**.

- **Model**  
  **OUTPUT only**. Storage location for saved model checkpoints / exported models.

- **Logger**  
  Contains inference results

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

All results are appended to: `logger.txt`

---

### Changeable paths (what they mean + examples)

These variables at the top control where the script reads from and writes to. 
To run BLIP inference on **one image at a time** and return a caption.
Users must provide:

```python
MODEL_DIR = "PATH TO THE MODEL FOLDER"
LOGGER_PATH = "PATH TO A FOLDER FOR LOGS"
IMAGE_DIR = "IMAGE INPUT PATH"
PROMPT = "Describe this physics diagram:"
```

## Where to run the script (file location)

The inference script is located at:

`ai/inference/rev0/test.py`

Run it either from the repo root:

```bash
python ai/inference/rev0/test.py
```
## Requirements

- Python 3.9+ recommended

Install dependencies:

```bash
pip install torch transformers pillow
```

