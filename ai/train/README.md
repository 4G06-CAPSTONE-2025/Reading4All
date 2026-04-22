# ai/train/

This folder contains all scripts for fine-tuning and training vision-language models on the Reading4All annotated dataset.

---

## Folder Structure

| Item | Description |
|---|---|
| `classifier/` | Classifier model training utilities |
| `train.py` | Primary training script — use this for standard fine-tuning runs |
| `train_from_splits.py` | Trains using pre-split train/val/test data |
| `train_from_splits_NEW.py` | Updated version of `train_from_splits.py` with increased max token length |
| `train_from_splits_pix2struct.py` | Training script specifically for the Pix2Struct model (3-epoch run) |

---

## Quickstart

### Train the BLIP model

```bash
python ai/train/train.py
```

### Train using pre-split data

```bash
python ai/train/train_from_splits_NEW.py
```

### Train Pix2Struct (comparison baseline)

```bash
python ai/train/train_from_splits_pix2struct.py
```

---

## Training Notes

- Training epochs were reduced from 5 to 3 for improved efficiency after early experiments showed diminishing returns beyond 3 epochs
- `train_from_splits_NEW.py` is the recommended script for BLIP fine-tuning — it includes comments and an increased max sequence length
- Pix2Struct was trained as a comparison baseline; BLIP was selected as the final model
- Use `ai/utils/splitData.py` to generate the train/val/test splits before running split-based training scripts
- Trained model checkpoints are saved to the model output directory configured in the script

---

## Dependencies

```bash
pip install torch torchvision transformers pillow
```
