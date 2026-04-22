# ai/utils/

This folder contains shared utility scripts used to prepare and validate data before training.

---

## Files

| File | Description |
|---|---|
| `splitData.py` | Splits the combined annotation CSV into train, validation, and test sets |
| `splitData_NEW.py` | Updated version of `splitData.py` with added comments |
| `splitData_NF.py` | Variant of the split script configured for Pix2Struct training |
| `test_hf.py` | Tests connectivity and access to Hugging Face model hub |
| `validateImagePaths.py` | Checks that all image paths referenced in the CSV datasets actually exist on disk |

---

## Usage

### Split the dataset

Run this before any split-based training script:

```bash
python ai/utils/splitData_NEW.py
```

This will generate separate CSV files for the train, validation, and test splits.

### Validate image paths

Run this if you're getting missing file errors during training:

```bash
python ai/utils/validateImagePaths.py
```

This will report any image paths in the dataset CSVs that cannot be found on disk.

---

## Notes

- Always run `validateImagePaths.py` after moving or reorganizing image directories
- `splitData_NF.py` is specifically configured for the Pix2Struct training pipeline — use `splitData_NEW.py` for BLIP
