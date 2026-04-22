# ai/annotations/

This folder contains the annotated CSV datasets used for training and evaluating the Reading4All alt-text generation model.

---

## Files

| File | Description |
|---|---|
| `annotated_physics_data(Sheet1).csv` | Manually annotated physics diagram data with corrected symbols, equations, and units |
| `combinedData.csv` | Merged dataset combining multiple annotation sources — primary training input |
| `extracted_images_textbook2.csv` | Annotations for a second batch of images extracted from textbooks |
| `textbook_extracted_images.csv` | Initial set of textbook-extracted image annotations |

---

## Format

Each CSV row represents one diagram image and contains fields for the image path and its corresponding human-written alt-text annotation. These are consumed directly by the training scripts in `ai/train/`.

---

## Notes

- `combinedData.csv` is the final merged dataset and is the one referenced by training scripts
- Individual source CSVs are kept for traceability and reproducibility
- Annotation fixes (symbols, equations, units) were applied to the physics dataset to improve model accuracy on scientific diagrams
