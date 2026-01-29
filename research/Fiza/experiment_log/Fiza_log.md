# Experiment Log (Training)

### [2026-01-21 11:00PM] <Results from Train for tester 1>

**Hypothesis:**  
- After trying different arguments, training with no prompt and with a prompt I decided to look into the training data as when i have trained models in the past during my internship errors like "???" that arent detectable, something is wrong with the training data. I foundout that the training data had alot of "???" for unknown symbols when we converted our excel to csv

**Test:**  
- Evaluated generation with **no prompt** 
- Evaluated generation with **prompted input** in inference code ("Describe this physics diagram:")

**Run:** output_dir=OUT_DIR | per_device_train_bs=1 | per_device_eval_bs=1 | grad_accum=1 | lr=5e-5 | epochs=3 | warmup_ratio=0.05 | eval_strategy=epoch | save_strategy=no | logging_steps=10 | remove_unused_columns=false | fp16=false | num_workers=0 | pin_memory=false

**Outcome:**  
- describe this physics diagram : the illustration illustrates a three - dimensional system consisting of coils, labeled as'v'and'b'and'v'r '???? 1 ', which consists of two distinct
- describe this physics diagram : the illustration illustrates a three - dimensional system consisting of coils, labeled as'v'and'b'and'v'r '????????????????????

**Decision:**  
- fix the annotated data to not have ??? as that disrupts model training

### [2026-01-22 12:30AM] <Post-cleaning check: remove “???” from CSV>

**Hypothesis:**  
- Replacing/removing the `"???"` placeholder tokens in the annotated CSV will reduce noisy targets and improve caption quality because the model won’t learn to emit unknown-symbol garbage during generation.

**Test:**  
- Cleaned annotated CSV by converting to UTF-8   
- Re-ran inference on the trained model using the same evaluation approach as before  
- Compared generations *before vs after* cleaning for presence of `"???"` and “physics-ness” of language

**Run:**  
- dataset_size=100 images | eval_prompt="Describe this physics diagram:" | compared to prior run with `"???"`-contaminated CSV  
- (training args same as previous unless otherwise noted)

**Outcome:**  
- Generations **no longer contained `"???"`** tokens  
- Outputs were **more physics-style / diagram-relevant** (more consistent with expected domain language)  
- This supported the idea that `"???"` in labels was actively harming learning

**Decision:**  
- Keep dataset-cleaning step as **required preprocessing** before any future training  
- Continue training only on cleaned annotations moving forward

### [2026-01-24 2:00PM] <Dataset size sensitivity: 50 vs 100 images>

**Hypothesis:**  
- Smaller datasets (ex: 50 images) will cause weaker generalization and more generic/incorrect captions. Increasing to 100 images should improve stability, but overall performance will still be limited by dataset scale.

**Test:**  
- Trained/evaluated with **~50 images** vs **~100 images** (same general setup)  
- Compared caption quality across a small fixed set of validation examples (qualitative comparison)

**Run:**  
- condition_A: dataset_size≈50 images  
- condition_B: dataset_size≈100 images  
- (other args held constant as much as possible to isolate dataset size effect)

**Outcome:**  
- **50-image condition produced worse results** (more errors / more generic captions)  
- **100-image condition was noticeably better**, but still limited and not “production-level”  
- Trend indicates performance is strongly dataset-size dependent

**Decision:**  
- Treat 100 images as **only a sanity-check baseline**, not a final dataset  
- Prioritize scaling labels rather than over-tuning hyperparameters at this stage

