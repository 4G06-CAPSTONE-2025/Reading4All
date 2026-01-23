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

