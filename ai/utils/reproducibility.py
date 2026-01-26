'''
Description:
Utility functions for reproducibility and configuration management in machine learning workflows.

Key Features:
1. `set_seed(seed)`:
   - Sets seeds for Python `random`, NumPy, and PyTorch (CPU and all available GPUs) to ensure deterministic behavior.
   - Configures PyTorch's cuDNN backend for reproducible results (deterministic operations, disables benchmarking).

2. `save_config(cfg, out)`:
   - Saves a Python dictionary of configuration parameters to a JSON file (`config.json`) in the specified output directory.
   - Creates the output directory if it does not exist.

These utilities help maintain reproducibility across experiments and preserve experiment configurations for future reference.
'''

import random, json, os, numpy as np, torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_config(cfg, out):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)