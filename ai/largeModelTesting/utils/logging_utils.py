'''
Description:
Custom logging utility designed for NF's dedicated PC to monitor machine learning 
training runs. This script provides:

1. `setup_logger(name, out)`: Initializes a logger that writes both to a file 
   ('training.log') and the console, with timestamps for easy tracking.
2. `log_metrics(metrics, out)`: Appends training metrics (e.g., loss, accuracy) 
   to a JSONL file ('metrics.jsonl') for structured storage and later analysis.

The script ensures that all logs and metrics are consistently formatted, stored, 
and easily accessible for monitoring and debugging training processes.
'''

import logging, os, json

def setup_logger(name, out):
    os.makedirs(out, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(os.path.join(out, "training.log"))
    sh = logging.StreamHandler()
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger

def log_metrics(metrics, out):
    with open(os.path.join(out, "metrics.jsonl"), "a") as f:
        f.write(json.dumps(metrics) + "\n")