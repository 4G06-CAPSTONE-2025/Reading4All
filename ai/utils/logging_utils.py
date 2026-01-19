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