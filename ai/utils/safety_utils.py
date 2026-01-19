import torch, time, psutil

def log_health(logger):
    ram = psutil.virtual_memory().percent
    logger.info(f"RAM {ram:.1f}% | GPU {torch.cuda.memory_allocated()/1e9:.2f} GB")

def cooldown(sec=0.05):
    time.sleep(sec)