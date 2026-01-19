from tqdm import tqdm
def progress(it, desc):
    return tqdm(it, desc=desc, ncols=120, dynamic_ncols=True)