'''Description:
Lightweight utility function to wrap any iterable with a tqdm progress bar. 
Features:

1. Customizable description via `desc` parameter.
2. Sets a fixed console width (ncols=120) with dynamic resizing enabled.
3. Can be used throughout preprocessing or training scripts to provide
   consistent, readable progress feedback.

Usage Example:
    for item in progress(my_iterable, desc="Processing"):
        ...
'''

from tqdm import tqdm
def progress(it, desc):
    return tqdm(it, desc=desc, ncols=120, dynamic_ncols=True)