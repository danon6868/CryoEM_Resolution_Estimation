import time
import os
from os.path import join
import numpy as np
import tqdm


# This code was used to run Resmap for all `.mrc` files in `mrc_for_chosen_pdb` directory

pdb_files_path = "data/pdb_files"
mrc_files_path = "data/mrc_for_pdb_files"
processed_files = "data/targets"

if not os.path.isdir(processed_files):
    os.mkdir(processed_files)

for file in tqdm.tqdm(os.listdir(pdb_files_path)):
    if file[:-3] + "mrc" not in os.listdir(processed_files):
        try:
            expr = (
                f"ResMap-1.1.4-linux64 --noguiSingle {mrc_files_path}/{file[:-3]+'mrc'}"
            )
            os.system(expr)
            time.sleep(2)
        except:
            continue
