import time
import os
from os.path import join
import numpy as np
import tqdm


pdb_files_path = "../data/chosen_pdb"
mrc_files_path = "../data/mrc_for_chosen_pdb"
processed_files = "../data/inputs"


for file in tqdm.tqdm(os.listdir(pdb_files_path)):
    if file[:-3]+"mrc" not in os.listdir(processed_files):
        try:
            expr = f"ResMap-1.1.4-linux64 --noguiSingle {mrc_files_path}/{file[:-3]+'mrc'}"
            os.system(expr)
            time.sleep(2)
        except:
            continue
