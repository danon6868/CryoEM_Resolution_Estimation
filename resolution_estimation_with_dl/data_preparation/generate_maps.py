import os
from os.path import join
import numpy as np


# This code was used to generate electron density maps from raw `.pdb` files
# using `pdb2mrc` function from `EMAN2` package

box_size = 64
sampling_rate = 1
pdb_files_path = "data/pdb_files"
mrc_files_path = "data/mrc_for_pdb_files"
res_vals = np.arange(2.0, 12.1, 0.1)

if not os.path.exists(mrc_files_path):
    os.mkdir(mrc_files_path)

for i, file in enumerate(os.listdir(pdb_files_path)):
    if (i + 1) % 100 == 0:
        print(f"Proccesed {i+1} pdb files...")
    res = round(np.random.choice(res_vals), 1)
    expr = f"e2pdb2mrc.py {join(pdb_files_path, file)} {join(mrc_files_path, file[:-3]+'mrc')} --apix={1} --res={res}"
    os.system(expr)
