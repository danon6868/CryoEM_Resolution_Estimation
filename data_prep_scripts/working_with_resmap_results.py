import mrcfile
import numpy as np


with mrcfile.open("../data/mrc_for_chosen_pdb/4KZV.mrc") as mrc:
    print(mrc.data.max())

with mrcfile.open("../data/mrc_for_chosen_pdb/4KZV_resmap.mrc") as mrc:
    print(np.unique(mrc.data))
