import os
from os.path import join
import shutil
import numpy as np

data_path = "../data/pdb_files"
chosen_pdb_path = "../data/chosen_pdb"
file_cnt = 0
files_to_stay = []
number_of_files_to_stay = 3000
denum = 1024 * 1024

# Create directory for chosen 10k pdb files
if not os.path.exists(chosen_pdb_path):
    os.mkdir(chosen_pdb_path)

# Choose only not a big files and append to files_to_stay
for file in os.listdir(data_path):
    if os.path.getsize(join(data_path, file)) / denum < 0.8:
        file_cnt += 1
        files_to_stay.append(file)

# Choose 10000k from files_to_stay
files_to_stay = np.random.choice(files_to_stay, size=number_of_files_to_stay)

for file in files_to_stay:
    shutil.copyfile(join(data_path, file), join(chosen_pdb_path, file))


print("END")
