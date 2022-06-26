import numpy as np
import mrcfile as mrc
import os
import h5py
from os.path import join
from loguru import logger
from sklearn.model_selection import train_test_split
from data_prep_utils import *


SEED = 10101
TRAIN_SIZE = 0.7
MODEL_INPUT_SHAPE = (16, 16, 16)
MAP_FILES_PATH = "../data/raw_inputs"
RESMAP_TARGETS_PATH = "../data/resmap_targets"
PROCESSED_FILES_PATH = "../data/resmap_processed_dataset"


def process_files_resmap(density_map_files, target_files):
    # TODO: create the similar thing for monores when it will be repaired.
    non_zero_cube_count = 0
    cube_count = 0
    all_target_cubes = []
    all_density_map_cubes = []

    for map_file, target_file in zip(density_map_files, target_files):
        density_map = mrc.open(join(MAP_FILES_PATH, map_file)).data
        target = mrc.open(join(RESMAP_TARGETS_PATH, target_file)).data

        density_map = normalize_map(density_map)
        target, density_map = pad(target, density_map)
        target_cubes = cubify(target, MODEL_INPUT_SHAPE)
        density_map_cubes = cubify(density_map, MODEL_INPUT_SHAPE)

        if target_cubes.shape != density_map_cubes.shape:
            raise ValueError("Unequal target and density map shapes")

        cube_count += target_cubes.shape[0]
        target_cubes, density_map_cubes = remove_empty_cubes(
            target_cubes, density_map_cubes
        )
        all_target_cubes.append(target_cubes)
        all_density_map_cubes.append(density_map_cubes)

        non_zero_cube_count += target_cubes.shape[0]

    logger.info(f"Found non-zero cubes {non_zero_cube_count} from {cube_count}")
    all_target_cubes = np.concatenate(all_target_cubes)
    all_density_map_cubes = np.concatenate(all_density_map_cubes)

    return all_target_cubes, all_density_map_cubes


def save_dataset(file_path, data_dict):
    h = h5py.File(file_path, mode="w")
    for k, data in data_dict.items():
        h.create_dataset(k, data=data)
    h.close()


def process_data(density_map_path, target_path):
    density_map_files = sorted(os.listdir(density_map_path))
    target_files = sorted(os.listdir(target_path))

    logger.info("Split files on train, valid and test sets...")
    (
        density_map_train,
        density_map_valid,
        targets_train,
        targets_valid,
    ) = train_test_split(
        density_map_files, target_files, train_size=TRAIN_SIZE, random_state=SEED
    )
    density_map_valid, density_map_test, targets_valid, targets_test = train_test_split(
        density_map_valid, targets_valid, train_size=0.5, random_state=SEED
    )

    logger.info("Processing train data...")
    train_target_cubes, train_density_map_cubes = process_files_resmap(
        density_map_train, targets_train
    )
    logger.info("Processing valid data...")
    valid_target_cubes, valid_density_map_cubes = process_files_resmap(
        density_map_valid, targets_valid
    )
    logger.info("Processing test data...")
    test_target_cubes, test_density_map_cubes = process_files_resmap(
        density_map_test, targets_test
    )

    # Create hdf5 files for storing maps and targets
    train_dataset = {"map": train_density_map_cubes, "target": train_target_cubes}
    if train_density_map_cubes.shape[0] != train_target_cubes.shape[0]:
        raise ValueError("Unequal train label and map counts")

    logger.info(f"Saving {train_target_cubes.shape[0]} train samples")
    save_dataset(join(PROCESSED_FILES_PATH, "train_repository.hdf5"), train_dataset)

    valid_dataset = {"map": valid_density_map_cubes, "target": valid_target_cubes}
    if valid_density_map_cubes.shape[0] != valid_target_cubes.shape[0]:
        raise ValueError("Unequal validation label and map counts")

    logger.info(f"Saving {valid_target_cubes.shape[0]} valid samples")
    save_dataset(join(PROCESSED_FILES_PATH, "valid_repository.hdf5"), valid_dataset)

    test_dataset = {"map": test_density_map_cubes, "target": test_target_cubes}
    if test_density_map_cubes.shape[0] != test_target_cubes.shape[0]:
        raise ValueError("Unequal test label and map counts")

    logger.info(f"Saving {test_target_cubes.shape[0]} test samples")
    save_dataset(join(PROCESSED_FILES_PATH, "test_repository.hdf5"), test_dataset)


def main():
    logger.info("Running process_data.py...")
    logger.info("Process Resmap data...")

    process_data(MAP_FILES_PATH, RESMAP_TARGETS_PATH)


if __name__ == "__main__":
    main()
