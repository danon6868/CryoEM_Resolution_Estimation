from typing import Tuple
import os
from loguru import logger
import mrcfile as mrc
import numpy as np
import torch
from torch.utils.data import DataLoader

from resolution_estimation_with_dl.model_training.model import UNet3D
from resolution_estimation_with_dl.data_preparation.data_prep_utils import (
    normalize_map,
    pad,
)
from resolution_estimation_with_dl.model_training.training_config import TrainParameters


def read_mrc(mrc_file_path: str) -> np.array:
    """Reads mrc file and converts data into `np.array`.

    Args:
        mrc_file_path (str): Path to mrc file.

    Returns:
        np.array: 3D object which is represented by voxels with electron density values.
    """

    density_file = mrc.open(mrc_file_path)
    density_map = density_file.data
    density_file.close()
    return density_map


def save_mrc(density_map: np.array, mrc_file_path: np.array) -> None:
    """Save mrc file in a given directory.

    Args:
        density_map (np.array): Electron density or local resolution map.
        mrc_file_path (np.array): Path where mrc file will be saved.
    """
    if not os.path.exists(mrc_file_path):
        mrc_file = mrc.mmap(mrc_file_path, mode="w+")
        mrc_file.set_data(density_map)
        mrc_file.close()
    else:
        logger.warning(
            f"{mrc_file_path} is already exists. Do you want to overwrite it?"
        )
        acception = ""
        while acception not in ("y", "n"):
            acception = input("Do you want to overwrite this file? y/n:")

        if acception == "y":
            os.remove(mrc_file_path)
            mrc_file = mrc.mmap(mrc_file_path, mode="w+")
            mrc_file.set_data(density_map)
            mrc_file.close()
        else:
            logger.warning(
                f"{mrc_file_path} was not saved because a file with that name already exists."
            )


def cubify_input(density_map: np.array, newshape: tuple) -> Tuple[np.array, np.array]:
    """Divide electron density map into small cubes with a gived shape.

    Args:
        density_map (np.array): Electron density or local resolution map.
        newshape (tuple): The shape of the small cubes into which `density_map` will be divided.

    Returns:
        Tuple[np.array, np.array]: Divided cubes and additional array to reconstruct whole map from cubes.
    """

    oldshape = np.array(density_map.shape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.column_stack([repeats, newshape]).ravel()
    order = np.arange(len(tmpshape))
    order = np.concatenate([order[::2], order[1::2]])
    cubes = density_map.reshape(tmpshape).transpose(order).reshape(-1, *newshape)

    return cubes, repeats


def decubify_output(
    output_cubes: np.array, target_shape: tuple, repeats: np.array
) -> np.array:
    """Reconstruct entire map from cubes.

    Args:
        output_cubes (np.array): Cubes from which entire map will be reconstructed.
        target_shape (tuple): The shape of the final map.
        repeats (np.array): Additional array to reconstruct whole map from cubes.

    Returns:
        np.array: Entire electron density and local resolution map.
    """

    output_map = (
        output_cubes.reshape(*repeats, *output_cubes.shape[1:])
        .transpose(0, 3, 1, 4, 2, 5)
        .reshape(*target_shape)
    )

    return output_map


def run_model(
    mrc_file_path: str,
    device: str,
    model: UNet3D,
    state_dict_path: str,
    batch_size: int = TrainParameters.batch_size,
    noise_threshold: float = 0.0,
    constant: float = 100.0,
) -> Tuple[np.array, np.array]:
    """Predicts local resolution map based on electron density map using pretrained weights.

    Args:
        mrc_file_path (str): Path to electron density map for which you want to estimate local resolution map.
        device (str): Which device (`cpu` or `cuda`) will be used for prediction. Usually `cuda` is much faster.
        model (UNet3D): UNet3D model.
        state_dict_path (str): Path to model weights.
        batch_size (int, optional): Batch size that will be used during prediction. Defaults to TrainParameters.batch_size.
        noise_threshold (float, optional): If value in electron density map will be less or equal to `noise_threshold`
        then corresponding value in local resolution map will be set a constant. Defaults to 0.0.
        constant (float, optional): Constant for `noise_threshold`. Defaults to 100.0.

    Returns:
        Tuple[np.array, np.array]: Padded electron density map and corresponding local resolution map.
    """

    density_map = read_mrc(mrc_file_path)
    normalized_map = normalize_map(density_map.copy())
    _, normalized_map = pad(normalized_map, normalized_map)
    _, density_map = pad(density_map, density_map)

    normalized_map_cubes, repeats = cubify_input(
        normalized_map, TrainParameters.model_input_shape
    )

    own_loader = DataLoader(normalized_map_cubes, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    model.load_state_dict(torch.load(state_dict_path))

    outputs = []
    model.eval()
    with torch.no_grad():
        for cube in own_loader:
            cube = cube.to(device)
            out = model(cube[:, None, :, :, :])

            outputs.append(out.detach().cpu().numpy())

            cube = cube.to("cpu")
            out = out.to("cpu")
            del cube, out

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    outputs = np.concatenate(outputs, axis=0)
    outputs = outputs.reshape(-1, *TrainParameters.model_input_shape)
    output_map = decubify_output(outputs, normalized_map.shape, repeats)
    output_map[(density_map <= noise_threshold)] = constant

    return density_map, output_map
