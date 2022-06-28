import mrcfile as mrc
import numpy as np
import torch
from torch.utils.data import DataLoader
from ..data_preparation.data_prep_utils import normalize_map, pad
from ..model_training.training_config import TrainParameters


def read_mrc(mrc_file_path):
    density_file = mrc.open(mrc_file_path)
    density_map = density_file.data
    density_file.close()
    return density_map


def save_mrc(density_map, mrc_file_path):
    mrc_file = mrc.mmap(mrc_file_path, mode="w+")
    mrc_file.set_data(density_map)
    mrc_file.close()


def cubify_input(density_map: np.array, newshape: tuple):
    oldshape = np.array(density_map.shape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.column_stack([repeats, newshape]).ravel()
    order = np.arange(len(tmpshape))
    order = np.concatenate([order[::2], order[1::2]])
    cubes = density_map.reshape(tmpshape).transpose(order).reshape(-1, *newshape)

    return cubes, repeats


def decubify_output(output_cubes: np.array, target_shape: tuple, repeats: np.array):
    output_map = (
        output_cubes.reshape(*repeats, *output_cubes.shape[1:])
        .transpose(0, 3, 1, 4, 2, 5)
        .reshape(*target_shape)
    )

    return output_map


def run_model(
    mrc_file_path,
    device,
    model,
    state_dict_path,
    batch_size=TrainParameters.batch_size,
    noise_threshold=0.0,
    constant=100.0,
):
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
