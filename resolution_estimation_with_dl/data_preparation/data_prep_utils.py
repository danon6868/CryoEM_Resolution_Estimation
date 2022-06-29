from typing import List, Tuple
import numpy as np


def normalize_map(density_map: np.array) -> np.array:
    """Computes feature min-max normalization.

    Args:
        density_map (np.array): Electron density map.

    Returns:
        np.array: Normalized election density map.
    """

    min = density_map.min()
    difference = density_map.max() - min
    density_map = (density_map - min) / difference

    return density_map


def nearest_power_two(val: int) -> int:
    """Returns the first power of two greater than or equal to value val if val is positive.

    Args:
        val (int): Here it is one of the dimention of electron density map.

    Returns:
        int: The first power of two greater than or equal to value `val`.
    """

    if val <= 0:
        raise ValueError("val must be positive")

    power = 1
    while power < val:
        power *= 2

    return power


def pad_to_power_2_cube(
    arr: np.array, remainder_decisions: List[bool], constant_value: float = 0.0
) -> np.array:
    """Return a power of two cube padded with `constant_value` for any additional
    elements appended. Use remainder_decisions to add the ith dimension to
    before if true or after if false for the respective dimension.

    Args:
        arr (np.array): Electron density or local resolution map.
        remainder_decisions (List[bool]): Use remainder_decisions to add the ith dimension to
        before if true or after if false for the respective dimension.
        constant_value (float, optional): Value for padding. Defaults to 0..

    Returns:
        np.array: Padded electron density or local resolution map.
    """

    cube_edge_length = nearest_power_two(max(arr.shape))
    dim_deltas = cube_edge_length - np.array(arr.shape)
    padding = []

    for count, dim_delta in enumerate(dim_deltas):
        before = dim_delta // 2
        after = before
        if (
            dim_delta % 2
        ):  # odd -> append the remainder according to remainder_decisions
            if remainder_decisions[count]:
                before += 1
            else:
                after += 1
        tup = (before, after)
        padding.append(tup)

    return np.pad(arr, tuple(padding), mode="constant", constant_values=constant_value)


def pad(
    label: np.array, density_map: np.array, constant_value=0.0
) -> Tuple[np.array, np.array]:
    """Pad label and map. Resolve odd dimensions with a random array of
    remainder choices.

    Args:
        label (np.array): Electron density map.
        density_map (np.array): Local resolution map.
        constant_value (float, optional): Constant value for padding maps. Defaults to 0.

    Returns:
        Tuple[np.array, np.array]: Padded electron density and local resolution maps.
    """

    if label.shape != density_map.shape:
        raise ValueError("invalid shape")

    remainder_decisions = np.random.choice([False, True], label.ndim)

    return (
        pad_to_power_2_cube(label.data, remainder_decisions, constant_value=100.0),
        pad_to_power_2_cube(density_map.data, remainder_decisions),
    )


def cubify(arr: np.array, newshape: tuple) -> np.array:

    """Divides input array into cudes of a given size.

    Args:
        arr (np.array): Electron density or local resolution map.

    Returns:
        np.array: Array with a cubes of a given shape.
    """

    oldshape = np.array(arr.shape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.column_stack([repeats, newshape]).ravel()
    order = np.arange(len(tmpshape))
    order = np.concatenate([order[::2], order[1::2]])

    # newshape must divide oldshape evenly or else ValueError will be raised
    return arr.reshape(tmpshape).transpose(order).reshape(-1, *newshape)


def remove_empty_cubes(
    label_cubes: np.array, map_cubes: np.array
) -> Tuple[np.array, np.array]:
    """Remove map cubes containing no non-zero values and the corresponding
    label cubes.

    Args:
        label_cubes (np.array): Cubes with local resolution values.
        map_cubes (np.array): Cubes with electron density values.

    Returns:
        Tuple[np.array, np.array]: Local resolution and electron density cubes after removing empty ones.
    """

    empty_cubes = []
    for cube_i in range(map_cubes.shape[0]):
        if not np.count_nonzero(map_cubes[cube_i, ...]):
            empty_cubes.append(cube_i)

    map_cubes = np.delete(map_cubes, empty_cubes, 0)
    label_cubes = np.delete(label_cubes, empty_cubes, 0)

    return label_cubes, map_cubes
