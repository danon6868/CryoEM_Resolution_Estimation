import numpy as np


def normalize_map(density_map: np.array) -> np.array:
    """Computes feature min-max normalization.

    Args:
        target (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    min = density_map.min()
    difference = density_map.max() - min
    density_map = ((density_map - min) / difference)

    return density_map


def nearest_power_two(val):
    """Returns the first power of two greater than or equal to value val if val is positive.

    Args:
        val (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    if val <= 0:
        raise ValueError("val must be positive")

    power = 1
    while power < val:
        power *= 2

    return power

        
def pad_to_power_2_cube(arr, remainder_decisions, 
                        constant_value=0.):
    """Return a power of two cube padded with `constant_value` for any additional 
    elements appended. Use remainder_decisions to add the ith dimension to 
    before if true or after if false for the respective dimension.

    Args:
        arr (_type_): _description_
        remainder_decisions (_type_): _description_
        constant_value (_type_, optional): _description_. Defaults to 0..

    Returns:
        _type_: _description_
    """
    
    cube_edge_length = nearest_power_two(max(arr.shape))
    dim_deltas = cube_edge_length - np.array(arr.shape)
    padding = []

    for count, dim_delta in enumerate(dim_deltas):
        before = dim_delta // 2 
        after = before
        if dim_delta % 2:  # odd -> append the remainder according to remainder_decisions
            if remainder_decisions[count]:
                before += 1
            else:
                after += 1
        tup = (before, after)
        padding.append(tup)

    return np.pad(arr, tuple(padding), mode="constant", constant_values=constant_value)


def pad(label, density_map, constant_value=0.):
    """Pad label and map. Resolve odd dimensions with a random array of
    remainder choices.

    Args:
        label (_type_): _description_
        density_map (_type_): _description_
        constant_value (_type_, optional): _description_. Defaults to 0.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    if label.shape != density_map.shape:
        raise ValueError("invalid shape")

    remainder_decisions = np.random.choice([False, True], label.ndim)

    return (pad_to_power_2_cube(label.data, remainder_decisions, constant_value=100.), 
                pad_to_power_2_cube(density_map.data, remainder_decisions))


def cubify(arr: np.array, newshape: tuple) -> np.array:
    
    """_summary_

    Args:
        arr (_type_): _description_

    Returns:
        _type_: _description_
    """

    oldshape = np.array(arr.shape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.column_stack([repeats, newshape]).ravel()
    order = np.arange(len(tmpshape))
    order = np.concatenate([order[::2], order[1::2]])

    # newshape must divide oldshape evenly or else ValueError will be raised
    return arr.reshape(tmpshape).transpose(order).reshape(-1, *newshape)


def remove_empty_cubes(label_cubes: np.array, map_cubes: np.array) -> np.array:
    """Remove map cubes containing no non-zero values and the corresponding 
    label cubes.

    Args:
        label_cubes (_type_): _description_
        map_cubes (_type_): _description_

    Returns:
        _type_: _description_
    """

    empty_cubes = []
    for cube_i in range(map_cubes.shape[0]):
        if not np.count_nonzero(map_cubes[cube_i, ...]):
            empty_cubes.append(cube_i)

    map_cubes = np.delete(map_cubes, empty_cubes, 0)
    label_cubes = np.delete(label_cubes, empty_cubes, 0)

    return label_cubes, map_cubes
