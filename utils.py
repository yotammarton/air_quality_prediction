import numpy as np


def smape(A, F):
    """
    SMAPE measure
    :param A: (np.ndarray) Actual values
    :param F: (np.ndarray) Forecasted values
    :return: SMAPE score (in % units)
    """
    return 100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


def split_array_to_size(arr, size, drop_last):
    """
    split an array to mini-arrays of the same size `size`
    :param arr: (np.array) input array to split
    :param size: (int) the size of mini-arrays
    :param drop_last: (bool) if True and last mini-array is not with len()==size,
                        drops the last mini-array
    :return: (list) of mini-arrays
    """
    split_positions = list(range(size, len(arr), size))
    split_result = np.array_split(arr, split_positions)
    if drop_last and len(split_result[-1]) != size:
        return split_result[:-1]
    return split_result
