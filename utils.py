import os
import numpy as np


def get_last_folder_id(folder_path):
    t = 0
    for fn in os.listdir(folder_path):
        t = max(t, int(fn))
    return t


def movingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma


def draw_equispaced_items_from_sequence(m, n):
    """
    draw_equispaced_items_from_sequence(m, n)
    Args:
        m (int): How many items to draw.
        n (int): Length of sequence to draw from.
    """
    return [i * n // m + n // (2 * m) for i in range(m)]
