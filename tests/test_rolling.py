import numpy as np
import pytest

from src.rolling import strided_indexing_roll_2d, strided_indexing_roll_3d


def test_2d_rolling():
    array = np.array([
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 1],
    ])
    roll_values = np.array([1, 2, 3, 4, 5])

    expected = np.array([
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
    ])

    rolled = strided_indexing_roll_2d(
        array=array,
        roll_values=roll_values
    )
    assert pytest.approx(rolled) == expected


def test_3d_rolling():
    array = np.array([
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 1],
    ])
    array = np.expand_dims(array, axis=-1)
    roll_values = np.array([1, 2, 3, 4, 5])

    expected = np.array([
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
    ])
    expected = np.expand_dims(expected, axis=-1)

    rolled = strided_indexing_roll_3d(
        array=array,
        roll_values=roll_values
    )
    assert pytest.approx(rolled) == expected
