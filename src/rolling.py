import numpy as np
from skimage.util.shape import view_as_windows


def strided_indexing_roll_2d(
    array: np.ndarray,
    roll_values: np.ndarray
) -> np.ndarray:
    """Perform an efficient rolling operation on a 2D array.

    Each row is rolled (shifted) separately by the corresponding value
    in roll values.
    Positive roll values shift data to the 'left', to lower indices.

    Args:
        array (np.ndarray): 2D array to roll.
        roll_values (np.ndarray): By how much to shift each row.

    Returns:
        np.ndarray: Rolled array.

    Raises:
        ValueError: Array and doll_values don't have the same length.
    """
    if len(array) != len(roll_values):
        raise ValueError("Array and roll values should have same length.")

    # reverse roll direction for compatibility with slicing
    roll_values = -roll_values

    # concatenate with self to handle wrap-around
    a_ext = np.concatenate((array, array[:, :-1]), axis=1)

    # generate a strided view of the extended array
    n_columns = array.shape[1]
    strided_view = view_as_windows(a_ext, (1, n_columns))

    # select the appropriate rolled windows for each row
    rolled = strided_view[
        np.arange(len(roll_values)),            # row indices
        (n_columns-roll_values) % n_columns,    # column offsets for rolling
        0                                       # squeeze
    ]
    return rolled


def strided_indexing_roll_3d(
        array: np.ndarray,
        roll_values: np.ndarray
) -> np.ndarray:
    """Perform an efficient rolling operation on a 3D array.

    Each row is rolled (shifted) separately by the corresponding roll value in
    roll values. Rolling is performed along the second axis.

    Args:
        array (np.ndarray): 3D array to roll.
        roll_values (np.ndarray): By how much to shift each row.

    Returns:
        np.ndarray: Rolled array.

    Raises:
        ValueError: Array and doll_values don't have the same length.
    """
    if len(array) != len(roll_values):
        raise ValueError("Array and roll values should have same length.")

    # reverse roll direction for compatibility with slicing
    roll_values = -roll_values

    # concatenate with self to handle wrap-around
    # done along second axis
    a_ext = np.concatenate((array, array[:, :-1, :]), axis=1)

    # shape
    n_columns = array.shape[1]
    depth = array.shape[2]

    # normalize roll values to ensure they are within bounds
    roll_values %= n_columns

    # combine the column and depth axes for sliding windows
    sliding_windows = view_as_windows(a_ext, (1, n_columns, depth))

    # select the appropriate rolled windows for each row
    rolled = sliding_windows[
        np.arange(len(roll_values)),                # row indices
        (n_columns - roll_values) % n_columns,      # column offsets
        0,                                          # squeeze
        0,                                          # squeeze
    ]

    return rolled
