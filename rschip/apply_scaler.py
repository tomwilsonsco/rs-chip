import numpy as np


def apply_scaler(array, scaler_dict):
    """Standard scales a numpy array based on mean and std values from a dictionary.

    Dict can be created from `standard_scale_array`.

    Args:
        array (np.ndarray): A numpy array of shape (m, n, n), where m is the length of the first dimension.
        info_dict (dict): A dictionary containing 'mean' and 'std' for each band, keyed by the first dimension index.

    Returns:
        np.ndarray: A standard scaled numpy array of the same shape as the input array.
    """
    # Validate the number of bands against the number of band names provided
    if array.shape[0] != len(scaler_dict.keys()):
        raise ValueError(
            "The number of bands in scaler dict does not match the number of bands in the image array."
        )
    scaled_array = np.empty_like(array, dtype=np.float32)
    for i in range(array.shape[0]):
        band_info = scaler_dict.get(i, {"mean": 0, "std": 1})
        mean = band_info["mean"]
        std = band_info["std"]
        scaled_array[i, :, :] = (array[i, :, :] - mean) / std
    return scaled_array


def unapply_scaler(scaled_array, scaler_dict):
    """Performs the inverse of standard scaling on a numpy array.

    Note that the dtype of the array is not changed, so if for example converting back to
    a int16 type then this would have to be done separately.

    Args:
        scaled_array (np.ndarray): A standard scaled numpy array of shape (m, n, n), where m is the length of the first dimension.
        info_dict (dict): A dictionary containing 'mean' and 'std' for each band, keyed by the first dimension index.

    Returns:
        np.ndarray: A numpy array transformed back to its original scale, of the same shape as the input scaled array.
    """
    if scaled_array.shape[0] != len(scaler_dict.keys()):
        raise ValueError(
            "The number of bands in scaler dict does not match the number of bands in the image array."
        )
    unscaled_array = np.empty_like(scaled_array)
    for i in range(scaled_array.shape[0]):
        band_info = scaler_dict.get(i, {"mean": 0, "std": 1})
        mean = band_info["mean"]
        std = band_info["std"]
        unscaled_array[i, :, :] = scaled_array[i, :, :] * std + mean
    return unscaled_array
