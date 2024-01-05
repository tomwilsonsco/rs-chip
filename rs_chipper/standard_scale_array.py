import numpy as np
from .load_scaler import load_scaler


def standard_scale_array(image_array, scaler_source, band_names):
    """
    Standard scales each band of the input image array using the statistics
    provided either in a pickle file or a scaler_dict dictionary, and a list of band names.

    Args:
        image_array (numpy.array): The image data array to be scaled, with shape
                                   (bands, height, width).
        scaler_source (str or dict): The source of the scaling parameters. It can
                                     be a path to a pickle file or a dictionary
                                     with statistics.
        band_names (list): A list of names corresponding to the bands in the image array. The names
                           must be in the same order as they are found in the image_array.

    Returns:
        numpy.array: The image array with each band scaled to have mean 0 and
                     standard deviation 1.

    Raises:
        ValueError: If scaler_source is not a dictionary or a valid path to a pickle file.
        ValueError: If the number of provided band names does not match the number of bands in the image array.
        KeyError: If the mean or standard deviation for any band name is not found in the scaler dictionary.
    """

    # Load the scaling parameters
    scaler_dict = load_scaler(scaler_source)

    # Validate the number of bands against the number of band names provided
    if image_array.shape[0] != len(scaler_dict.keys()):
        raise ValueError(
            "The number of bands in scaler dict does not match the number of bands in the image array."
        )

    # Apply standard scaling to each band
    scaled_image_array = np.empty_like(image_array, dtype=np.float32)
    for band_index in range(0, image_array.shape[0]):
        band_vals = scaler_dict[band_index]
        if "mean" in band_vals and "std" in band_vals:
            mean = band_vals["mean"]
            std = band_vals["std"]
            # Perform standard scaling on the band
            scaled_image_array[band_index] = (image_array[band_index] - mean) / std
        else:
            raise KeyError(
                f"Mean or standard deviation for band index {band_index} not found in scaler dictionary."
            )

    return scaled_image_array
