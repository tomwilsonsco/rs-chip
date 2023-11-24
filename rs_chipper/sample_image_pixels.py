from pathlib import Path
import numpy as np
import rasterio
import pickle


def sample_image_pixels(input_image_path, sample_size=10000):
    """
    Samples pixel values from an image at random coordinates and calculates the
    mean and standard deviation for each band.

    Args:
        input_image_path (str or Path): Path to the input image file.
        sample_size (int, optional): The number of pixel coordinates to sample
        from the image. Defaults to 10000.

    Returns:
        dict: A dictionary with keys as '<band_name>_mean' and '<band_name>_std'
        for each band in the image, and values as the calculated mean and
        standard deviation of the sampled pixel values.
    """
    input_image_path = Path(input_image_path)
    stats_dict = {}

    with rasterio.open(input_image_path) as src:
        # Get bounds of the image
        left, bottom, right, top = src.bounds
        # Generate random coordinates within the image bounds
        xs = np.random.uniform(left, right, sample_size)
        ys = np.random.uniform(bottom, top, sample_size)

        # Prepare a list of coordinates
        coordinates = list(zip(xs, ys))

        # Read the pixel values at the coordinates for all bands
        pixel_values = np.array(list(src.sample(coordinates)))

        # Get band descriptions
        band_names = src.descriptions

        # Calculate mean and standard deviation for each band
        for band_index in range(src.count):
            # Use band description if available, otherwise default to 'Band_<index>'
            band_name = (
                band_names[band_index]
                if band_names[band_index]
                else f"Band_{band_index + 1}"
            )
            band_pixel_values = pixel_values[:, band_index]
            valid_band_pixel_values = band_pixel_values[~np.isnan(band_pixel_values)]
            stats_dict[f"{band_name}_mean"] = np.mean(valid_band_pixel_values)
            stats_dict[f"{band_name}_std"] = np.std(valid_band_pixel_values)

    # Create the pickle file path
    pickle_file_name = f"{input_image_path.stem}_{sample_size}.pkl"
    pickle_file_path = input_image_path.with_name(pickle_file_name)

    # Save the dictionary to a pickle file
    with open(pickle_file_path, "wb") as f:
        pickle.dump(stats_dict, f)

    return stats_dict
