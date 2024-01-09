from pathlib import Path
import rasterio as rio
from rasterio.windows import Window
import numpy as np
from tqdm import tqdm
from .load_scaler import load_scaler
from .sample_image_pixels import sample_image_pixels
from .apply_scaler import apply_scaler


def _save_batch_as_npz(output_path, batch, batch_index):
    """
    Save a batch of chips as an NPZ file.

    Args:
        output_path (Path): Output directory path.
        file_name_stem (str): Base name for the output file.
        batch (list): List of image chips (np.ndarray).
        batch_index (int): Index of the batch.
    """
    np.savez_compressed(output_path / f"batch_{batch_index}.npz", **batch)


def chip_image_to_npz(
    input_image_path,
    output_path,
    output_name=None,
    pixel_dimensions=128,
    offset=64,
    standard_scale=True,
    sample_size=10000,
    scaler_source=None,
    batch_size=1000,
):
    """
    Split a satellite image into smaller tiles or chips.

    Args:
        input_image_path (str): The path to the input satellite image.
        output_path (str): The directory path where the output .npz files will be saved.
        output_name (str, optional): The stem name of each array in npz dict,
        if not specified it will be the input image name.
        pixel_dimensions (int, optional): The height and width of each tile in pixels. Defaults to 128.
        offset (int, optional): The offset used when creating tiles, to define the step size. Defaults to 64.
        standard_scale (bool, optional): Whether to standard scale from a sample of pixel values. Defaults to True.
        sample_size (int, optional): If standard_scale is True, how many pixels are sampled
        to derive mean and standard deviation for scaling.
        scaler_source (str or dict): Optional scaling parameters returned from `sample_image_pixels`. Path
        to a pickle file or a dictionary with statistics. If not specified these will be derived (default)
        if standard_scale is True.
        batch_size (int, optional): Number of image tiles per NPZ batch. Defaults to 100.

    Returns:
        None

    This function uses rasterio to read a satellite image, then splits the image into
    smaller square tiles of specified dimensions. Optionally the chip pixel values can be standard
    scaled before saving, using a sample of the full image pixels.
    """
    print(f"Chipping {input_image_path} to /{output_path}...")

    # Convert string paths to Path objects
    input_image_path = Path(input_image_path)
    output_path = Path(output_path)

    # Create output directory if it does not exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Create or load standard scaler dictionary if required
    if standard_scale:
        if scaler_source is None:
            scaler_dict = sample_image_pixels(input_image_path, sample_size)
        else:
            scaler_dict = load_scaler(scaler_source)

    # Calculate the range for x and y to create chips
    with rio.open(input_image_path) as src:
        batch = {}
        batch_index = 0
        for y in tqdm(range(0, src.height, offset), desc="Chipping image..."):
            for x in range(0, src.width, offset):
                window = Window(x, y, pixel_dimensions, pixel_dimensions)

                # Initialize an array for the chip data
                chip_data = np.zeros(
                    (src.count, pixel_dimensions, pixel_dimensions), dtype=src.dtypes[0]
                )
                # Read data into the chip array
                chip = src.read(window=window, out=chip_data, boundless=True)
                if standard_scale:
                    chip = apply_scaler(chip, scaler_dict)

                if output_name is None:
                    arr_name = f"{input_image_path.stem}_{x}_{y}"
                else:
                    arr_name = f"{output_name}_{x}_{y}"

                batch[arr_name] = chip

                # Check if batch is full
                if len(batch) == batch_size:
                    # Save the batch as an NPZ file
                    _save_batch_as_npz(output_path, batch, batch_index)
                    batch = {}  # Start a new batch
                    batch_index += 1

        if batch:
            _save_batch_as_npz(output_path, batch, batch_index)