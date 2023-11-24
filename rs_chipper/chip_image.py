from pathlib import Path
import rasterio as rio
from rasterio.windows import Window
import numpy as np
from tqdm import tqdm

def chip_image(input_image_path, output_path, pixel_dimensions=128, offset=64):
    """
    Split a satellite image into smaller tiles or chips.

    Args:
        input_image_path (str): The path to the input satellite image.
        output_path (str): The directory path where the output tiles will be saved.
        pixel_dimensions (int, optional): The height and width of each tile in pixels. Defaults to 128.
        offset (int, optional): The offset used when creating tiles, to define the step size. Defaults to 64.

    Returns:
        None

    This function uses rasterio to read a satellite image, then splits the image into
    smaller square tiles of specified dimensions and saves them to the output path.
    The output tiles are named using the base name of the input file with appended
    x and y offsets and saved as TIFF files.
    """
    print(f"Chipping {input_image_path} to /{output_path}...")

    # Convert string paths to Path objects
    input_image_path = Path(input_image_path)
    output_path = Path(output_path)

    # Create output directory if it does not exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Calculate the range for x and y to create chips
    with rio.open(input_image_path) as src:
        for y in tqdm(range(0, src.height, offset), desc="Chipping image..."):
            for x in range(0, src.width, offset):
                window = Window(x, y, pixel_dimensions, pixel_dimensions)
                
                transform = src.window_transform(window)
                
                # Initialize an array for the chip data
                chip_data = np.zeros((src.count, pixel_dimensions, pixel_dimensions), dtype=src.dtypes[0])
                # Read data into the chip array
                chip = src.read(window=window, out=chip_data, boundless=True)

                output_file_name = f"{input_image_path.stem}_{x}_{y}.tif"
                output_file_path = output_path / output_file_name

                with rio.open(
                    output_file_path, 'w',
                    driver='GTiff',
                    height=pixel_dimensions,
                    width=pixel_dimensions,
                    count=src.count,
                    dtype=src.dtypes[0],
                    crs=src.crs,
                    transform=transform,
                ) as dst:
                    # Write the chip data to the destination file
                    dst.write(chip)