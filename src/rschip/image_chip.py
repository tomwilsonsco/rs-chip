from pathlib import Path
import rasterio as rio
from rasterio.windows import Window
import numpy as np
import pickle
import multiprocessing
import time


class ImageChip:
    """
    Split satellite images into smaller tiles or chips, with optional standard scaling.

    Attributes:
        input_image_path (Path): The path to the input satellite image.
        output_path (Path): The directory path where the output tiles will be saved.
        output_name (str): The stem name of each chip. Optional and defaults to image file name in `input_image_path`.
        pixel_dimensions (int): The height and width of each tile in pixels. Defaults to 128.
        offset (int): The offset used when creating tiles, i.e. the step size. Defaults to 64.
        use_multiprocessing (bool): Whether to use multiprocessing for chipping. Defaults to True.
        output_format (str): The format of the output files, either 'tif' or 'npz'.
                             If tif then tif file written per tile window. If npz then `batch_size` batches
                              of array tiles are written into one npz file. Defaults to tif.
        max_batch_size (int): The maximum number of tiles to process in a batch.
                             If multiprocessing is enabled, the actual batch size may be less. Defaults to 1000.
    """

    def __init__(
        self,
        input_image_path,
        output_path,
        output_name=None,
        pixel_dimensions=128,
        offset=64,
        use_multiprocessing=True,
        output_format="tif",
        max_batch_size=1000,
    ):
        self.input_image_path = Path(input_image_path)
        self.output_path = Path(output_path) if output_path else Path(input_image_path)
        self.output_name = output_name if output_name else Path(input_image_path).stem
        self.pixel_dimensions = pixel_dimensions
        self.offset = offset
        self.standard_scaler = None
        self.normaliser = None
        self.use_multiprocessing = use_multiprocessing
        self.output_format = output_format
        self.max_batch_size = max_batch_size

    def _generate_windows(self, src):
        """
        Generate sliding windows (tiles) across the input image.

        Yields the x and y coordinates of the top-left corner of each window and window of pixel dimensions size.

        Args:
            src: The source rasterio image object that is being split into windows.

        Yields:
            tuple: A tuple containing:
                - x (int): The x-coordinate of the bottom-left corner of the window.
                - y (int): The y-coordinate of the bottom-left corner of the window.
                - window (rasterio.windows.Window): A Window of the region of the image to be processed.
        """
        for y in range(0, src.height, self.offset):
            for x in range(0, src.width, self.offset):
                window = Window(x, y, self.pixel_dimensions, self.pixel_dimensions)
                yield x, y, window

    def _save_chip(self, chip, transform, output_file_path, d_type, src) -> None:
        """
        Save a chip (tile) to a GeoTIFF file.

        Args:
            chip (np.ndarray): The chip data to be written, typically a numpy array representing the pixel values.
            transform: The affine transformation to apply to the chip (from the window's position in the original image).
            output_file_path (Path): The path where the output chip will be saved.
            d_type: The data type of the chip (e.g., uint8, float32).
            src: The source raster file, used to retrieve metadata like CRS and number of bands.

        Returns:
            None: Writes the chip to a GeoTIFF file at the specified path.
        """
        with rio.open(
            output_file_path,
            "w",
            driver="GTiff",
            height=self.pixel_dimensions,
            width=self.pixel_dimensions,
            count=src.count,
            dtype=d_type,
            crs=src.crs,
            transform=transform,
            nodata=None,
        ) as dst:
            dst.write(chip)

    def _save_batch_as_npz(self, batch, batch_index) -> None:
        """
        Save a batch of chips as an NPZ file.

        Args:
            batch (dict): Dictionary containing image chips.
            batch_index (int): Index of the batch.

        Returns:
            None: Writes the batch to an NPZ file at the specified path.
        """
        output_file_path = self.output_path / f"batch_{batch_index}.npz"
        if output_file_path.exists():
            output_file_path.unlink()
        np.savez_compressed(output_file_path, **batch)

    def _output_file(self, x: int, y: int) -> Path:
        """
        Generate the output file path for a chip based on its x and y coordinates.

        Args:
            x (int): The x-coordinate of the bottom-left corner of the chip.
            y (int): The y-coordinate of the bottom-left corner of the chip.

        Returns:
            Path: The full path (as a `Path` object) where the chip will be saved, including the generated file name.
        """
        if self.output_name is None:
            output_file_name = f"{self.input_image_path.stem}_{x}_{y}.tif"
        else:
            output_name = self.output_name.replace(".tif", "")
            output_file_name = f"{output_name}_{x}_{y}.tif"
        return self.output_path / output_file_name

    def set_scaler(self, sample_size=10000, write_file=True, write_path=None):
        """Sets the standard scaler to be used for standard scaling each array when chipping.

        A specified number of pixels are sampled from the image set in `ImageChip.input_image_path`.
        This derives a per band mean and standard deviation, stored in a dictionary and used in `apply_scaler`.

        The dictionary is (optionally) written as a pickle file. The pickle file is a way to retrieve the scaler applied to
        images during training a model for making predictions from that model.

        Args:
            sample_size (int): The number of pixels to sample from the pre-chipped image to determine per band
            mean and standard deviation.
            write_file (bool): If True a pickle file is written containing the scaler dictionary.
            write_path (string): The directory and filename (.pkl) where the scaler will be written if `write_file`.
            None by default when the pickle file is written to the same dir as ImageChip.input_image_path with file name
            <image_file>_scaler_<sample_size>.pkl.
        """
        if self.normaliser is not None:
            print("normaliser will be set to None")
            self.normaliser = None
        self.standard_scaler = self.sample_to_scaler(sample_size=sample_size)
        if write_file:
            if not write_path:
                # Create the pickle file path
                pickle_file_name = (
                    f"{self.input_image_path.stem}_scaler_{sample_size}.pkl"
                )
                output_dir = Path(self.output_path)
                pickle_file_path = output_dir / pickle_file_name
            else:
                pickle_file_path = write_path
            # Save the dictionary to a pickle file
            with open(pickle_file_path, "wb") as f:
                pickle.dump(self.standard_scaler, f)
                print(f"Written scaler to {pickle_file_path}")

    def _validate_normaliser_inputs(self, value, name):
        """
        Validate if the input value is either a number or a list of numbers.

        Args:
            value (int, float, list): The value to be validated.
            name (str): The name of the value being validated.

        Raises:
            ValueError: If value is not valid.
        """
        with rio.open(self.input_image_path) as f:
            bands = f.profile["count"]
        if isinstance(value, list):
            if len(value) != bands:
                raise ValueError(
                    f"{name} list {value} must be the same length as the number of bands in the image ({bands})."
                )
            elif not all(isinstance(i, (int, float)) for i in value):
                raise ValueError(
                    f"{name} list {value} must only contain integer or float numbers."
                )
        elif isinstance(value, (int, float)):
            return [value] * bands
        else:
            raise ValueError(f"{name} must be either list, float, or int")
        return value

    def set_normaliser(
        self, min_val=None, max_val=None, write_file=True, write_path=None
    ):
        """Sets the min-max normaliser values to be used to min-max normalise each array when chipping.

        Sets the min and max values in a dictionary used by the `apply_normaliser` method. If single min and max
        values as input then these values are repeated in a list for each band in ImageChip.input_image_path image.

        The dictionary is (optionally) written as a pickle file. The pickle file is a way to retrieve the normaliser
        min max ranges applied to images during training a model for making predictions from that model.

        Args:
            min_val (int, float, or list): The minimum value to use for normalisation. If a list, should be
            a list with one value for each band in the image and the list values should be in the index order of the bands.
            max_val (int, float, or list): The maximum value to use for normalisation. If a list, should be
            a list with one value for each band in the image and the list values should be in the index order of the bands.
            write_file (bool): If True a pickle file is written containing the normaliser dictionary.
            write_path (string): The directory and filename (.pkl) where the scaler will be written if `write_file`.
            None by default when the file path is written to the same dir as ImageChip.input_image_path and file name
            <image_file>_scaler_<sample_size>.pkl.

        """
        if min_val is None or max_val is None:
            print(
                "normaliser set to None as both min_val and max_val need to be specified"
            )
            self.normaliser = None
            return

        # Validate min_val and max_val
        min_val = self._validate_normaliser_inputs(min_val, "min_val")
        max_val = self._validate_normaliser_inputs(max_val, "max_val")

        if self.standard_scaler is not None:
            print("standard_scaler will be set to None")
            self.standard_scaler = None

        self.normaliser = {"min_val": min_val, "max_val": max_val}
        print(
            f"Set normaliser as {self.normaliser}. There are {len(min_val)} min and max values, one for each band."
        )

        if write_file:
            if not write_path:
                # Create the pickle file path
                pickle_file_name = f"{self.input_image_path.stem}_normaliser.pkl"
                output_dir = Path(self.output_path)
                pickle_file_path = output_dir / pickle_file_name
            else:
                pickle_file_path = write_path
                # Save the dictionary to a pickle file
            with open(pickle_file_path, "wb") as f:
                pickle.dump(self.normaliser, f)
                print(f"Written normaliser to {pickle_file_path}")

    @staticmethod
    def apply_normaliser(array: np.ndarray, normaliser_dict: dict) -> np.ndarray:
        """Normalises a numpy array based on min and max values created by `set_normaliser`.

        Args:
            array (np.ndarray): A numpy array of shape (m, n, n), where m is the length of the first dimension.
            normaliser_dict (dict): A dictionary containing 'min_val' and 'max_val', which are
             lists of min and max values which must be of the same length as array first dimension.
            per band. The pixel values in each band are clipped to their corresponding min-max range and then normalised.


        Returns:
            np.ndarray: A normalised numpy array of the same shape as the input array.
        """
        min_vals = normaliser_dict["min_val"]
        max_vals = normaliser_dict["max_val"]
        if array.shape[0] != len(min_vals):
            raise ValueError(
                f"Array band dimension {array.shape[0]} does not match length of normaliser clip values."
            )
        normalised_array = np.zeros_like(array, dtype=np.float32)

        for i in range(array.shape[0]):
            min_val = min_vals[i]
            max_val = max_vals[i]
            # Apply normalising only to non-zero values (assuming 0 is nodata)
            mask = array[i, :, :] != 0
            clipped = np.clip(array[i, :, :], min_val, max_val)
            normalised_array[i, :, :] = np.where(
                mask, (clipped - min_val) / (max_val - min_val), 0
            )

        return normalised_array

    def sample_to_scaler(self, sample_size: int) -> dict:
        """
        Samples pixel values from an image at a sample of random coordinates and calculates the
        mean and standard deviation for each band.

        Args:
             sample_size (int): How many random points to sample pixel values for each band to derive mean and std dev.

        Returns:
            dict: A dictionary with keys as '<band_name>_mean' and '<band_name>_std'
            for each band in the image, and values as the calculated mean and
            standard deviation of the sampled pixel values.
        """
        stats_dict = {}

        with rio.open(self.input_image_path) as src:
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
                band_pixel_values = pixel_values[:, band_index]
                valid_band_pixel_values = band_pixel_values[
                    ~np.isnan(band_pixel_values)
                ]
                band_vals = {
                    "band_name": band_names[band_index],
                    "mean": np.mean(valid_band_pixel_values),
                    "std": np.std(valid_band_pixel_values),
                }
                stats_dict[band_index] = band_vals

        return stats_dict

    @staticmethod
    def apply_scaler(
        array: np.ndarray, scaler_dict: dict[int, dict[str, float]]
    ) -> np.ndarray:
        """Standard scales a numpy array based on mean and std values from a dictionary.

        Dict can be created from `sample_to_scaler`.

        Args:
            array (np.ndarray): A numpy array of shape (m, n, n), where m is the length of the first dimension.
            scaler_dict (dict): A dictionary containing 'mean' and 'std' for each band, keyed by the first dimension index.

        Returns:
            np.ndarray: A standard scaled numpy array of the same shape as the input array.
        """
        # Validate the number of bands against the number of band names provided
        if array.shape[0] != len(scaler_dict.keys()):
            raise ValueError(
                f"Expected {len(scaler_dict)} bands in scaler dict, but got {array.shape[0]} in the image array."
            )
        scaled_array = np.zeros_like(array, dtype=np.float32)
        for i in range(array.shape[0]):
            band_info = scaler_dict.get(i)
            mean = band_info["mean"]
            std = band_info["std"]
            # Apply scaling only to non-zero values (assuming 0 is nodata)
            mask = array[i, :, :] != 0
            scaled_array[i, :, :] = np.where(mask, (array[i, :, :] - mean) / std, 0)
        return scaled_array

    @staticmethod
    def unapply_scaler(scaled_array, scaler_dict) -> np.ndarray:
        """Performs the inverse of standard scaling on a numpy array.

        Note that the dtype of the array is not changed, so if for example converting back to
        an int16 type then this would have to be done separately.

        Args:
            scaled_array (np.ndarray): A standard scaled numpy array of shape (m, n, n), where m is the length of the
            first dimension.
            scaler_dict (dict): A dictionary containing 'mean' and 'std' for each band, keyed by the first dimension
            index.

        Returns:
            np.ndarray: A numpy array transformed back to its original scale, of the same shape as the input scaled array.
        """
        if scaled_array.shape[0] != len(scaler_dict.keys()):
            raise ValueError(
                f"Expected {len(scaler_dict)} bands in scaler dict, but got {scaled_array.shape[0]} in the image array."
            )
        unscaled_array = np.zeros_like(scaled_array)
        for i in range(scaled_array.shape[0]):
            band_info = scaler_dict.get(i, {"mean": 0, "std": 1})
            mean = band_info["mean"]
            std = band_info["std"]
            unscaled_array[i, :, :] = scaled_array[i, :, :] * std + mean
        return unscaled_array

    def _process_batch(self, batch_vals):
        """
        Process a batch of chips by reading the windows, scaling, and saving them.

        Args:
            batch_vals (tuple): Tuple of batch id and list of window, minx, miny to process in the batch.

        Returns:
            None
        """
        batch_id, batch = batch_vals
        out = {}
        with rio.open(self.input_image_path) as src:
            for x, y, window in batch:
                chip = src.read(window=window, boundless=True, fill_value=0)
                if self.standard_scaler:
                    chip = self.apply_scaler(chip, self.standard_scaler)
                if self.normaliser:
                    chip = self.apply_normaliser(chip, self.normaliser)

                if self.output_format == "tif":
                    output_file_path = self._output_file(x, y)
                    transform = src.window_transform(window)
                    self._save_chip(chip, transform, output_file_path, chip.dtype, src)
                elif self.output_format == "npz":
                    arr_name = f"{self.output_name}_{x}_{y}"
                    out[arr_name] = chip

        if self.output_format == "npz" and out:
            self._save_batch_as_npz(out, batch_id)

    def _calculate_batches(self, windows):
        """
        Calculate the optimal batch size and split windows into batches.

        Args:
            windows (list): List of windows to be processed.

        Returns:
            list: A list of tuples, where each tuple has an id and a list of windows and minx, miny window identifiers.
        """
        if self.use_multiprocessing:
            num_cores = multiprocessing.cpu_count() - 1  # leave one core free?
            batch_size = min(self.max_batch_size, max(1, len(windows) // num_cores))
        else:
            batch_size = self.max_batch_size

        print(f"Using batch size {batch_size}")

        num_batches = len(windows) // batch_size + (
            1 if len(windows) % batch_size != 0 else 0
        )
        batches = [
            (i, windows[i * batch_size : (i + 1) * batch_size])  # noqa: E203
            for i in range(num_batches)
        ]

        return batches

    def chip_image(self) -> None:
        """
        Split a satellite image into smaller tiles or chips.

        Method uses rasterio to read a satellite image, then splits the image into
        smaller square tiles of specified dimensions and saves them to the output path.

        The output tile file names are suffixed with x and y offsets and saved as TIFF files or NPZ files.
        Optionally the chip pixel values can be standard scaled before saving, using a sample of the full image pixels.

        Returns:
            None
        """
        print(f"Chipping {self.input_image_path} to {self.output_path}/")

        start_time = time.time()

        # Create output directory if it does not exist
        self.output_path.mkdir(parents=True, exist_ok=True)

        with rio.open(self.input_image_path) as src:
            windows = list(self._generate_windows(src))

        batches = self._calculate_batches(windows)

        if self.use_multiprocessing:
            print(f"Processing {len(batches)} batches in parallel.")
            num_cores = multiprocessing.cpu_count() - 1  # leave a core free?
            print(f"Using {num_cores} cores.")
            with multiprocessing.Pool(processes=num_cores) as pool:
                pool.map(self._process_batch, batches)
        else:
            print(f"Processing in {len(batches)} batches")
            for i, batch in enumerate(batches):
                self._process_batch(batch)
                print(f"Processed batch {i + 1} of {len(batches)}.")

        elapsed_time = time.time() - start_time
        print(f"Chipping completed in {elapsed_time:.2f} seconds.")
