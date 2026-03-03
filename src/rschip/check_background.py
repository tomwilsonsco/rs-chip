from pathlib import Path
import rasterio as rio
import numpy as np
from typing import Optional, Union
import pandas as pd


class CheckBackgroundOnly:
    """
    Check arrays where the segmentation mask class values show background only.

    This class is used to identify image chips that contain only background based on
    their corresponding segmentation masks.

    Attributes:
        background_val (int | float): The value in the mask image array that represents the background class. Defaults to 0.
        non_background_min (int): The minimum number of non-background pixels required to consider a chip as not background-only. Defaults to 1.
    """

    def __init__(self, background_val: Union[int, float] = 0, non_background_min: int = 1):
        self.background_val = background_val
        self.non_background_min = non_background_min

    @staticmethod
    def _prefix_checker(prefix: Optional[str]) -> str:
        return "" if prefix is None else prefix

    def _find_image_eq_mask(
        self,
        class_chip_path: Path,
        image_chips_dir: str,
        masks_prefix: Optional[str],
        images_prefix: Optional[str],
    ) -> Path:
        image_chips_dir = Path(image_chips_dir)
        class_file = class_chip_path.name
        image_file = class_file.replace(
            self._prefix_checker(masks_prefix), self._prefix_checker(images_prefix)
        )
        return image_chips_dir / image_file

    def check_background_only(self, class_arr: np.ndarray) -> bool:
        """
        Check if an image mask has fewer than the specified number of non-background pixels.

        Args:
            class_arr (numpy.ndarray): A 2D NumPy array representing the class labels for each pixel in an image mask.

        Returns:
            bool: True if the image mask has a non-background pixel count < `non_background_min`. False otherwise.
        """
        return np.sum(class_arr != self.background_val) < self.non_background_min

    def check_background_chips(
        self,
        class_chips_dir: str,
        image_chips_dir: str,
        image_extn: str = "tif",
        masks_prefix: Optional[str] = None,
        images_prefix: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Checks chip files to identify which ones are background only and returns a DataFrame.

        This method iterates through mask chip files, checks if they are background-only,
        and writes the results to a CSV file named 'background_only_check.csv' in the
        `class_chips_dir`. The results are also returned as a pandas DataFrame.

        The CSV file and DataFrame will contain the following columns:
        - mask_file: The path to the mask chip file.
        - image_file: The path to the corresponding image chip file.
        - is_background_only: A boolean indicating if the mask is background-only.

        Args:
            class_chips_dir (str): Directory containing the chip mask image files to check.
            image_chips_dir (str): Corresponding chip image file directory.
            image_extn (str, optional): The extension for the image files. Defaults to "tif".
            masks_prefix (str, optional): Prefix for mask files. Defaults to None. This prefix is removed when checking for
            equivalent mask to image file.
            images_prefix (str, optional): As `masks_prefix`. Prefix for image files. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame with the check results.

        Raises:
            FileNotFoundError: If no files with the specified extension are found in the input directory.
        """
        class_chips_dir = Path(class_chips_dir)
        mask_files = list(class_chips_dir.glob(f"**/*.{image_extn}"))
        if not mask_files:
            raise FileNotFoundError(f"No {image_extn} files found in {class_chips_dir}")

        print(f"Checking {len(mask_files)} files in {class_chips_dir}.")

        audit_data = []
        for f in mask_files:
            with rio.open(f) as src:
                img = src.read(1)
            is_background = self.check_background_only(img)
            image_file = self._find_image_eq_mask(
                f, image_chips_dir, masks_prefix, images_prefix
            )
            audit_data.append(
                {
                    "mask_file": f,
                    "image_file": image_file,
                    "is_background_only": is_background,
                }
            )

        df = pd.DataFrame(audit_data)

        # Save to CSV
        output_csv_path = class_chips_dir / "background_only_check.csv"
        df.to_csv(output_csv_path, index=False)

        print(f"Check results written to {output_csv_path}")
        return df
