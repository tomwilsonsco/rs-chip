from pathlib import Path
import rasterio as rio
import numpy as np
from typing import Optional


class RemoveBackgroundOnly:
    """
    Remove arrays where the segmentation mask class values show background only.

    Attributes:
        background_val (int): The value in the mask image array that represents the background class. Defaults to 0.
        non_background_min (int): The minimum number of non-background pixels required to retain a chip. Defaults to 1000.
    """

    def __init__(self, background_val: int = 0, non_background_min: int = 1000):
        self.background_val = background_val
        self.non_background_min = non_background_min

    @staticmethod
    def _prefix_checker(prefix: Optional[str]) -> str:
        return "" if prefix is None else prefix

    def _find_image_eq_mask(
        self,
        class_chip_dir: Path,
        image_chips_dir: str,
        masks_prefix: Optional[str],
        images_prefix: Optional[str],
    ) -> Path:
        image_chips_dir = Path(image_chips_dir)
        class_file = class_chip_dir.name
        image_file = class_file.replace(
            self._prefix_checker(masks_prefix), self._prefix_checker(images_prefix)
        )
        return image_chips_dir / image_file

    def _find_img_npz_eq_mask(self, class_npz_file: Path, image_npz_dir: str) -> Path:
        image_npz_dir = Path(image_npz_dir)
        class_file = class_npz_file.name
        return image_npz_dir / class_file

    def _find_img_key_from_mask_key(
        self, class_key: str, masks_prefix: Optional[str], images_prefix: Optional[str]
    ) -> str:
        img_key = class_key.replace(
            self._prefix_checker(masks_prefix), self._prefix_checker(images_prefix)
        )
        return img_key

    def check_background_only(self, class_arr: np.ndarray) -> bool:
        """
        Check if an image mask has more than the specified number of non-background pixels.

        Args:
            class_arr (numpy.ndarray): A 2D NumPy array representing the class labels for each pixel in an image mask.

        Returns:
            bool: True if the image mask has non-background pixel count < `non_background_min`. False otherwise.
        """
        return np.sum(class_arr != self.background_val) < self.non_background_min

    def remove_background_only_files(
        self,
        class_chips_dir: str,
        image_chips_dir: str,
        image_extn: str = "tif",
        masks_prefix: Optional[str] = None,
        images_prefix: Optional[str] = None,
    ) -> None:
        """
        Remove the chip files where the mask contains background only and no other classes.

        Args:
            class_chips_dir (str): Directory containing the chip mask image files to check.
            image_chips_dir (str): Corresponding chip image file directory - if mask is all background, image is removed too.
            image_extn (str, optional): The extension for the image files. Defaults to "tif".
            masks_prefix (str, optional): Prefix for mask files. Defaults to None. This prefix is removed when checking for
            equivalent mask to image file.
            images_prefix (str, optional): As `masks_prefix`. Prefix for image files. Defaults to None.

        Raises:
            FileNotFoundError: If no files with the specified extension are found in the input directory or if a file
            referenced by a mask does not exist.
        """
        class_chips_dir = Path(class_chips_dir)
        image_files = list(class_chips_dir.glob(f"**/*.{image_extn}"))
        if not image_files:
            raise FileNotFoundError(f"No {image_extn} files found in {class_chips_dir}")

        print(f"{len(image_files)} in {class_chips_dir} before.")

        for f in image_files:
            with rio.open(f) as src:
                img = src.read(1)
            if self.check_background_only(img):
                image_file = self._find_image_eq_mask(
                    f, image_chips_dir, masks_prefix, images_prefix
                )
                if not image_file.exists():
                    raise FileNotFoundError(
                        f"The image file {image_file} does not exist."
                    )
                image_file.unlink()
                f.unlink()

        image_files = list(class_chips_dir.glob(f"**/*.{image_extn}"))
        print(f"{len(image_files)} in {class_chips_dir} after.")

    def remove_background_only_npz(
        self,
        class_npz_dir: str,
        image_npz_dir: str,
        masks_prefix: Optional[str] = None,
        images_prefix: Optional[str] = None,
    ) -> None:
        """
        Remove arrays from NPZ files where the mask contains background only.

        Args:
            class_npz_dir (str): Directory containing the chip mask NPZ files to check.
            image_npz_dir (str): Corresponding chip image NPZ file directory - if mask is all background, image is removed too.
            masks_prefix (str, optional): Prefix for mask files. Defaults to None. This prefix is removed when checking for
            equivalent mask to image file.
            images_prefix (str, optional): As `masks_prefix`. Prefix for image files. Defaults to None.

        Raises:
            FileNotFoundError: If no NPZ files are found in the input directory.
        """
        class_npz_dir = Path(class_npz_dir)
        npz_files = list(class_npz_dir.glob("**/*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No npz files found in {class_npz_dir}")

        for f in npz_files:
            npz_class_dict = np.load(f)
            img_npz_file = self._find_img_npz_eq_mask(f, image_npz_dir)
            npz_image_dict = np.load(img_npz_file)

            out_class_dict = {}
            out_img_dict = {}

            print(f"{f.name} initially {len(npz_class_dict.keys())}...")
            for key in npz_class_dict.files:
                class_arr = npz_class_dict[key]
                if not self.check_background_only(class_arr):
                    out_class_dict[key] = class_arr
                    img_key = self._find_img_key_from_mask_key(
                        key, masks_prefix, images_prefix
                    )
                    out_img_dict[img_key] = npz_image_dict[img_key]
            print(f"{f.name} finally {len(out_class_dict.keys())}...")
            npz_class_dict.close()
            npz_image_dict.close()
            f.unlink()
            img_npz_file.unlink()

            if out_class_dict:
                np.savez(f, **out_class_dict)
                np.savez(img_npz_file, **out_img_dict)
            else:
                print(f"No valid entries left in {f.name}, deleting the NPZ file.")
