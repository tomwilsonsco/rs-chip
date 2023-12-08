from pathlib import Path
import rasterio as rio
import numpy as np


def _prefix_checker(prefix):
    if prefix is None:
        return ""
    else:
        return prefix


def _find_image_eq_mask(class_chip_dir, image_chips_dir, masks_prefix, images_prefix):
    image_chips_dir = Path(image_chips_dir)
    class_file = class_chip_dir.name
    image_file = class_file.replace(
        _prefix_checker(masks_prefix), _prefix_checker(images_prefix)
    )
    return image_chips_dir / image_file


def remove_background_only(
    class_chips_dir,
    image_chips_dir,
    image_extn="tif",
    background_val=0,
    masks_prefix=None,
    images_prefix=None,
):
    """Remove the chip files where the mask contains background only and no other classes.

    Args:
        class_chips_dir (str): Directory containing the mask images to check and delete from.
        image_chips_dir (str): Corresponding images directory. The corresponding image for a mask
        will be deleted if the mask fails a check.
        image_extn (str, optional): The extension for the image files. Defaults to "tif".
        background_val (int, optional): Pixel value for background in the masks.
        A sum of all values not equal to the background value are used for the check. Defaults to 0.
        masks_prefix (str, optional): If image and mask file names are not identical
        what is the name of the mask file (aside from its index). Defaults to None.
        images_prefix (str, optional): Used in combination with `masks_prefix`.
        If image and mask file names are not identical what is the name of the image file
        (aside from its index). Default assumes identical file names therefore defaults to None.

    Returns:
        None

    Raises:
        FileNotFoundError: If no files with the specified extension are found in the input directory
        or if a file referenced by a mask does not exist.
    """
    class_chips_dir = Path(class_chips_dir)
    image_files = list(class_chips_dir.glob(f"**/*.{image_extn}"))
    if not image_files:
        raise FileNotFoundError(f"No {image_extn} files found in {class_chips_dir}")

    print(f"{len(image_files)} in {class_chips_dir} before.")

    for f in image_files:
        with rio.open(f) as src:
            img = src.read(1)
        if np.sum(img != background_val) == 0:
            image_file = _find_image_eq_mask(
                f, image_chips_dir, masks_prefix, images_prefix
            )
            if not image_file.exists():
                raise FileNotFoundError(f"The image file {image_file} does not exist.")
            image_file.unlink()
            f.unlink()

    image_files = list(class_chips_dir.glob(f"**/*.{image_extn}"))
    print(f"{len(image_files)} in {class_chips_dir} after.")
