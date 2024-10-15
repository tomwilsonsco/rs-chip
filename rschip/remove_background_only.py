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


def _find_img_npz_eq_mask(class_npz_file, image_npz_dir):
    image_npz_dir = Path(image_npz_dir)
    class_file = class_npz_file.name
    return image_npz_dir / class_file


def _find_img_key_from_mask_key(class_key, masks_prefix, images_prefix):
    img_key = class_key.replace(
        _prefix_checker(masks_prefix), _prefix_checker(images_prefix)
    )
    return img_key


def check_background_only(class_arr, background_val=0, non_background_min=1000):
    """
    Check if an image mask has more than specified number of non-background pixels.


    Args:
        class_arr (numpy.ndarray): A 2D NumPy array representing the class labels for
        each pixel in an image mask. Each element in the array corresponds to a class label.
        background_val (int, optional): The value in `class_arr` that represents the
        background class. Defaults to 0.
        non_background_min (int, optional): The minimum number of non-background pixels
        required for check to return False. Defaults to 1000.

    Returns:
        bool: True if the image mask has
        non-background pixel count < than `non_background_min`. False otherwise.
    """
    if np.sum(class_arr != background_val) < non_background_min:
        return True
    else:
        return False


def remove_background_only_files(
    class_chips_dir,
    image_chips_dir,
    image_extn="tif",
    background_val=0,
    non_background_min=1000,
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
        non_background_min (int, optional): How many non-background pixels must be found in a chip
        image in order to retain it. Defaults to 1000.
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
        if check_background_only(img, background_val, non_background_min):
            image_file = _find_image_eq_mask(
                f, image_chips_dir, masks_prefix, images_prefix
            )
            if not image_file.exists():
                raise FileNotFoundError(f"The image file {image_file} does not exist.")
            image_file.unlink()
            f.unlink()

    image_files = list(class_chips_dir.glob(f"**/*.{image_extn}"))
    print(f"{len(image_files)} in {class_chips_dir} after.")


def remove_background_only_npz(
    class_npz_dir,
    image_npz_dir,
    background_val=0,
    non_background_min=1000,
    masks_prefix=None,
    images_prefix=None,
):
    class_npz_dir = Path(class_npz_dir)
    npz_files = list(class_npz_dir.glob(f"**/*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No npz files found in {class_npz_dir}")

    out_class_dict = {}
    out_img_dict = {}

    for f in npz_files:
        npz_class_dict = np.load(f)
        img_npz_file = _find_img_npz_eq_mask(f, image_npz_dir)
        npz_image_dict = np.load(img_npz_file)
        print(f"{f.name} initially {len(npz_class_dict.keys())}...")
        for key in npz_class_dict.files:
            class_arr = npz_class_dict[key]
            if not check_background_only(class_arr, background_val, non_background_min):
                out_class_dict[key] = class_arr
                img_key = _find_img_key_from_mask_key(key, masks_prefix, images_prefix)
                out_img_dict[img_key] = npz_image_dict[img_key]
        print(f"{f.name} finally {len(out_class_dict.keys())}...")
        npz_class_dict.close()
        npz_image_dict.close()
        f.unlink()
        img_npz_file.unlink()
        np.savez(f, **out_class_dict)
        np.savez(img_npz_file, **out_img_dict)
