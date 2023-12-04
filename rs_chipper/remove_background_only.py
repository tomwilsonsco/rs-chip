from pathlib import Path
import rasterio as rio
import numpy as np
import warnings


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
    class_chips_dir = Path(class_chips_dir)
    image_files = list(class_chips_dir.glob(f"**/*.{image_extn}"))
    print(f"{len(image_files)} in {class_chips_dir} before.")

    for f in image_files:
        with rio.open(f) as src:
            img = src.read(1)
        if np.sum(img != background_val) == 0:
            image_file = _find_image_eq_mask(
                f, image_chips_dir, masks_prefix, images_prefix
            )
            if image_file.exists():
                image_file.unlink()
            else:
                warnings.warn(f"Cannot delete image corresponding to {f}")
            f.unlink()

    image_files = list(class_chips_dir.glob(f"**/*.{image_extn}"))
    print(f"{len(image_files)} in {class_chips_dir} after.")
