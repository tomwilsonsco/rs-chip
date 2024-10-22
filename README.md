# rschip
![PyPI version](https://img.shields.io/pypi/v/rschip)
![License](https://img.shields.io/github/license/tomwilsonsco/rs-chip)
![Build Status](https://img.shields.io/github/actions/workflow/status/tomwilsonsco/rs-chip/main.yml?branch=main)
![codecov](https://codecov.io/github/tomwilsonsco/rs-chip/branch/main/graph/badge.svg?token=W27NY55T4B)

Split satellite images into smaller fixed-sized tiles, for input into convolutional neural networks (cnn), or vision 
transformers (ViT) such as [Segment Anything](https://arxiv.org/abs/2304.02643).

## Features

- **Tile Satellite Images**: Split large satellite images into smaller chips of specified dimensions. Includes option to 
  sample pixels for standard scaling and write a scaler object to use when making predictions from a trained model.
- **Mask Segmentation**: Generate segmentation mask images from geopackage or shapefile features for supervised 
  segmentation, e.g using [U-Net](https://arxiv.org/abs/1505.04597).
- **Remove Background Chips**: Filter out image chips containing only background. Useful for when preparing training 
  and testing datasets.

## Installation

Install rschip with pip:

```bash
pip install rschip
```

Requires `rasterio`, `numpy`, `geopandas`, and `shapely`.

## Usage

### 1. ImageChip Class
The `ImageChip` class provides functionality for creating tiles (also known as chips) from large satellite images.

```python
from rschip import ImageChip

# Initialize the ImageChip instance for 128 by 128 tiles
image_chipper = ImageChip(
    input_image_path="path/to/large_image.tif",
    output_path="path/to/output_directory_image",
    pixel_dimensions=128,
    offset=64,
    output_format="tif",
)

# Generate chips
image_chipper.chip_image()
```
With the `output_format` parameter set to `"tif"`, each resulting tile is named using a suffix that represents the bottom left `(x, y)`
pixel coordinate position. If output_format is set to `"npz"`, the resulting .npz zip file contains a dictionary of arrays, 
where the keys are the same as these tile names. By default, the prefix of each tile name is taken from the input image file name 
(`input_image_path`), unless you specify `output_name`.

Using the parameter `use_multiprocessing=True` (default) makes chipping process faster by using multiple cores. 

### 2. SegmentationMask Class
The `SegmentationMask` class is used to create a segmentation mask images from geopackage or shapefile using an input image as extent and pixel size reference.

Once the segmentation mask has been created, the segmentation image can also be split into tiles. Some deep learning 
frameworks expect images and corresponding masks to have the same file name in separate directories. The `output_name` 
argument of ImageChip can ensure this is the case.

```python
from rschip import SegmentationMask, ImageChip

# Initialize the SegmentationMask
seg_mask = SegmentationMask(
    input_image_path="path/to/large_image.tif",
    input_features_path="path/to/geopackage_features.gpkg",
    output_path="path/to/output_mask.tif",
    class_field="ml_class"
)

# Generate segmentation mask image
seg_mask.create_mask()

# Chip the segmentation image to match satellite image
image_chipper = ImageChip(
    input_image_path="path/to/output_mask.tif",
    output_dir="path/to/output_directory_mask",
    output_name="large_image",
    pixel_dimensions=128,
    offset=64,
    output_format="tif",
)
image_chipper.chip_image()
```

### 3. RemoveBackgroundOnly Class
The `RemoveBackgroundOnly` class provides functionality to remove image chips (either could be tifs or numpy arrays inside npz file) that contain only background. Filtering out images only containing background helps to prepare a dataset more suitable for training models.
 
```python
from rschip.remove_background_only import RemoveBackgroundOnly

# Initialize the RemoveBackgroundOnly instance
remover = RemoveBackgroundOnly(background_val=0, non_background_min=100)

# Remove chips with only background
remover.remove_background_only_files(
    class_chips_dir='path/to/mask_directory',
    image_chips_dir='path/to/image_directory'
)
```
The default assumption is that image and mask equivalent have the same file names as shown in example 2. above. If that is
not the case, use the `masks_prefix`, `images_prefix` arguments which are prefix strings which are removed on checking for
image to mask equivalent using the bottom left (x,y) indices found in the outputs generated by `ImageChip.create_chips()`.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
