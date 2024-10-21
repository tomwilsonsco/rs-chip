# rschip

Split satellite images into small, fixed-sized tiles, for input into convolutional neural networks (cnn), [Segment Anything](https://arxiv.org/abs/2304.02643).

## Features

- **Tile Satellite Images**: Split large satellite images into smaller chips of specified dimensions for easier processing. Includes option to sample pixels for standard scaling and writing of scaler object for use in predictions.
- **Mask Segmentation**: Generate segmentation mask images from geopackage or shapefile features for supervised segmentation, e.g [U-Net](https://arxiv.org/abs/1505.04597).
- **Remove Background Chips**: Filter out image chips that contain only background. Useful for when preparing training datasets.

## Installation

To install rschip, clone the repository and install the package in editable mode:

```bash
pip install -e .
```

Requires `rasterio`, `numpy`, `geopandas`, and `shapely`. These will be automatically installed if using `pip`.

## Usage

### 1. ImageChip Class
The `ImageChip` class provides functionality for creating tiles or chips from large satellite images.

```python
from rschip.image_chip import ImageChip

# Initialize the ImageChip instance
image_chipper = ImageChip(
    input_image_path='path/to/large_image.tif',
    output_dir='path/to/output_directory',
    tile_size=128,
    offset=64
)

# Generate chips
image_chipper.create_chips()
```
The `output_format` parameter can either be `"tif"` or `"npz"` in which case arrays of specified batch sized are written into numpy zip (.npz) files.

`use_multiprocessing=True` (default) makes chipping process faster by using multiple cores. 

### 2. SegmentationMask Class
The `SegmentationMask` class is used to create a segmentation mask images from geopackage or shapefile using an input image as extent and pixel size reference.

```python
from rschip.segmentation_mask import SegmentationMask

# Initialize the SegmentationMask instance
seg_mask = SegmentationMask(
    input_image_path="path/to/image.tif",
    output_path="path/to/output_mask.tif",
    class_field="ml_class"
)

# Generate segmentation mask image
seg_mask.create_mask()
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
## License
This project is licensed under the MIT License - see the LICENSE file for details.
