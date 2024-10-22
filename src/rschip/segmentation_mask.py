import geopandas as gpd
import rasterio as rio
import rasterio.features
import numpy as np
from shapely.geometry import box


class SegmentationMask:
    """
    Create segmentation mask from polygon features to raster image extent.

    Attributes:
        input_image_path (str): Path to the input tif image.
        input_features_path (str): Path to the input features (shapefile or GeoPackage).
        output_path (str): Path where to create the output mask image.
        class_field (str): Attribute field name in input features that determines the pixel value.
        Defaults to 'ml_class'.
    """

    def __init__(
        self,
        input_image_path: str,
        input_features_path: str,
        output_path: str,
        class_field: str = "ml_class",
    ) -> None:
        """
        Initializes SegmentationMask with input image, features, output path, and class field.
        """
        self.input_image_path = input_image_path
        self.input_features_path = input_features_path
        self.output_path = output_path
        self.class_field = class_field

    def create_mask(self) -> None:
        """
        Creates the segmentation mask.
        """
        image_crs, image_bounds, image_transform, image_shape = (
            self._load_image_metadata()
        )
        input_features = self._load_and_validate_features(image_crs)
        input_features = self._clip_features_to_image(input_features, image_bounds)
        mask = self._rasterize_features(input_features, image_shape, image_transform)
        self._write_mask(mask)

    def _load_image_metadata(self):
        """
        Loads metadata from the input image.

        Returns:
            tuple: A tuple containing the CRS, bounds, transform, and shape of the image.
        """
        with rio.open(self.input_image_path) as src:
            image_crs = src.crs
            image_bounds = src.bounds
            image_transform = src.transform
            image_height = src.height
            image_width = src.width
            image_shape = (image_height, image_width)
        return image_crs, image_bounds, image_transform, image_shape

    def _load_and_validate_features(self, image_crs) -> gpd.GeoDataFrame:
        """
        Loads the input features and validates the class field.

        Args:
            image_crs (CRS): Coordinate reference system of the input image.

        Returns:
            GeoDataFrame: The reprojected input features if necessary.

        Raises:
            ValueError: If the class field does not exist or is not an integer field.
        """
        input_features = gpd.read_file(self.input_features_path)
        # Check if class_field exists and is integer
        if self.class_field not in input_features.columns:
            raise ValueError(
                f"The class_field '{self.class_field}' does not exist in input features."
            )
        if not np.issubdtype(input_features[self.class_field].dtype, np.integer):
            raise ValueError(
                f"The class_field '{self.class_field}' must be an integer field."
            )
        # Reproject input features if needed
        if input_features.crs != image_crs:
            input_features = input_features.to_crs(image_crs)
        return input_features

    def _clip_features_to_image(
        self,
        input_features: gpd.GeoDataFrame,
        image_bounds: rasterio.coords.BoundingBox,
    ) -> gpd.GeoDataFrame:
        """
        Clips the input features to the extent of the input image.

        Args:
            input_features (GeoDataFrame): The input features to be clipped.
            image_bounds (BoundingBox): The bounds of the input image.

        Returns:
            GeoDataFrame: The clipped input features.
        """
        # Use shapely.box to create extent of input image
        image_bbox = box(*image_bounds)
        input_features = gpd.clip(input_features, image_bbox)
        return input_features

    def _rasterize_features(
        self,
        input_features: gpd.GeoDataFrame,
        image_shape: (int, int),
        image_transform: rasterio.transform.Affine,
    ) -> np.ndarray:
        """
        Converts input features to raster to create the mask.

        Args:
            input_features (GeoDataFrame): The input features to be rasterized.
            image_shape (tuple): The shape of the input image (height, width).
            image_transform (Affine): The affine transform of the input image.

        Returns:
            ndarray: The rasterized mask as a NumPy array.
        """
        shapes = (
            (geom, value)
            for geom, value in zip(
                input_features.geometry, input_features[self.class_field]
            )
        )
        mask = rasterio.features.rasterize(
            shapes=shapes,
            out_shape=image_shape,
            transform=image_transform,
            fill=0,
            dtype=np.uint8,
        )
        return mask

    def _write_mask(self, mask: np.ndarray) -> None:
        """
        Writes the mask array to the output image file.

        Args:
            mask (ndarray): The mask array to be written.
        """
        with rio.open(self.input_image_path) as src:
            meta = src.meta.copy()
        meta.update({"count": 1, "dtype": "uint8", "compress": "lzw"})
        with rio.open(self.output_path, "w", **meta) as dst:
            dst.write(mask, 1)
            print(f"written {self.input_image_path}")
