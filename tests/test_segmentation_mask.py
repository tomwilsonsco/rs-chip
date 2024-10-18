from rschip import SegmentationMask
import tempfile
import pytest
from pathlib import Path
import rasterio as rio
import numpy as np


@pytest.fixture(scope="function")
def temp_output_dir():
    """
    Fixture to create a temporary directory for storing output files.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_run_mask_creator(temp_output_dir):
    """
    Test SegmentationMask class to create a mask from test image and features.
    """
    out_fp = temp_output_dir / "output_mask.tif"
    mask_creator = SegmentationMask(
        "tests/data/test_img.tif", "tests/data/test_features.gpkg", out_fp
    )
    mask_creator.create_mask()

    # Check if the output file was created
    assert out_fp.exists(), "Output mask file was not created."


def test_invalid_class_field(temp_output_dir):
    """
    Test SegmentationMask with an invalid class field that does not exist.
    """
    out_fp = temp_output_dir / "output_mask.tif"
    with pytest.raises(
        ValueError,
        match="The class_field 'invalid_field' does not exist in input features.",
    ):
        mask_creator = SegmentationMask(
            "tests/data/test_img.tif",
            "tests/data/test_features.gpkg",
            out_fp,
            class_field="invalid_field",
        )
        mask_creator.create_mask()


def test_non_integer_class_field(temp_output_dir):
    """
    Test SegmentationMask with a class field that is not an integer type.
    """
    out_fp = temp_output_dir / "output_mask.tif"
    with pytest.raises(
        ValueError,
        match="The class_field 'non_int_test' must be an integer field.",
    ):
        mask_creator = SegmentationMask(
            "tests/data/test_img.tif",
            "tests/data/test_features.gpkg",
            out_fp,
            class_field="non_int_test",
        )
        mask_creator.create_mask()


def test_mask_output_crs(temp_output_dir):
    """
    Test if the output mask has the same CRS as the input image.
    """
    out_fp = temp_output_dir / "output_mask.tif"
    mask_creator = SegmentationMask(
        "tests/data/test_img.tif", "tests/data/test_features.gpkg", out_fp
    )
    mask_creator.create_mask()

    # Check if the output file has the same CRS as the input image
    with rio.open("tests/data/test_img.tif") as src:
        expected_crs = src.crs
    with rio.open(out_fp) as mask_src:
        assert (
            mask_src.crs == expected_crs
        ), "CRS of the output mask does not match the input image."


def test_rasterized_mask_values(temp_output_dir):
    """
    Test if the rasterized mask contains expected values.
    """
    out_fp = temp_output_dir / "output_mask.tif"
    mask_creator = SegmentationMask(
        "tests/data/test_img.tif", "tests/data/test_features.gpkg", out_fp
    )
    mask_creator.create_mask()

    # Check if the output mask contains expected values
    with rio.open(out_fp) as mask_src:
        mask = mask_src.read(1)
        unique_values = np.unique(mask)
        assert (
            len(unique_values) > 1
        ), "The output mask should contain more than one unique value."
        assert 0 in unique_values, "The output mask should contain background value 0."
